"""
mutation_rerun_rules.py
===============
Brings existing output up to the current detection code -- all five
branches (mutations, prep, plural, numeral, quantifier) -- without
re-transcribing (Whisper) or re-calling Cysill.

Full rerun (folders processed from 2026-09-29 on)
-------------------------------------------------
Each run now saves its tagged word stream (tagged_*.json.gz: the words after
Whisper filters, contraction splitting, Cysill, spaCy and lexicon tagging,
before detection). A rerun feeds it through corpus_ops.rows_from_enriched()
-- the very function a normal run uses for detection -- so rerun output is
exactly what processing the video today would give, minus the tagging.
What a rerun picks up: anything in the detection branches, their tables,
mutation_engine's detection code, and lemma lookups. What it can't: changes
before detection (Whisper and its filters, contraction splitting, Cysill,
spaCy, Bangor-lexicon tags) -- those need the video processed again.

A folder that output_merge.py stacked from several runs holds one tagged
file per run whose rows it kept; they are stacked again the same way. Rows
from runs that saved no tagged file stay as they are (and keep their
pipeline_version). Every rerun row carries pipeline_version = the current
detection code and tagging_version = the code that tagged its words.

Dry run (default): per folder and branch, what would change (rows added,
removed, changed, and status changes such as erosion -> correct_mutation),
and rerun_candidate_<file>.csv next to each file that would change. With
--commit the files are replaced. Manual reviews are carried to the matching
new row (same rule, trigger, target and status within 1s); a reviewed row
whose result changed is not overwritten silently -- it goes to
superseded_reviews_mutations.csv for a fresh look. A stale corroborated
file is regenerated from the folder's captions, or renamed stale_*.

    python mutation_rerun_rules.py                          # dry run, every folder
    python mutation_rerun_rules.py --branch prep,numeral    # only these branches
    python mutation_rerun_rules.py --video davies --commit  # apply to matching folders

Folders from before 2026-09-29 have no tagged file. Process them once more
to enable full reruns; until then the legacy mode below still re-runs the
mutation branch for them when --trigger/--rule is given.

Legacy mode: mutation rules only, rebuilt from pos_*.csv
--------------------------------------------------------
Re-run mutation-rule detection on already-transcribed videos, without
re-transcribing (Whisper) or re-hitting Cysill -- for when you've changed
a rule in mutation_engine.py or a table in mutation_tables.py and want to
see its effect on existing corpus data, either everywhere or scoped to
specific trigger words / rule types.

Why this needs to re-parse spaCy rather than just reload the saved CSVs
-------------------------------------------------------------------------
process_comprehensive_mutations() runs on word dicts that carry a full
spacy_token sub-dict (dep, head, head_dep, head_pos, head_verbform, full
morph). The saved pos_*.csv only keeps flattened columns (spacy_dep,
spacy_pos, spacy_mutation, spacy_gender) -- head_verbform in particular
is gone, and that's the field the yn+verb-noun soft-mutation exemption
depends on to tell a finic verb from a verb-noun. Reloading those flat
columns alone would silently degrade exactly the rules most likely to
need re-running. spaCy is local and fast (unlike Whisper or Cysill), so
this re-parses each cached segment's text fresh via the same
parse_spacy_doc() + align_with_gap_tolerance() the main pipeline uses,
and reuses everything else (cysill_pos, cysill_mutation_type,
cysill_gender, confidence, timestamps) straight from the cached
pos_*.csv. Cysill itself is never called -- lemma lookups
(get_welsh_lemma) still happen at evaluation time same as a normal run,
but they hit the warm lemma_cache.json first and only reach out to
Cysill/simplemma for genuinely new words.

Output is always written to a separate rerun_candidate_*.csv file next
to the real mutations CSV -- nothing touches the real file unless you
pass --commit, and even then:
  - only rows matching your --trigger/--rule filter are considered for
    replacement (everything else in the real CSV is left byte-identical)
  - rows with manual_reviewed == True are NEVER overwritten, regardless
    of filter match. If the new rule output disagrees with a row a human
    already reviewed, it's listed in the printed diff as "needs manual
    re-check" instead of being silently applied.

Row identity across old/new output is matched on
(video_title, timestamp, trigger_word, following_word) -- the best
stable-ish key available given the CSV schema. This is best-effort: a
rule change that alters how many words get consumed by a trigger (rare)
can shift this key and show up as a spurious add/remove pair rather than
a clean "changed" row. Worth a skim of the diff before --commit on a rule
change that touches consumption logic, not just classification.

Usage
-----
    python mutation_rerun_rules.py --trigger yn                     # dry run, all videos
    python mutation_rerun_rules.py --trigger yn,ei --rule word_trigger
    python mutation_rerun_rules.py --rule phantom_check --video Diddymur_Senedd
    python mutation_rerun_rules.py --trigger yn --commit             # apply after reviewing the diff
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mutation_engine import (
    normalize_word, process_comprehensive_mutations,
    align_with_gap_tolerance, load_lemma_cache, save_lemma_cache,
)
# PATCH: previously redefined MUT_DIR/TRANS_DIR locally here (BASE_DIR /
# "mutations", BASE_DIR / "transcriptions") instead of importing the real
# ones -- harmless while both definitions matched, but a second place
# either could silently drift from corpus_io.py's own. Importing directly
# now.
from corpus_io import BASE_DIR, MUT_DIR, TRANS_DIR
from spacy_tagging import load_spacy, parse_spacy_doc

# Columns manual_editing.py owns on a mutation row. Never overwritten by
# --commit, and their presence (manual_reviewed == True specifically) is
# what protects a row from replacement at all.
MANUAL_REVIEW_COLUMNS = [
    "manual_reviewed", "is_erosion_original", "flagged",
    "flag_note", "review_count", "review_log",
]

JOIN_KEY_COLUMNS = ["video_title", "timestamp", "trigger_word", "following_word"]


# ========================= FULL RERUN (all branches) =========================
import io
from collections import Counter

from corpus_io import RUNS_DIR
import output_merge as om

# --branch name -> output file key (corpus_io._video_slug / output_merge.DATA_FILES)
BRANCHES = {"mutations": "mutations", "prep": "prep_mutations", "plural": "plural_mutations",
            "numeral": "numeral_mutations", "quantifier": "quantifier_mutations"}
TRANSCRIPT_KEYS = ("segments", "words", "lemmas", "pos")
OUTPUT_KEYS = list(om.DATA_FILES)      # the order rows_from_enriched() returns them in
# what identifies "the same row" in each detection file
ROW_KEYS = {
    "mutations": ["timestamp", "trigger_word", "following_word", "rule"],
    "prep_mutations": ["timestamp", "preposition", "following_pronoun"],
    "plural_mutations": ["timestamp", "trigger_word", "following_word"],
    "numeral_mutations": ["timestamp", "numeral_surface", "following_word"],
    "quantifier_mutations": ["timestamp", "quantifier_surface", "following_word"],
}
# bookkeeping columns, not results -- a difference in these isn't a change
NOT_COMPARED = {"pipeline_version", "tagging_version", "video_title", "video_url", "source",
                "video_duration_seconds", "video_word_count", "video_codeswitch_word_count",
                "speaker", *MANUAL_REVIEW_COLUMNS, "is_erosion"}
CANDIDATE_PREFIX = "rerun_candidate_"


def _text_frame(rows):
    """Rows as the text a CSV of them holds -- the form existing files are
    read in -- so old and new values compare like for like."""
    if not rows:
        return None
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return pd.read_csv(io.StringIO(buffer.getvalue()), dtype=str, keep_default_na=False)


def _attach_speakers(frames, data):
    """Siarad/Patagonia rows carry the speaker of their utterance (added by
    corpus_siarad.py after detection, so not in rows_from_enriched's output)."""
    speakers = [s[5] if len(s) > 5 else None for s in data["segments"]]
    if not any(speakers):
        return
    by_start = {}
    for w in data["words"]:
        seg_id, start = w.get("_seg_id"), w.get("start")
        if seg_id is not None and start is not None and 0 <= seg_id < len(speakers):
            by_start[round(float(start), 3)] = speakers[seg_id]
    for key, df in frames.items():
        if key == "segments":
            if len(speakers) == len(df):        # one row per segment, same order
                df["speaker"] = speakers
        else:
            df["speaker"] = om._start_times(df, key).round(3).map(by_start).fillna("")


def regenerate_folder(folder):
    """Every output file of `folder`, rebuilt by the current detection code
    from its saved tagged word stream(s), stacked like output_merge stacks
    runs, with the folder's rows that have no tagged file kept as they are.
    Returns (frames, #segments kept without tagging, tagged files used), or
    None when the folder has no tagged file."""
    caches = sorted(Path(folder).glob("tagged_*.json.gz"))
    if not caches:
        return None
    from corpus_ops import rows_from_tagged_cache   # heavy import, only when needed
    parts = []
    for path in caches:
        rows, data = rows_from_tagged_cache(path)
        frames = {k: f for k, f in zip(OUTPUT_KEYS, map(_text_frame, rows)) if f is not None}
        _attach_speakers(frames, data)
        window = data.get("window")
        if not window:
            spans = om._segment_spans(frames.get("segments"))
            window = [min(a for a, _ in spans), max(b for _, b in spans)] if spans else [0, 0]
        parts.append((data.get("tagged_at") or "", frames, om._union([window]), data))
    parts.sort(key=lambda p: p[0], reverse=True)          # newest run first

    _, current, coverage, newest = parts[0]
    counts = {c: om._to_float(newest.get(c)) for c in om.COUNT_COLUMNS}
    title = (newest.get("video_meta") or {}).get("title")
    for _, frames, window, _ in parts[1:]:
        kept, _, _, _ = om._select_old(current, frames, coverage)
        om.absorb(current, counts, frames, kept)
        coverage = om._union(coverage + window)

    # the folder's own rows, oldest of all: only what no tagged run covers survives
    existing = om._load(folder)
    existing_coverage, _ = om.read_coverage(folder, existing)
    kept, _, n_untagged, _ = om._select_old(current, existing, coverage)
    om.absorb(current, counts, existing, kept)
    coverage = om._union(coverage + existing_coverage)
    return om.finalize(current, coverage, counts, title), n_untagged, len(caches)


def diff_rows(key, old_df, new_df):
    """(added, removed, changed, reviewed-and-changed, status changes) between
    the existing and regenerated rows of one detection file."""
    cols = ROW_KEYS[key]

    def keyed(df):
        out = {}
        if df is not None:
            for _, r in df.iterrows():
                out.setdefault(tuple(str(r.get(c, "")) for c in cols), r)
        return out

    old_rows, new_rows = keyed(old_df), keyed(new_df)
    compare = [c for c in (new_df.columns if new_df is not None else [])
               if old_df is not None and c in old_df.columns and c not in NOT_COMPARED]
    added = [k for k in new_rows if k not in old_rows]
    removed = [k for k in old_rows if k not in new_rows]
    changed, reviewed, statuses = [], [], Counter()
    for k in new_rows.keys() & old_rows.keys():
        old, new = old_rows[k], new_rows[k]
        if all(str(old.get(c, "")) == str(new.get(c, "")) for c in compare):
            continue
        is_reviewed = str(old.get("manual_reviewed", "")).strip().lower() == "true"
        (reviewed if is_reviewed else changed).append(k)
        if old.get("status") != new.get("status"):
            statuses[(old.get("status"), new.get("status"))] += 1
    for k in added:
        statuses[(None, new_rows[k].get("status"))] += 1
    for k in removed:
        statuses[(old_rows[k].get("status"), None)] += 1
    return added, removed, changed, reviewed, statuses


def _folder_paths(folder, keys):
    """Each key's existing file in `folder`, or the name the pipeline would give it."""
    folder = Path(folder)
    segments = om._find_file(folder, "segments")
    folder_name = segments.stem[len("segments_"):] if segments else folder.name
    return {k: om._find_file(folder, om.DATA_FILES[k])
               or folder / f"{om.DATA_FILES[k]}_{folder_name}.csv" for k in keys}


def _without_kept_reviews(original, new):
    """Drops from `original` the reviewed rows that are already in `new` with
    their review -- rows kept as they were (no tagged file) -- so they aren't
    handed on a second time and reported as unmatched."""
    key_cols = list(om.REVIEW_KEY)
    if original is None or original.empty or new is None or new.empty \
            or "manual_reviewed" not in new.columns \
            or not set(key_cols) <= set(new.columns) or not set(key_cols) <= set(original.columns):
        return original
    kept = {tuple(str(v) for v in row) for row in
            new.loc[om._is_true(new["manual_reviewed"]).values, key_cols].itertuples(index=False)}
    if not kept:
        return original
    keys = original[key_cols].astype(str).apply(tuple, axis=1)
    return original[~keys.isin(kept).values]


def _refresh_corroborated(folder, mutations_path):
    """After the mutation file changed, its corroborated copy is stale:
    regenerated from the folder's caption track when there is one, else
    renamed stale_* so no tool reads it as current."""
    stale = om._find_file(folder, om.CORROBORATED_PREFIX)
    if stale is None:
        return
    captions = [v.with_suffix(".csv") for v in sorted(Path(folder).glob("*.vtt"))
                if v.with_suffix(".csv").exists()]
    if captions:
        try:
            import mutation_captions
            import spacy_tagging
            cap_kind = om._first(om._read_csv(stale), "caption_kind") or None
            nlp = spacy_tagging.SPACY_NLP if load_spacy() else None
            mutation_captions.run_corroboration(mutations_path, captions[0], nlp=nlp,
                                                cap_kind=cap_kind, output_path=stale)
            return
        except (Exception, SystemExit) as e:
            tqdm.write(f"  ⚠️ Couldn't redo caption corroboration ({e}).")
    stale.replace(stale.with_name("stale_" + stale.name))
    tqdm.write(f"  {stale.name} no longer matches the mutation file -- renamed stale_{stale.name}.")


def rerun_folder(folder, keys, rewrite_transcripts, commit):
    """Full rerun of one folder. Returns a per-file summary, or None when the
    folder has no tagged file."""
    result = regenerate_folder(folder)
    if result is None:
        return None
    frames, n_untagged, n_caches = result
    paths = _folder_paths(folder, OUTPUT_KEYS)
    existing = om._load(folder)
    summary = {"untagged_segments_kept": n_untagged, "tagged_files": n_caches, "files": {}}
    for key in keys:
        added, removed, changed, reviewed, statuses = diff_rows(
            key, existing.get(key), frames.get(key))
        summary["files"][key] = {"added": len(added), "removed": len(removed),
                                 "changed": len(changed), "reviewed_changed": len(reviewed),
                                 "statuses": statuses}

    differs = [k for k in keys if any(v for n, v in summary["files"][k].items()
                                      if n != "statuses")]
    if not commit:
        for key in differs:
            if frames.get(key) is not None:
                target = paths[key].with_name(CANDIDATE_PREFIX + paths[key].name)
                frames[key].to_csv(target, index=False, encoding="utf-8-sig", quoting=1)
        return summary

    to_write = list(keys) + (list(TRANSCRIPT_KEYS) if rewrite_transcripts else [])
    out = {}
    for key in to_write:
        new = frames.get(key)
        if new is None:                          # no rows any more: header only
            old = existing.get(key)
            if old is None:
                continue
            new = old.iloc[0:0]
        out[key] = new
    if "mutations" in out:
        corroborated_path = om._find_file(folder, om.CORROBORATED_PREFIX)
        corroborated = om._read_csv(corroborated_path) if corroborated_path else None
        original, stray = om._overlay_reviews(existing.get("mutations"), corroborated)
        original = _without_kept_reviews(original, out["mutations"])
        out["mutations"], stray_new = om._transfer_reviews(out["mutations"], original)
        om.save_stray_reviews(folder, [stray, stray_new])
    om.write_frames(out, {k: paths[k] for k in out})
    for key in out:
        candidate = paths[key].with_name(CANDIDATE_PREFIX + paths[key].name)
        candidate.unlink(missing_ok=True)
    if "mutations" in out:
        _refresh_corroborated(folder, paths["mutations"])
    return summary


def _print_summary(folder, summary, commit):
    lines = []
    for key, s in summary["files"].items():
        if not any(v for n, v in s.items() if n != "statuses"):
            continue
        moves = ", ".join(f"{a or 'new'}→{b or 'gone'}: {n}"
                          for (a, b), n in s["statuses"].most_common(6))
        lines.append(f"    {key}: +{s['added']} -{s['removed']} ~{s['changed']}"
                     + (f", {s['reviewed_changed']} reviewed row(s) changed" if s["reviewed_changed"] else "")
                     + (f"  [{moves}]" if moves else ""))
    head = f"  {folder.parent.name}/{folder.name}"
    if summary["untagged_segments_kept"]:
        head += (f"  ({summary['untagged_segments_kept']} segment(s) with no tagged file "
                 f"kept as they were)")
    if lines:
        tqdm.write(head + (" -- applied" if commit else ""))
        for line in lines:
            tqdm.write(line)


def _mutations_dir_for(mutations_csv_path):
    """The transcriptions/<stamp>/<slug>/ folder matching a given
    mutations/<stamp>/<slug>/mutations_<stamp>_<slug>.csv path.

    # PATCH: updated for _video_slug()'s restored run-then-video nesting --
    # mutations_csv_path now has two parent levels to read off (slug, then
    # stamp above it), not one flat <stamp>_<slug> folder. Falls back to a
    # BASE_DIR-rooted rglob for anything from before this fix (older flat
    # layout, or the layout before that), same pattern as
    # mutation_captions.py/mutation_manual_editing.py's find_segments_csv.
    """
    slug  = mutations_csv_path.parent.name
    stamp = mutations_csv_path.parent.parent.name
    candidate = TRANS_DIR / stamp / slug
    if candidate.exists():
        return candidate

    for prefix in ("mutations_original_", "mutations_corroborated_", "mutations_"):
        if mutations_csv_path.stem.startswith(prefix):
            folder_name = mutations_csv_path.stem[len(prefix):]
            break
    else:
        folder_name = mutations_csv_path.stem
    matches = list(BASE_DIR.rglob(f"pos_{folder_name}.csv"))
    return matches[0].parent if matches else candidate


def rebuild_words_only(pos_df):
    """
    Reconstruct the word-dict list process_comprehensive_mutations()
    expects, from a cached pos_*.csv -- reusing cached Cysill fields
    as-is, but re-parsing spaCy fresh per segment so head_verbform/morph
    survive (see module docstring).
    """
    load_spacy()
    words_only = []

    # Group rows into contiguous runs sharing the same segment_text --
    # NOT a groupby (a repeated identical segment_text elsewhere in the
    # video must not get merged with this one; contiguity in original
    # row order is what actually defines a segment here).
    for _, group in itertools.groupby(
            pos_df.to_dict("records"), key=lambda r: r["segment_text"]):
        seg_rows = list(group)
        segment_text = seg_rows[0]["segment_text"] or ""

        chunk = [{"word": r["word"], "synthetic": False} for r in seg_rows]
        spacy_raw     = parse_spacy_doc(segment_text) or []
        spacy_aligned = align_with_gap_tolerance(spacy_raw, chunk)

        for r, (_, spacy_tok) in zip(seg_rows, spacy_aligned):
            words_only.append({
                "word":                 r["word"],
                "start":                r.get("word_start"),
                "end":                  r.get("word_end"),
                "confidence":           r.get("confidence", 0.0),
                "cysill_pos":           r.get("cysill_pos"),
                "cysill_mutation_type": r.get("cysill_mutation_type"),
                "cysill_gender":        r.get("cysill_gender"),
                "gender":               r.get("gender_unified"),
                # Bangor-lexicon noun features (pos_*.csv from 2026-09-26 on).
                # An empty cell reads back as NaN, which is truthy and would
                # beat the spaCy fallback -- only a real string counts.
                "lex_gender":           r.get("lex_gender") if isinstance(r.get("lex_gender"), str) else None,
                "lex_number":           r.get("lex_number") if isinstance(r.get("lex_number"), str) else None,
                "spacy_token":          spacy_tok,
                "synthetic":            False,
                # PATCH: carried through from the cached pos_*.csv (see
                # corpus_ops.py's pos_rows construction) so a rerun still
                # correctly excludes Bangor-lexicon/code-switch-resolved
                # words from Cysill corroboration credit in
                # compute_confidence()/_build_row() -- without this, a
                # rerun would silently default every row to "not locally
                # resolved" (r.get() on a missing/older-cache column
                # returns None), which happens to be the SAFE direction
                # (undercounts corroboration rather than overcounts it)
                # but still loses real signal that's available and
                # should be used when the cache actually has it.
                "locally_resolved":    bool(r.get("locally_resolved")),
                # PATCH: pos_*.csv never stored a raw "cysill_aligned"
                # flag at all -- without this, EVERY rerun (regardless of
                # the locally_resolved fix above) would see
                # target_node.get("cysill_aligned") come back None for
                # every row, collapsing all_three/cysill+heuristic credit
                # to near-zero across the board, not just for locally-
                # resolved rows. cysill_pos is the right proxy to
                # reconstruct it from: parse_pos_result_extended() always
                # populates "pos" for a genuine live Cysill answer, and
                # it's left None for every locally-resolved row by
                # design -- so "cysill_pos is not None" reproduces the
                # original cysill_aligned semantics exactly, with the
                # locally-resolved contamination already excluded rather
                # than needing a second check.
                "cysill_aligned":      r.get("cysill_pos") is not None,
            })

    return words_only


def rerun_one_video(mutations_csv_path, triggers, rules):
    trans_dir = _mutations_dir_for(mutations_csv_path)
    pos_csv   = trans_dir / f"pos_{mutations_csv_path.stem.split('mutations_', 1)[-1]}.csv"
    if not pos_csv.exists():
        tqdm.write(f" ⚠️ No cached pos CSV for {mutations_csv_path.name} "
                    f"(expected {pos_csv}) -- skipping, can't rerun without it.")
        return None

    pos_df = pd.read_csv(pos_csv, keep_default_na=False, na_values=[""])
    old_df = pd.read_csv(mutations_csv_path, keep_default_na=False, na_values=[""])

    words_only = rebuild_words_only(pos_df)
    fresh_rows = process_comprehensive_mutations(words_only)

    # PATCH (Phase 3): channel_register removed from the schema entirely --
    # see corpus_io.py's CURATED_CHANNELS and corpus_analyzer.py's
    # formality-based figures, which replace the old register grouping.
    meta_cols = ["video_title", "video_url", "source",
                 "video_duration_seconds"]
    meta = {c: old_df.iloc[0][c] for c in meta_cols if c in old_df.columns and len(old_df)}
    for row in fresh_rows:
        row.update(meta)

    new_df = pd.DataFrame(fresh_rows)

    if triggers:
        new_df = new_df[new_df["trigger_word"].isin(triggers)]
    if rules:
        new_df = new_df[new_df["rule"].isin(rules)]

    return old_df, new_df


def diff_and_write(video_slug, old_df, new_df, out_path, commit, triggers, rules,
                    mutations_csv_path):
    old_df = old_df.copy()
    for col in JOIN_KEY_COLUMNS:
        if col not in old_df.columns:
            old_df[col] = None
        if col not in new_df.columns:
            new_df[col] = None

    # Scope the OLD side to exactly the same filter used to build new_df,
    # so "in old but not in new" means "this rule genuinely stopped
    # firing here" -- not "this row was never in scope of the filter".
    old_scoped = old_df
    if triggers:
        old_scoped = old_scoped[old_scoped["trigger_word"].isin(triggers)]
    if rules:
        old_scoped = old_scoped[old_scoped["rule"].isin(rules)]

    old_keyed = old_scoped.set_index(JOIN_KEY_COLUMNS, drop=False)
    new_keyed = new_df.set_index(JOIN_KEY_COLUMNS, drop=False)

    changed, protected, added, removed = [], [], [], []
    compare_cols = [c for c in new_df.columns
                    if c in old_df.columns and c not in MANUAL_REVIEW_COLUMNS]

    for key in new_keyed.index:
        if key not in old_keyed.index:
            added.append(key)
            continue
        old_row = old_keyed.loc[key]
        new_row = new_keyed.loc[key]
        if isinstance(old_row, pd.DataFrame):  # duplicate key, be conservative
            continue
        differs = any(str(old_row.get(c)) != str(new_row.get(c)) for c in compare_cols)
        if not differs:
            continue
        if bool(old_row.get("manual_reviewed", False)):
            protected.append(key)
        else:
            changed.append(key)

    for key in old_keyed.index:
        if key not in new_keyed.index:
            removed.append(key)

    print(f"\n=== {video_slug} ===")
    print(f"  {'would change' if not commit else 'changed'}:  {len(changed)}")
    print(f"  needs manual re-check (already reviewed, rule output disagrees): {len(protected)}")
    print(f"  new rows (rule now fires where it didn't before): {len(added)}")
    print(f"  removed rows (rule no longer fires here):         {len(removed)}")

    if protected:
        print("  -- rows needing manual re-check --")
        for key in protected[:10]:
            print(f"     {key}")
        if len(protected) > 10:
            print(f"     ...and {len(protected) - 10} more")

    if not commit:
        combined = new_df.copy()
        combined["_rerun_status"] = "candidate"
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Comparison file written: {out_path.name} "
              f"(not applied -- rerun with --commit to apply)")
        return

    result_df = old_df.copy()
    # PATCH (2.6): the diff computation above already treats a duplicate
    # OLD-side join key conservatively (skips comparison via `isinstance
    # (old_row, pd.DataFrame)`), but this commit-application loop had no
    # equivalent guard on either side -- a duplicate key here would either
    # silently overwrite every matching OLD row with the same NEW row's
    # values (mask matching more than one row), or blow up/misbehave on
    # an ambiguous new_keyed.loc[key] lookup (a DataFrame where a Series
    # is expected). Both sides get the same conservative skip the diff
    # loop already applies, logged so it's visible rather than silent.
    skipped_ambiguous = []
    for key in changed + added:
        new_row = new_keyed.loc[key]
        if isinstance(new_row, pd.DataFrame):
            skipped_ambiguous.append(key)
            continue
        mask = pd.Series(True, index=result_df.index)
        for col in JOIN_KEY_COLUMNS:
            mask &= (result_df[col] == new_row[col])
        if mask.sum() > 1:
            skipped_ambiguous.append(key)
            continue
        if mask.any():
            for col in compare_cols:
                if col in new_row.index:
                    result_df.loc[mask, col] = new_row[col]
        else:
            result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)

    if skipped_ambiguous:
        print(f"  ⚠️ Skipped {len(skipped_ambiguous)} row(s) with an ambiguous "
              f"(duplicate) join key -- not applied, needs a manual look:")
        for key in skipped_ambiguous[:10]:
            print(f"     {key}")
        if len(skipped_ambiguous) > 10:
            print(f"     ...and {len(skipped_ambiguous) - 10} more")

    result_df.to_csv(mutations_csv_path, index=False, encoding="utf-8-sig")
    print(f"  Applied -- {mutations_csv_path.name} updated in place "
          f"(manual-reviewed rows untouched, removed-row candidates left "
          f"in place rather than deleted -- see 'removed' count above).")


def _video_folders(video):
    # CorCenCC output from an older reader (no READER_MARKER) is skipped: its
    # word stream itself is out of date, so only re-reading it (menu 3 -> c)
    # fixes it -- and re-scoring it while that runs could collide with the
    # reader moving the same folder to _deleted (2026-10-02).
    from corpus_corcencc import READER_MARKER
    folders = sorted({p.parent for p in RUNS_DIR.glob("*/*/segments_*.csv")
                      if "_deleted" not in p.parts
                      and not (p.parent.name.startswith("CorCenCC_")
                               and not (p.parent / READER_MARKER).exists())})
    if video and video != "all":
        folders = [f for f in folders if video in f.name or video in f.parent.name]
    return folders


def run_rerun(trigger_arg=None, rule_arg=None, video="all", commit=False, branches=None):
    """
    Core entry point, usable both from the CLI (main(), below) and
    imported directly by welsh_pipeline.py's menu.

    Folders with a saved tagged word stream get a full rerun of the
    branches in `branches` (comma-separated names from BRANCHES, or None for
    all five; with all five the segments/words/lemmas/pos files are
    rewritten too). Folders without one get the legacy mutation-only rerun
    when trigger_arg/rule_arg are given (comma-separated, as from
    --trigger/--rule), and are listed otherwise.
    """
    names = [b.strip() for b in branches.split(",")] if branches else list(BRANCHES)
    unknown = [b for b in names if b not in BRANCHES]
    if unknown:
        print(f"Unknown branch(es): {', '.join(unknown)}. Choose from: {', '.join(BRANCHES)}")
        return False
    keys = [BRANCHES[b] for b in names]
    rewrite_transcripts = len(names) == len(BRANCHES)

    folders = _video_folders(video)
    if not folders:
        print("No matching output folders found.")
        return False

    load_lemma_cache()
    from mutation_engine import load_bangor_lexicon
    # Detection reads the lexicon live (inflected verbs, noun number, lemmas):
    # without it the object rule produced no rows and guarded contexts came
    # back ("sy gynno"), rewriting every folder wrongly (2026-09-30). Stop.
    if not load_bangor_lexicon():
        print("Nothing rerun: the Bangor lexicon is required for a rerun.")
        return False
    untagged, totals = [], Counter()
    for folder in tqdm(folders, desc="Folders"):
        try:
            summary = rerun_folder(folder, keys, rewrite_transcripts, commit)
        except Exception as e:
            tqdm.write(f"  💥 {folder.parent.name}/{folder.name}: {e} -- left unchanged.")
            continue
        if summary is None:
            untagged.append(folder)
            continue
        _print_summary(folder, summary, commit)
        for s in summary["files"].values():
            for name in ("added", "removed", "changed", "reviewed_changed"):
                totals[name] += s[name]
    save_lemma_cache()

    print(f"\nFull rerun: {len(folders) - len(untagged)} folder(s), branches: {', '.join(names)}. "
          f"+{totals['added']} -{totals['removed']} ~{totals['changed']} row(s)"
          + (f", {totals['reviewed_changed']} reviewed row(s) changed" if totals["reviewed_changed"] else "")
          + ("" if commit else f" -- dry run; see {CANDIDATE_PREFIX}*.csv, apply with --commit"))
    if untagged:
        print(f"{len(untagged)} folder(s) have no saved tagging (processed before full reruns "
              f"existed), e.g. {untagged[0].parent.name}/{untagged[0].name}. Process them once "
              f"more to enable full reruns"
              + ("; running the legacy mutation-only rerun on them now." if (trigger_arg or rule_arg)
                 else "; --trigger/--rule runs the legacy mutation-only rerun on them."))
    if not (trigger_arg or rule_arg) or not untagged:
        return True
    return _run_legacy(trigger_arg, rule_arg, untagged, commit)


def _run_legacy(trigger_arg, rule_arg, folders, commit):
    """The original mutation-only rerun, for folders with no saved tagging."""
    triggers = [normalize_word(t) for t in trigger_arg.split(",")] if trigger_arg else None
    rules    = [r.strip() for r in rule_arg.split(",")] if rule_arg else None

    # PATCH: mutations_original_*.csv and mutations_corroborated_*.csv now
    # both match "mutations_*.csv" -- restricted to originals only, since
    # rule re-classification changes status/is_erosion/mutation_found,
    # which only makes sense against the raw pipeline output. Running it
    # against the corroborated file too would reclassify the same video
    # twice and leave the corroboration columns stale against the new
    # classification underneath them; re-run mutation_captions.py's
    # corroboration pass afterward if the corroborated version needs
    # refreshing against the new results.
    all_csvs = sorted(p for folder in folders
                      for p in folder.glob("mutations_original_*.csv"))
    all_csvs = [p for p in all_csvs if "precaption_backup" not in p.name
                and "_rerun_candidate" not in p.name and "_deleted" not in p.parts]

    if not all_csvs:
        print("No matching mutations CSVs found.")
        return False

    load_lemma_cache()

    for mutations_csv_path in tqdm(all_csvs, desc="Videos (legacy)"):
        result = rerun_one_video(mutations_csv_path, triggers, rules)
        if result is None:
            continue
        old_df, new_df = result
        # prefixed, not suffixed: a name starting "mutations_" is read as
        # real output by corpus_analyzer.py and mutation_manual_editing.py
        out_path = mutations_csv_path.with_name(CANDIDATE_PREFIX + mutations_csv_path.name)
        diff_and_write(mutations_csv_path.parent.name, old_df, new_df,
                        out_path, commit, triggers, rules, mutations_csv_path)

    save_lemma_cache()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", help="Comma-separated branch(es) to rerun: "
                                      f"{', '.join(BRANCHES)} [default: all five]")
    ap.add_argument("--trigger", help="Legacy mode only (folders without saved tagging): "
                                      "comma-separated trigger word(s), e.g. yn,ei")
    ap.add_argument("--rule", help="Legacy mode only: comma-separated rule name(s), "
                                   "e.g. word_trigger,phantom_check")
    ap.add_argument("--video", default="all",
                     help="Substring to match against output folder names, or 'all'")
    ap.add_argument("--commit", action="store_true",
                     help="Apply changes in place instead of writing comparison files only")
    args = ap.parse_args()

    ok = run_rerun(trigger_arg=args.trigger, rule_arg=args.rule,
                    video=args.video, commit=args.commit, branches=args.branch)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
