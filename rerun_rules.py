"""
rerun_rules.py
===============
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

Output is always written to a separate *_rerun_candidate.csv file next
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
    python rerun_rules.py --trigger yn                     # dry run, all videos
    python rerun_rules.py --trigger yn,ei --rule word_trigger
    python rerun_rules.py --rule phantom_check --video Diddymur_Senedd
    python rerun_rules.py --trigger yn --commit             # apply after reviewing the diff
"""
import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mutation_engine import (
    BASE_DIR, normalize_word, process_comprehensive_mutations,
    align_with_gap_tolerance, load_lemma_cache, save_lemma_cache,
)
from spacy_tagging import load_spacy, parse_spacy_doc

MUT_DIR   = BASE_DIR / "mutations"
TRANS_DIR = BASE_DIR / "transcriptions"

# Columns manual_editing.py owns on a mutation row. Never overwritten by
# --commit, and their presence (manual_reviewed == True specifically) is
# what protects a row from replacement at all.
MANUAL_REVIEW_COLUMNS = [
    "manual_reviewed", "is_erosion_original", "flagged",
    "flag_note", "review_count", "review_log",
]

JOIN_KEY_COLUMNS = ["video_title", "timestamp", "trigger_word", "following_word"]


def _mutations_dir_for(mutations_csv_path):
    """The transcriptions/<stamp>/<slug>/ folder matching a given
    mutations/<stamp>/<slug>/mutations_<stamp>_<slug>.csv path.

    # PATCH: updated for _video_slug()'s restored run-then-video nesting --
    # mutations_csv_path now has two parent levels to read off (slug, then
    # stamp above it), not one flat <stamp>_<slug> folder. Falls back to a
    # BASE_DIR-rooted rglob for anything from before this fix (older flat
    # layout, or the layout before that), same pattern as
    # fetch_captions.py/manual_editing.py's find_segments_csv.
    """
    slug  = mutations_csv_path.parent.name
    stamp = mutations_csv_path.parent.parent.name
    candidate = TRANS_DIR / stamp / slug
    if candidate.exists():
        return candidate

    folder_name = mutations_csv_path.stem[len("mutations_"):]
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
                "spacy_token":          spacy_tok,
                "synthetic":            False,
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

    meta_cols = ["video_title", "video_url", "source", "channel_register",
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
    for key in changed + added:
        new_row = new_keyed.loc[key]
        mask = pd.Series(True, index=result_df.index)
        for col in JOIN_KEY_COLUMNS:
            mask &= (result_df[col] == new_row[col])
        if mask.any():
            for col in compare_cols:
                if col in new_row.index:
                    result_df.loc[mask, col] = new_row[col]
        else:
            result_df = pd.concat([result_df, new_row.to_frame().T], ignore_index=True)

    result_df.to_csv(mutations_csv_path, index=False, encoding="utf-8-sig")
    print(f"  Applied -- {mutations_csv_path.name} updated in place "
          f"(manual-reviewed rows untouched, removed-row candidates left "
          f"in place rather than deleted -- see 'removed' count above).")


def run_rerun(trigger_arg=None, rule_arg=None, video="all", commit=False):
    """
    Core entry point, usable both from the CLI (main(), below) and
    imported directly by welsh_pipeline.py's menu. trigger_arg/rule_arg
    are comma-separated strings (or None) exactly as they'd arrive from
    --trigger/--rule; keeping that shape here (rather than pre-split
    lists) means both callers pass through the same normalization path
    instead of the menu needing to duplicate it.
    """
    if not trigger_arg and not rule_arg:
        print("Specify at least a trigger or a rule (nothing to filter to otherwise --"
              " re-running every rule on every video is just a full-corpus reprocess,"
              " which this tool intentionally doesn't do; use option 3 for that).")
        return False

    triggers = [normalize_word(t) for t in trigger_arg.split(",")] if trigger_arg else None
    rules    = [r.strip() for r in rule_arg.split(",")] if rule_arg else None

    all_csvs = sorted(MUT_DIR.rglob("mutations_*.csv"))
    all_csvs = [p for p in all_csvs if "precaption_backup" not in p.name
                and "_rerun_candidate" not in p.name]
    if video and video != "all":
        all_csvs = [p for p in all_csvs if video in p.parent.name]

    if not all_csvs:
        print("No matching mutations CSVs found.")
        return False

    load_lemma_cache()

    for mutations_csv_path in tqdm(all_csvs, desc="Videos"):
        result = rerun_one_video(mutations_csv_path, triggers, rules)
        if result is None:
            continue
        old_df, new_df = result
        out_path = mutations_csv_path.with_name(
            mutations_csv_path.stem + "_rerun_candidate.csv")
        diff_and_write(mutations_csv_path.parent.name, old_df, new_df,
                        out_path, commit, triggers, rules, mutations_csv_path)

    save_lemma_cache()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trigger", help="Comma-separated trigger word(s), e.g. yn,ei")
    ap.add_argument("--rule", help="Comma-separated rule name(s), e.g. word_trigger,phantom_check")
    ap.add_argument("--video", default="all",
                     help="Substring to match against mutations CSV folder names, or 'all'")
    ap.add_argument("--commit", action="store_true",
                     help="Apply changes in place instead of writing a comparison file only")
    args = ap.parse_args()

    ok = run_rerun(trigger_arg=args.trigger, rule_arg=args.rule,
                    video=args.video, commit=args.commit)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()