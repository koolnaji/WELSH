r"""
manual_editing.py -- interactive manual review tool for mutations_*.csv files.
NOT part of the automated pipeline. Run this by itself, same spirit as
alaw.py -- standalone, duplicated constants rather than imported, so it
has no dependency on mutation_engine.py actually loading (no Whisper, no
spaCy, no Cysill API -- just pandas and a terminal).

What this does, per row you're asked about:
    1. Shows you the mutation context: trigger word, expected mutation,
       what was actually found, status, current is_erosion value.
    2. Pulls the sibling segments_*.csv for that video and shows you 1-2
       sentences of surrounding transcript, with the trigger/following
       words marked, so you have enough context to judge without having
       to reopen the transcript file separately.
    3. Asks you to confirm or correct is_erosion by ear (Y/N), skip, flag
       for follow-up, jump to a specific row, or move back/forward
       through the queue.
    4. Autosaves after every single decision -- if you close the terminal
       mid-review, nothing since the last row is lost, and next run picks
       up where you left off (already-reviewed rows are skipped by
       default).
    5. Tracks how many times each row has been handled (review_count /
       review_log columns) so repeat visits are visible, with a way to
       clear that log per-row or in bulk if it gets cluttered.

The column your prompt referred to as "is_mutation" is is_erosion in the
actual CSV schema (the boolean that determines whether a row counts toward
the erosion rate). This script edits that column. If you meant something
else, change EDIT_COLUMN below.

Usage:
    python manual_editing.py
        -- finds every mutations_*.csv under the default mutations/ folder
           and reviews every row with is_erosion == True (the population
           the 60% erosion rate claim actually rests on), skipping rows
           already manually reviewed in a previous session.

    python manual_editing.py path\to\mutations_20250101_120000_SomeVideo.csv
        -- review just this one file instead.

    python manual_editing.py --pick
        -- show a numbered list of every discovered mutations_*.csv (with
           row counts) and interactively choose which one(s) to review,
           e.g. "1,3,5-7" or "all". Ignored if explicit file paths were
           already given on the command line.

    python manual_editing.py --sample 20
        -- randomly sample 20 rows from the filtered set instead of
           reviewing all of them. Use --seed N for a reproducible sample.

    python manual_editing.py --status erosion,wrong_mutation_type,mutation_mismatch
        -- review a different/wider set of statuses instead of the
           is_erosion==True default (see EROSION_STATUSES below).

    python manual_editing.py --review-all
        -- re-review rows even if they already have a manual decision
           recorded (normally these are skipped).

    python manual_editing.py --context 2
        -- show 2 segments of context before/after instead of the
           default 1.

    python manual_editing.py --flagged-only
        -- review only rows previously marked with [f]lag, regardless of
           status filter. Useful as a dedicated follow-up pass.

    python manual_editing.py --clear-all-logs
        -- bulk-clear review_count/review_log for every row in the
           filtered queue (does NOT touch is_erosion, manual_reviewed, or
           flags), print how many were cleared, then exit without
           entering interactive review. Asks for confirmation unless
           --yes is also given. Useful maintenance after a lot of
           skip/revisit noise has piled up in the logs.

During review:
    [y]es / [n]o   -- record is_erosion decision, save, advance.
    [s]kip         -- leave decision untouched, advance (still shows up
                      in future default-mode sessions since it's not
                      marked reviewed). Still logged as a "handled" visit.
    [f]lag         -- mark row for follow-up and optionally jot a memo
                      (multiple lines OK, blank line to finish). Doesn't
                      record an is_erosion decision or mark the row
                      reviewed. Flagging an already-flagged row appends a
                      new timestamped memo below the old one rather than
                      overwriting it, so notes accumulate across visits.
    [j]ump         -- jump straight to a specific row number in the
                      current queue (1-indexed, as shown in "[row X/Y]").
                      Doesn't count as handling the row you jump away from.
    [/]search      -- search trigger_word/following_word/rule/video_title/
                      note/flag_note across the WHOLE queue (not just what's
                      ahead), case-insensitive substring match. Shows up to
                      20 matches with their row numbers so you can jump
                      straight to one -- e.g. searching "oes" to revisit
                      every row that trigger touched.
    prev/next file -- "[" jumps to the first queued row of the previous
                      file in the queue, "]" to the next file. Works even
                      with --sample, which interleaves files' rows in
                      random order (doesn't assume files are contiguous).
    [c]lear-log    -- clear this row's review_count/review_log only (with
                      confirmation). Doesn't touch its is_erosion decision,
                      manual_reviewed, or flag status -- just the "how many
                      times have I looked at this" history.
    [p]revious     -- go back one row in the queue to re-check or change
                      a decision. No limit on how far back; stops at the
                      first row.
    [q]uit         -- stop. Everything up through the last saved decision
                      is already on disk.
"""
import argparse
import csv
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Same locations mutation_engine.py uses -- duplicated, not imported,
# per this project's existing standalone-script convention (see alaw.py).
BASE_DIR  = Path.home() / "Documents" / "welsh_analysis"
TRANS_DIR = BASE_DIR / "transcriptions"
MUT_DIR   = BASE_DIR / "mutations"

# The boolean column this script edits. "is_mutation" in casual conversation
# maps to "is_erosion" in the actual CSV -- change this if that's wrong.
EDIT_COLUMN = "is_erosion"

# Default population to review: the two statuses that actually feed the
# erosion rate numerator (see EVALUABLE_STATUSES / erosion_rate formula in
# corpus_analyzer.py). mutation_mismatch is deliberately NOT included by
# default -- it's a different, separately-flagged methodological question
# (see conversation history) -- but --status can widen this any time.
EROSION_STATUSES = ["erosion", "wrong_mutation_type"]

# PATCH: plain-English gloss for each rule_name, shown next to the raw value
# in review_row. Saves re-deriving "wait, which environment is this again"
# from the raw rule string every single row -- built from the actual rules
# in mutation_engine.py, including the recent fixes (sy/sydd as triggers,
# un's ll/rh exemption, DOM finite-verb gating, unmutable-initial exemption).
RULE_DESCRIPTIONS = {
    "word_trigger":                   "flat trigger word (mae/ydy/oes/sy/sydd/prepositions/etc.) -> soft",
    "definite_article+fem_noun":      "y/yr/r + feminine singular noun -> soft_limited (ll/rh exempt)",
    "fem_noun+adjective":             "feminine singular noun + adjective -> soft (plural exempt)",
    "fem_noun+genitive_noun":         "feminine singular noun + genitive noun -> soft (\"cot law\", \"Gwyl Ddewi\")",
    "numeral+fem_noun":               "un + feminine singular noun -> soft_limited (ll/rh exempt)",
    "numeral_general+noun":           "dau/dwy + noun (any gender) -> soft",
    "restricted_nasal_numeral":       "nasal-triggering numeral + noun -> nasal",
    "feminine_ordinal+noun":          "feminine ordinal (trydedd/pedwaredd/...) + noun -> soft",
    "preposed_adjective+noun":        "preposed adjective (hen/rhyw/...) + noun -> soft (any number/gender)",
    "decompounded_noun":              "decompounded compound noun, second element -> soft",
    "h_mutation":                     "possessive/prep trigger + vowel-initial word -> h- prefix",
    "mixed_mutation":                 "ni/na/oni/nid/a/etc. -> soft for most consonants, aspirate for p/t/c",
    "feminine_ei":                    "'ei' (her) -> aspirate, vs 'ei' (his) -> soft (homograph, disambiguated)",
    "phantom_check":                  "trigger omitted but following word still shows correct mutation",
    "spacy_obj_soft":                 "object of a FINITE verb -> soft (verb-nouns excluded)",
}

TIMESTAMP_RE = re.compile(r"([\d.]+)s\s*-\s*([\d.]+)s")


def find_mutation_csvs(explicit_paths):
    # PATCH: exclude "*_precaption_backup.csv" -- see the matching comment
    # in corpus_analyzer.py's load_and_merge_mutations(). These are
    # fetch_captions.py's one-time pre-corroboration backups, not separate
    # videos' worth of review data; without this filter they'd show up as
    # duplicate files to review, with stale (pre-corroboration) rows.
    if explicit_paths:
        paths = []
        for p in explicit_paths:
            p = Path(p)
            if p.is_dir():
                paths.extend(sorted(
                    f for f in p.rglob("mutations_*.csv")
                    if not f.name.endswith("_precaption_backup.csv")
                ))
            else:
                paths.append(p)
        return paths
    if not MUT_DIR.exists():
        print(f"{MUT_DIR} doesn't exist and no files were given. Either "
              f"point this at your mutations CSVs directly, or check "
              f"BASE_DIR at the top of this script.")
        sys.exit(1)
    return sorted(
        f for f in MUT_DIR.rglob("mutations_*.csv")
        if not f.name.endswith("_precaption_backup.csv")
    )


def quick_row_count(path):
    """Lightweight row count without a full pandas parse. Uses csv.reader
    (not naive line-splitting) specifically because flag_note/review_log
    can contain embedded newlines inside quoted multi-line memos -- a
    plain "count the lines" approach would overcount any file with a
    multi-line flag/log entry already saved."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return sum(1 for _ in csv.reader(f)) - 1  # minus header
    except Exception:
        return "?"


def pick_files_interactively(csv_paths):
    """Numbered list with row counts; accepts comma-separated indices
    and/or ranges (e.g. "1,3,5-7"), or "all"/blank for everything."""
    print(f"\nFound {len(csv_paths)} mutations CSV file(s):\n")
    for i, p in enumerate(csv_paths, 1):
        print(f"  [{i:>3}] {p.name}  ({quick_row_count(p)} rows)")
    print()
    raw = input("Which file(s)? Comma-separated numbers, ranges (e.g. "
                "1,3,5-7), or 'all': ").strip().lower()
    if raw in ("", "all"):
        return csv_paths
    chosen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                chosen.update(range(int(a), int(b) + 1))
            else:
                chosen.add(int(part))
        except ValueError:
            print(f"  couldn't parse {part!r}, ignoring it.")
    selected = [csv_paths[i - 1] for i in sorted(chosen) if 1 <= i <= len(csv_paths)]
    if not selected:
        print("  nothing valid selected -- reviewing all files instead.")
        return csv_paths
    return selected


def find_segments_csv(mutations_csv_path):
    """
    Mirrors _video_slug()'s layout in mutation_engine.py:
        mutations/<stamp>_<slug>/mutations_<stamp>_<slug>.csv
        transcriptions/<stamp>_<slug>/segments_<stamp>_<slug>.csv
    Falls back to a recursive filename search if that exact path doesn't
    exist (older files, or a layout that predates migrate_files.py).
    Returns None if nothing is found -- context just won't be available
    for that file, rather than crashing the whole review session over it.

    # PATCH: this was hardcoded to _video_slug()'s old two-level transcript
    # nesting (TRANS_DIR/<stamp>/<folder_name>/...) with mutations sitting
    # in a bare TRANS_DIR/<stamp>/ that had no slug subfolder at all --
    # matching what _video_slug() used to do, but not what its own
    # docstring said. Both output types now share one per-video
    # <stamp>_<slug> subfolder, so folder_name (read straight off the
    # mutations filename) is also the transcript subfolder name directly --
    # no separate "stamp" parent folder to read off mutations_csv_path.
    """
    stem = mutations_csv_path.stem  # "mutations_<stamp>_<slug>"
    if not stem.startswith("mutations_"):
        return None
    folder_name = stem[len("mutations_"):]  # "<stamp>_<slug>"

    candidate = TRANS_DIR / folder_name / f"segments_{folder_name}.csv"
    if candidate.exists():
        return candidate

    # Fallback: search from the corpus root for a same-named segments file
    # anywhere under it (older files, or a layout that predates this fix).
    target_name = f"segments_{folder_name}.csv"
    search_root = mutations_csv_path.parents[2] if len(mutations_csv_path.parents) >= 3 else BASE_DIR
    matches = list(search_root.rglob(target_name))
    return matches[0] if matches else None


def load_segments(mutations_csv_path, _cache={}):
    """Cached per mutations-CSV so we don't re-read the same segments file
    once per row when a video has many mutation rows."""
    if mutations_csv_path in _cache:
        return _cache[mutations_csv_path]
    seg_path = find_segments_csv(mutations_csv_path)
    if seg_path is None:
        _cache[mutations_csv_path] = None
        return None
    try:
        df = pd.read_csv(seg_path, encoding="utf-8-sig")
        df = df.sort_values("segment_start").reset_index(drop=True)
    except Exception as e:
        print(f"  (couldn't read segments file {seg_path}: {e})")
        df = None
    _cache[mutations_csv_path] = df
    return df


def highlight(text, words):
    out = text
    for w in words:
        w = str(w).strip()
        if not w or w == "[OMITTED]":
            continue
        out = re.sub(rf"(?i)\b{re.escape(w)}\b", f">>>{w}<<<", out)
    return out


def get_context(mutations_csv_path, row, context_n):
    segs = load_segments(mutations_csv_path)
    if segs is None or "segment_start" not in segs.columns:
        return "(no segments file found for this video -- no context available)"

    m = TIMESTAMP_RE.search(str(row.get("timestamp", "")))
    if not m:
        return "(couldn't parse timestamp on this row)"
    t_start, t_end = float(m.group(1)), float(m.group(2))

    covering = segs[(segs["segment_start"] <= t_end) & (segs["segment_end"] >= t_start)]
    if covering.empty:
        # nearest segment by start time as a fallback
        idx = (segs["segment_start"] - t_start).abs().idxmin()
        covering = segs.loc[[idx]]

    center_idx = covering.index.min()
    lo = max(0, center_idx - context_n)
    hi = min(len(segs) - 1, covering.index.max() + context_n)
    window = segs.loc[lo:hi]

    highlight_words = [row.get("trigger_word"), row.get("following_word")]
    lines = []
    for _, s in window.iterrows():
        marker = ">> " if s["segment_start"] <= t_end and s["segment_end"] >= t_start else "   "
        lines.append(f"{marker}[{s['segment_start']:.1f}s-{s['segment_end']:.1f}s] "
                      f"{highlight(str(s['segment_text']), highlight_words)}")
    return "\n".join(lines)


def fmt_mmss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def review_row(mutations_csv_path, video_title, row, context_n):
    m = TIMESTAMP_RE.search(str(row.get("timestamp", "")))
    t0 = fmt_mmss(float(m.group(1))) if m else None

    # PATCH: reorganized into labeled sections (mutation context /
    # confidence / current state / history, each only shown if it has
    # something to say) instead of a flat stack of prints -- easier to
    # scan at a glance, especially now that review_count/review_log adds
    # a 4th thing competing for attention at the top of each row.
    print("\n" + "=" * 78)
    if t0:
        print(f"VIDEO   : {video_title}")
        print(f"TIME    : @ {t0}  ({row.get('timestamp')})")
    else:
        print(f"VIDEO   : {video_title}  ({row.get('timestamp')})")
    print("-" * 78)
    print("MUTATION CONTEXT")
    rule_val = row.get("rule")
    gloss = RULE_DESCRIPTIONS.get(rule_val)
    print(f"  trigger    : {row.get('trigger_word')!r}   rule: {rule_val}")
    if gloss:
        print(f"               ({gloss})")
    print(f"  following  : {row.get('following_word')!r}")
    print(f"  expected   : {row.get('expected_mutation')}     "
          f"found: {row.get('mutation_found')}")
    print("-" * 78)
    print("CONFIDENCE")
    print(f"  status            : {row.get('status')}")
    print(f"  tagger_agreement  : {row.get('tagger_agreement')}")
    print(f"  confidence_score  : {row.get('confidence_score')}")
    print("-" * 78)
    print("CURRENT STATE")
    print(f"  {EDIT_COLUMN:<17s} : {row.get(EDIT_COLUMN)}   "
          f"(manual_reviewed: {row.get('manual_reviewed', False)})")

    rc = row.get("review_count", 0)
    flagged = bool(row.get("flagged", False))
    has_note = str(row.get("note", "")).strip()
    if (pd.notna(rc) and int(rc) > 0) or flagged or has_note:
        print("-" * 78)
        print("HISTORY")
        if pd.notna(rc) and int(rc) > 0:
            print(f"  reviewed {int(rc)} time(s) before this session")
            rl = row.get("review_log")
            if pd.notna(rl) and str(rl).strip():
                for line in str(rl).splitlines()[-3:]:
                    print(f"    {line}")
        if flagged:
            note = row.get("flag_note")
            if str(note).strip() and str(note) != "nan":
                print("  *** FLAGGED -- memo(s):")
                for line in str(note).splitlines():
                    print(f"      {line}")
            else:
                print("  *** FLAGGED (no memo) ***")
        if has_note:
            print(f"  note: {row.get('note')}")

    print("-" * 78)
    print("TRANSCRIPT CONTEXT")
    print(get_context(mutations_csv_path, row, context_n))
    print("-" * 78)

    while True:
        choice = input(f"Genuinely erosion (radical form actually spoken)? "
                        f"[y]es / [n]o / [s]kip / [f]lag / [j]ump / "
                        f"[/]search / [c]lear-log / prev-file[[]/next-file[]] / "
                        f"[p]revious / [q]uit: ").strip().lower()
        if choice in ("y", "n", "s", "f", "j", "/", "c", "[", "]", "p", "q"):
            return choice
        print("  please enter y, n, s, f, j, / (search), c, [ or ] (prev/next file), p, or q")


def prompt_memo():
    """Multi-line memo entry: keep taking lines until a blank one, so you
    can jot down more than a one-liner. Just hitting Enter immediately
    skips adding anything."""
    print("  memo (multiple lines OK, blank line to finish, Enter alone to skip):")
    lines = []
    while True:
        line = input("    ")
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def log_review_action(df, idx, action_desc):
    """Appends a timestamped line to review_log and bumps review_count by
    one. Called for every action that actually "handles" a row (y/n/s/f)
    -- not for [p]revious/[j]ump/[q]uit, which don't represent a decision
    or a deliberate look at this specific row's content."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    existing_log = df.at[idx, "review_log"]
    entry = f"[{stamp}] {action_desc}"
    df.at[idx, "review_log"] = (f"{existing_log}\n{entry}"
                                 if pd.notna(existing_log) and str(existing_log).strip()
                                 else entry)
    existing_count = df.at[idx, "review_count"]
    df.at[idx, "review_count"] = (int(existing_count) + 1
                                   if pd.notna(existing_count) else 1)


def save(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="mutations_*.csv file(s) or folder(s). "
                                              "Default: every mutations_*.csv under mutations/")
    ap.add_argument("--pick", action="store_true",
                     help="interactively choose which discovered mutations_*.csv "
                          "file(s) to review from a numbered list, instead of "
                          "reviewing every file found. Ignored if explicit file "
                          "paths were already given.")
    ap.add_argument("--status", default=None,
                     help=f"comma-separated status values to review. "
                          f"Default: {','.join(EROSION_STATUSES)}")
    ap.add_argument("--trigger", default=None,
                     help="comma-separated trigger word(s) to filter to "
                          "(case-insensitive, exact match against "
                          "trigger_word). Combines with --status/--rule.")
    ap.add_argument("--rule", default=None,
                     help="comma-separated rule name(s) to filter to, e.g. "
                          "fem_noun+adjective,definite_article+fem_noun. "
                          "Combines with --status/--trigger.")
    ap.add_argument("--summary", action="store_true",
                     help="print counts by trigger_word/rule/status for the "
                          "current filtered set and exit, without entering "
                          "interactive review. Useful for scoping a search "
                          "before committing to it, e.g. '--trigger sy,sydd "
                          "--summary' to see how many rows that'd pull in.")
    ap.add_argument("--sample", type=int, default=None,
                     help="randomly sample this many rows from the filtered set")
    ap.add_argument("--seed", type=int, default=None, help="random seed for --sample")
    ap.add_argument("--context", type=int, default=1,
                     help="segments of context before/after (default: 1)")
    ap.add_argument("--review-all", action="store_true",
                     help="include rows that already have a manual decision recorded")
    ap.add_argument("--flagged-only", action="store_true",
                     help="review only rows previously marked with [f]lag "
                          "(ignores --status filtering for row selection, "
                          "but --review-all still applies to already-decided flagged rows)")
    ap.add_argument("--clear-all-logs", action="store_true",
                     help="bulk-clear review_count/review_log for every row in "
                          "the filtered queue, then exit (does not touch "
                          "is_erosion decisions, manual_reviewed, or flags). "
                          "Use to de-clutter after lots of skip/revisit noise.")
    ap.add_argument("--yes", action="store_true",
                     help="skip the confirmation prompt for --clear-all-logs")
    args = ap.parse_args()

    statuses = args.status.split(",") if args.status else EROSION_STATUSES
    csv_paths = find_mutation_csvs(args.files)
    if not csv_paths:
        print("No mutations_*.csv files found.")
        return

    if args.pick and not args.files:
        csv_paths = pick_files_interactively(csv_paths)

    print(f"\n{len(csv_paths)} mutations CSV file(s) selected:")
    for p in csv_paths:
        print(f"  {p.name}  ({quick_row_count(p)} rows)")

    if args.flagged_only:
        print(f"\nReviewing previously flagged rows only.")
    else:
        print(f"\nReviewing statuses: {statuses}")
    if args.trigger:
        triggers_wanted = {t.strip().lower() for t in args.trigger.split(",")}
        print(f"  filtered to trigger(s): {sorted(triggers_wanted)}")
    else:
        triggers_wanted = None
    if args.rule:
        rules_wanted = {r.strip() for r in args.rule.split(",")}
        print(f"  filtered to rule(s): {sorted(rules_wanted)}")
    else:
        rules_wanted = None

    # Load everything, tag each row with its source file + original index
    # so decisions write back to the right place, and pick the review queue.
    loaded = {}         # path -> DataFrame
    raw_file_len = {}   # path -> total rows in the raw, unfiltered CSV
    queue = []           # list of (path, row_index)
    for path in csv_paths:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            print(f"  skipping {path.name}: couldn't read ({e})")
            continue
        if EDIT_COLUMN not in df.columns:
            print(f"  skipping {path.name}: no '{EDIT_COLUMN}' column")
            continue
        if "manual_reviewed" not in df.columns:
            df["manual_reviewed"] = False
        if f"{EDIT_COLUMN}_original" not in df.columns:
            df[f"{EDIT_COLUMN}_original"] = pd.NA
        if "flagged" not in df.columns:
            df["flagged"] = False
        if "flag_note" not in df.columns:
            df["flag_note"] = pd.NA
        if "review_count" not in df.columns:
            df["review_count"] = 0
        if "review_log" not in df.columns:
            df["review_log"] = pd.NA
        loaded[path] = df
        raw_file_len[path] = len(df)

        if args.flagged_only:
            mask = df["flagged"].fillna(False)
        else:
            mask = df["status"].isin(statuses) if "status" in df.columns else df[EDIT_COLUMN] == True
        if triggers_wanted is not None and "trigger_word" in df.columns:
            mask = mask & df["trigger_word"].fillna("").str.lower().isin(triggers_wanted)
        if rules_wanted is not None and "rule" in df.columns:
            mask = mask & df["rule"].fillna("").isin(rules_wanted)
        if not args.review_all:
            mask = mask & (~df["manual_reviewed"].fillna(False))
        for idx in df[mask].index:
            queue.append((path, idx))

    if not queue:
        print("Nothing to review (empty filtered set, or everything's "
              "already been reviewed -- try --review-all).")
        return

    if args.summary:
        all_rows = pd.concat(
            [loaded[p].loc[[i for pp, i in queue if pp == p]] for p in loaded
             if any(pp == p for pp, _ in queue)],
            ignore_index=True,
        )
        print(f"\n{len(queue)} row(s) match this filter. Breakdown:\n")
        if "trigger_word" in all_rows.columns:
            print("By trigger_word:")
            for val, n in all_rows["trigger_word"].value_counts().items():
                print(f"  {val!r:<20} {n}")
        if "rule" in all_rows.columns:
            print("\nBy rule:")
            for val, n in all_rows["rule"].value_counts().items():
                print(f"  {val!r:<40} {n}")
        if "status" in all_rows.columns:
            print("\nBy status:")
            for val, n in all_rows["status"].value_counts().items():
                print(f"  {val!r:<25} {n}")
        return

    if args.clear_all_logs:
        n_with_logs = sum(
            1 for path, idx in queue
            if pd.notna(loaded[path].at[idx, "review_count"])
            and int(loaded[path].at[idx, "review_count"]) > 0
        )
        print(f"\n{n_with_logs} row(s) in the current filtered queue "
              f"({len(queue)} rows total) have a review log to clear.")
        if n_with_logs == 0:
            print("Nothing to clear.")
            return
        if not args.yes:
            confirm = input("Clear review_count/review_log for all of these? "
                             "This does NOT touch is_erosion, manual_reviewed, "
                             "or flags. [y/n]: ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return
        touched_paths = set()
        for path, idx in queue:
            df = loaded[path]
            if pd.notna(df.at[idx, "review_count"]) and int(df.at[idx, "review_count"]) > 0:
                df.at[idx, "review_count"] = 0
                df.at[idx, "review_log"] = pd.NA
                touched_paths.add(path)
        for path in touched_paths:
            save(loaded[path], path)
        print(f"Cleared {n_with_logs} row(s) across {len(touched_paths)} file(s).")
        return

    if args.sample is not None:
        rng = random.Random(args.seed)
        queue = rng.sample(queue, k=min(args.sample, len(queue)))

    already_logged = sum(
        1 for path, idx in queue
        if pd.notna(loaded[path].at[idx, "review_count"])
        and int(loaded[path].at[idx, "review_count"]) > 0
    )

    print(f"\n{len(queue)} row(s) queued for review. "
          f"Progress autosaves after every decision -- safe to quit anytime.")
    if already_logged:
        print(f"({already_logged} of these already have a prior review log "
              f"from earlier sessions.)")
    print(f"[y]es/[n]o decide, [s]kip, [f]lag for later, [j]ump to a row, "
          f"/search, [ / ] prev/next file, [c]lear-log, "
          f"[p]revious to go back, [q]uit.\n")

    reviewed_count = 0
    flagged_count  = 0
    pos = 0  # index into queue -- a plain counter instead of a for-loop so
             # [p]revious/[j]ump can move it backward as well as forward.
    while 0 <= pos < len(queue):
        path, idx = queue[pos]
        df = loaded[path]
        row = df.loc[idx]
        video_title = row.get("video_title", path.stem)

        # PATCH: per-file position, not just the merged multi-file queue
        # position -- "[row X/Y overall]" alone doesn't tell you where you
        # are within a SPECIFIC file when reviewing several at once.
        file_queue_positions = [i for i, (p, _) in enumerate(queue) if p == path]
        file_pos = file_queue_positions.index(pos) + 1
        file_queue_total = len(file_queue_positions)

        print(f"\n[row {pos + 1}/{len(queue)} overall]  "
              f"[{file_pos}/{file_queue_total} queued in {path.name}, "
              f"{raw_file_len[path]} rows total in that file]")
        choice = review_row(path, video_title, row, args.context)

        if choice == "q":
            print("\nQuitting. Progress already saved through the last decision.")
            break

        if choice == "p":
            if pos == 0:
                print("  already at the first row in the queue.")
            else:
                pos -= 1
            continue

        if choice == "j":
            raw = input(f"  jump to row number (1-{len(queue)}): ").strip()
            try:
                target = int(raw)
                if 1 <= target <= len(queue):
                    pos = target - 1
                else:
                    print(f"  out of range -- must be between 1 and {len(queue)}.")
            except ValueError:
                print("  not a number, staying put.")
            continue

        # PATCH: trigger/following/rule/video/note search across the
        # whole queue (not just what's ahead), since you might want to
        # jump back to something you remember seeing a few rows ago just
        # as easily as finding the next occurrence of something new.
        if choice == "/":
            term = input("  search trigger/following/rule/video/notes for: ").strip().lower()
            if not term:
                continue
            matches = []
            for i, (p, ix) in enumerate(queue):
                r = loaded[p].loc[ix]
                haystack = " ".join(str(r.get(c, "")) for c in
                                     ("trigger_word", "following_word", "rule",
                                      "video_title", "note", "flag_note")).lower()
                if term in haystack:
                    matches.append(i)
            if not matches:
                print(f"  no matches for {term!r} in the current queue.")
                continue
            print(f"  {len(matches)} match(es):")
            for m_i in matches[:20]:
                p, ix = queue[m_i]
                r = loaded[p].loc[ix]
                marker = "->" if m_i == pos else "  "
                print(f"   {marker} [{m_i + 1}] {r.get('trigger_word')!r} -> "
                      f"{r.get('following_word')!r}  rule={r.get('rule')}  "
                      f"status={r.get('status')}  ({p.name})")
            if len(matches) > 20:
                print(f"   ... and {len(matches) - 20} more (not shown)")
            raw = input("  jump to which row number? (blank = stay here): ").strip()
            if raw:
                try:
                    target = int(raw)
                    if 1 <= target <= len(queue):
                        pos = target - 1
                    else:
                        print("  out of range.")
                except ValueError:
                    print("  not a number.")
            continue

        # PATCH: jump straight to the first queued row of the previous/
        # next FILE, wherever it is in the queue -- doesn't assume files
        # are stored contiguously, since --sample interleaves rows from
        # different files in random order.
        if choice == "]":
            next_idx = next((i for i in range(pos + 1, len(queue))
                              if queue[i][0] != path), None)
            if next_idx is not None:
                pos = next_idx
            else:
                print("  no other file ahead in the queue.")
            continue

        if choice == "[":
            prev_idx = next((i for i in range(pos - 1, -1, -1)
                              if queue[i][0] != path), None)
            if prev_idx is not None:
                pos = prev_idx
            else:
                print("  no other file behind in the queue.")
            continue

        if choice == "c":
            current_count = row.get("review_count", 0)
            if pd.isna(current_count) or int(current_count) == 0:
                print("  nothing to clear -- this row has no review log yet.")
                continue
            confirm = input(f"  clear {int(current_count)} logged review(s) "
                             f"for this row? This does not undo the "
                             f"{EDIT_COLUMN} decision itself. [y/n]: ").strip().lower()
            if confirm == "y":
                df.at[idx, "review_count"] = 0
                df.at[idx, "review_log"] = pd.NA
                save(df, path)
                print("  cleared.")
            else:
                print("  cancelled.")
            continue

        if choice == "s":
            log_review_action(df, idx, "skipped")
            save(df, path)
            pos += 1
            continue

        if choice == "f":
            existing = df.at[idx, "flag_note"]
            has_existing = pd.notna(existing) and str(existing).strip()
            if has_existing:
                print(f"  existing memo(s):\n{existing}\n")
            memo = prompt_memo()
            df.at[idx, "flagged"] = True
            if memo:
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                entry = f"[{stamp}] {memo}"
                df.at[idx, "flag_note"] = f"{existing}\n---\n{entry}" if has_existing else entry
            log_review_action(df, idx, "flagged")
            save(df, path)
            flagged_count += 1
            pos += 1
            continue

        # y/n
        new_val = (choice == "y")
        if pd.isna(df.at[idx, f"{EDIT_COLUMN}_original"]):
            df.at[idx, f"{EDIT_COLUMN}_original"] = row[EDIT_COLUMN]
        df.at[idx, EDIT_COLUMN] = new_val
        df.at[idx, "manual_reviewed"] = True
        log_review_action(df, idx, f"decided {EDIT_COLUMN}={new_val}")
        save(df, path)
        reviewed_count += 1
        pos += 1

    print(f"\nDone. {reviewed_count} row(s) reviewed, "
          f"{flagged_count} flagged for follow-up this session.")


if __name__ == "__main__":
    main()