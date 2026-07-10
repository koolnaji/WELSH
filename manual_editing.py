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
       for follow-up, or move back/forward through the queue.
    4. Autosaves after every single decision -- if you close the terminal
       mid-review, nothing since the last row is lost, and next run picks
       up where you left off (already-reviewed rows are skipped by
       default).

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

During review:
    [y]es / [n]o   -- record is_erosion decision, save, advance.
    [s]kip         -- leave decision untouched, advance (still shows up
                      in future default-mode sessions since it's not
                      marked reviewed).
    [f]lag         -- mark row for follow-up and optionally jot a memo
                      (multiple lines OK, blank line to finish). Doesn't
                      record an is_erosion decision or mark the row
                      reviewed. Flagging an already-flagged row appends a
                      new timestamped memo below the old one rather than
                      overwriting it, so notes accumulate across visits.
    [p]revious     -- go back one row in the queue to re-check or change
                      a decision. No limit on how far back; stops at the
                      first row.
    [q]uit         -- stop. Everything up through the last saved decision
                      is already on disk.
"""
import argparse
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

TIMESTAMP_RE = re.compile(r"([\d.]+)s\s*-\s*([\d.]+)s")


def find_mutation_csvs(explicit_paths):
    if explicit_paths:
        paths = []
        for p in explicit_paths:
            p = Path(p)
            if p.is_dir():
                paths.extend(sorted(p.rglob("mutations_*.csv")))
            else:
                paths.append(p)
        return paths
    if not MUT_DIR.exists():
        print(f"{MUT_DIR} doesn't exist and no files were given. Either "
              f"point this at your mutations CSVs directly, or check "
              f"BASE_DIR at the top of this script.")
        sys.exit(1)
    return sorted(MUT_DIR.rglob("mutations_*.csv"))


def find_segments_csv(mutations_csv_path):
    """
    Mirrors _video_slug()'s layout in mutation_engine.py:
        mutations/<stamp>/mutations_<stamp>_<slug>.csv
        transcriptions/<stamp>/<stamp>_<slug>/segments_<stamp>_<slug>.csv
    Falls back to a recursive filename search if that exact path doesn't
    exist (older files, or a layout that predates migrate_files.py).
    Returns None if nothing is found -- context just won't be available
    for that file, rather than crashing the whole review session over it.
    """
    stem = mutations_csv_path.stem  # "mutations_<stamp>_<slug>"
    if not stem.startswith("mutations_"):
        return None
    folder_name = stem[len("mutations_"):]  # "<stamp>_<slug>"
    stamp = mutations_csv_path.parent.name  # the "<stamp>" folder mutations_*.csv sits in

    candidate = TRANS_DIR / stamp / folder_name / f"segments_{folder_name}.csv"
    if candidate.exists():
        return candidate

    # Fallback: search from the corpus root (two levels up from the CSV,
    # i.e. mutations/<stamp>/file.csv -> corpus root) for a same-named
    # segments file anywhere under it.
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
    print("\n" + "=" * 78)
    m = TIMESTAMP_RE.search(str(row.get("timestamp", "")))
    if m:
        t0 = fmt_mmss(float(m.group(1)))
        print(f"{video_title}  @ {t0}  ({row.get('timestamp')})")
    else:
        print(f"{video_title}  ({row.get('timestamp')})")
    print(f"trigger: {row.get('trigger_word')!r}  rule: {row.get('rule')}  "
          f"following: {row.get('following_word')!r}")
    print(f"expected_mutation: {row.get('expected_mutation')}   "
          f"mutation_found: {row.get('mutation_found')}")
    print(f"status: {row.get('status')}   tagger_agreement: {row.get('tagger_agreement')}   "
          f"confidence: {row.get('confidence_score')}")
    print(f"current {EDIT_COLUMN}: {row.get(EDIT_COLUMN)}   "
          f"manual_reviewed: {row.get('manual_reviewed', False)}")
    if bool(row.get("flagged", False)):
        note = row.get("flag_note")
        if str(note).strip() and str(note) != "nan":
            print("*** FLAGGED -- memo(s):")
            for line in str(note).splitlines():
                print(f"    {line}")
        else:
            print("*** FLAGGED (no memo) ***")
    if str(row.get("note", "")).strip():
        print(f"note: {row.get('note')}")
    print("-" * 78)
    print(get_context(mutations_csv_path, row, context_n))
    print("-" * 78)

    while True:
        choice = input(f"Genuinely erosion (radical form actually spoken)? "
                        f"[y]es / [n]o / [s]kip / [f]lag / [p]revious / [q]uit: ").strip().lower()
        if choice in ("y", "n", "s", "f", "p", "q"):
            return choice
        print("  please enter y, n, s, f, p, or q")


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


def save(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="mutations_*.csv file(s) or folder(s). "
                                              "Default: every mutations_*.csv under mutations/")
    ap.add_argument("--status", default=None,
                     help=f"comma-separated status values to review. "
                          f"Default: {','.join(EROSION_STATUSES)}")
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
    args = ap.parse_args()

    statuses = args.status.split(",") if args.status else EROSION_STATUSES
    csv_paths = find_mutation_csvs(args.files)
    if not csv_paths:
        print("No mutations_*.csv files found.")
        return

    if args.flagged_only:
        print(f"Found {len(csv_paths)} mutations CSV file(s). "
              f"Reviewing previously flagged rows only.")
    else:
        print(f"Found {len(csv_paths)} mutations CSV file(s). "
              f"Reviewing statuses: {statuses}")

    # Load everything, tag each row with its source file + original index
    # so decisions write back to the right place, and pick the review queue.
    loaded = {}   # path -> DataFrame
    queue = []    # list of (path, row_index)
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
        loaded[path] = df

        if args.flagged_only:
            mask = df["flagged"].fillna(False)
        else:
            mask = df["status"].isin(statuses) if "status" in df.columns else df[EDIT_COLUMN] == True
        if not args.review_all:
            mask = mask & (~df["manual_reviewed"].fillna(False))
        for idx in df[mask].index:
            queue.append((path, idx))

    if not queue:
        print("Nothing to review (empty filtered set, or everything's "
              "already been reviewed -- try --review-all).")
        return

    if args.sample is not None:
        rng = random.Random(args.seed)
        queue = rng.sample(queue, k=min(args.sample, len(queue)))

    print(f"{len(queue)} row(s) queued for review. "
          f"Progress autosaves after every decision -- safe to quit anytime.\n"
          f"[y]es/[n]o decide, [s]kip, [f]lag for later, [p]revious to go back, [q]uit.\n")

    reviewed_count = 0
    flagged_count  = 0
    pos = 0  # index into queue -- a plain counter instead of a for-loop so
             # [p]revious can move it backward as well as forward.
    while 0 <= pos < len(queue):
        path, idx = queue[pos]
        df = loaded[path]
        row = df.loc[idx]
        video_title = row.get("video_title", path.stem)

        print(f"\n[row {pos + 1}/{len(queue)}]")
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

        if choice == "s":
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
        save(df, path)
        reviewed_count += 1
        pos += 1

    print(f"\nDone. {reviewed_count} row(s) reviewed, "
          f"{flagged_count} flagged for follow-up this session.")


if __name__ == "__main__":
    main()