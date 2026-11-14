"""
corpus_analyzer.py

Standalone analyzer for Welsh mutation pipeline output.
Reads and merges all mutations_*.csv files from the pipeline's output
directory, produces publication-ready figures and a clean joined CSV
suitable for later joining with syntactic/phonetic datasets.

Usage:
    python corpus_analyzer.py

Output (all written to welsh_analysis/analysis/):
    - merged_mutations.csv          : all runs merged, deduped, with batch tag
    - video_formality.csv / speaker_formality.csv : F-score etc. (corpus_formality.py)
    - figures/erosion_by_type.png   : every expected_mutation label, all sources pooled
    - figures/erosion_by_mutation_family.png : soft / nasal / aspirate / h-prothesis, per corpus
    - figures/erosion_by_rule.png   : grammatical environment, per corpus
    - figures/erosion_by_channel.png: erosion rate per source channel
    - figures/codeswitch_by_channel.png: code-switch rate per channel
    - figures/status_distribution.png  : full status breakdown (stacked bar)
    - figures/tagger_agreement.png  : heuristic vs tagger agreement
    - figures/erosion_vs_formality.png, erosion_vs_codeswitch.png : mutation branch, per video
    - figures/branches_vs_formality.png        : every branch, rate vs F-score, per video
    - figures/branches_by_formality_band.png   : every branch pooled by F-score third
    - figures/speaker_branches_vs_formality.png, speaker_branches_by_formality_band.png :
      the same two, one point per Siarad speaker
    - utterance_export.csv          : flat utterance-level rows for future joining
"""

import re
import sys
import csv
import math
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- saves files without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

# ========================= CONFIGURATION =========================
# PATCH: previously hardcoded BASE_DIR = Path.home() / "Documents" /
# "welsh_analysis" here, completely independent of mutation_engine.py's
# BASE_DIR (which honors WELSH_ANALYSIS_DIR, falling back to
# Path.home() / "welsh_analysis" -- note: no "Documents"). Since
# welsh_pipeline.py imports its BASE_DIR from mutation_engine.py, the two
# had silently diverged into two different folders on disk -- this file
# would look for mutations data in a location the actual pipeline never
# wrote to. Importing the same names from mutation_engine.py makes this
# the single source of truth: change WELSH_ANALYSIS_DIR once, and every
# file (pipeline + all three companion tools) follows automatically.
from corpus_io import (BASE_DIR, MUT_DIR, OUT_DIR, FIG_DIR, pipeline_version,
                       is_current_version)
import corpus_formality
import run_progress

# Seaborn theme -- clean, publication-friendly.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
# A plain hyphen for negative numbers: Korean fonts (figures/ko/) may lack
# the Unicode minus sign and would print a box.
plt.rcParams["axes.unicode_minus"] = False

# Color palette keyed by mutation type for consistency across all figures.
MUTATION_COLORS = {
    "soft":        "#4C72B0",
    "nasal":       "#DD8452",
    "aspirate":    "#55A868",
    "h-mutation":  "#C44E52",
    "soft|aspirate":"#8172B2",
    "soft|nasal":  "#937860",
    "code_switch": "#9E9E9E",
    "other":       "#CCCCCC",
}

# Map raw status strings to human-readable labels for figures.
# NOTE: selective_invariancy is mapped to the same display label as
# correct_mutation because selective invariance IS grammatically correct --
# triggers like mor/cyn/rhy/un simply do not mutate certain word classes,
# so the absence of mutation is the expected form, not erosion.
# The raw 'status' column in the CSV still distinguishes the two if needed
# for finer-grained analysis.
STATUS_LABELS = {
    "correct_mutation":              "Correct",
    "selective_invariancy":          "Selective Mutation (correct)",
    "erosion":                       "Erosion (radical)",
    "wrong_mutation_type":           "Erosion (wrong type)",
    "mutation_mismatch":             "Mismatch",
    "phantom_mutation":              "Phantom",
    "code_switch":                   "Code-Switch",
    "mutation_absent_expected_none": "No mutation required",
    "colloquial_variant":            "Colloquial Variant",
}

STATUS_COLORS = {
    "Correct":                      "#55A868",
    "Selective Mutation (correct)": "#55A860",
    "Erosion (radical)":            "#DD8452",
    "Erosion (wrong type)":         "#C44E52",
    "Mismatch":                     "#8172B2",
    "Phantom":                      "#937860",
    "Code-Switch":                  "#9E9E9E",
    "No mutation required":         "#CCE5FF",
    "Colloquial Variant":           "#FFD700",
}

# Statuses where erosion is genuinely evaluable -- imported from
# mutation_tables.py, the single source of truth (see that file for the
# full rationale and exclusion list). Previously defined independently
# here and had drifted from corpus_ops.py's own separate copy.
from mutation_tables import EVALUABLE_STATUSES, SPACY_MUTATION_MAP

# Single source of truth for channel label mapping.
# Each entry is (full_url_substring, short_slug, display_name).
# Both CHANNEL_MAP (full URL match) and CHANNEL_MAP_SHORT (slug match) are
# derived from this list so adding a new channel only requires one edit here.
_CHANNEL_ENTRIES = [
    ("https://www.youtube.com/c/HanshS4C/videos",   "HanshS4C",    "Hansh"),
    ("https://www.youtube.com/@RowndaRownd/videos", "RowndaRownd", "Rownd a Rownd"),
    ("https://www.youtube.com/@S4C/videos",         "@S4C",        "S4C"),
    ("https://www.youtube.com/@Sgorio/videos",      "Sgorio",      "Sgorio"),
    ("https://www.youtube.com/@BBCRadio_Cymru/videos", "BBCRadio_Cymru", "BBC Cymru"),
    ("https://www.youtube.com/channel/UCNH53DFjL1OBl3oWNxMmC_Q/videos",
     "UCNH53DFjL1OBl3oWNxMmC_Q", "BBC Cymru Wales"),
    ("local",                                        "local",       "Local MP3"),
    # Podcasts (2026-10-08: Haclediad showed up as its feed URL in figures).
    ("feeds.fireside.fm/haclediad",                  "haclediad",   "Haclediad"),
    ("spreaker.com/show/5059223",                    "5059223",     "Colli'r Plot"),
    ("anchor.fm/s/1064d88dc",                        "1064d88dc",   "Pryd ar Dafod"),
    ("podcast-cwins",                                "cwins",       "Siarad Siop"),
]

_CHANNEL_ENTRIES.insert(0, ("https://www.youtube.com/@NewyddionS4C/videos",
                            "NewyddionS4C", "Newyddion S4C"))

CHANNEL_MAP       = {url: name  for url, _slug, name in _CHANNEL_ENTRIES}
CHANNEL_MAP_SHORT = {slug: name for _url, slug, name in _CHANNEL_ENTRIES}

# Non-YouTube sources, as they should read in figures and tables.
CORPUS_LABELS = {
    "siarad":         "Siarad",
    "patagonia":      "Patagonia",
    "corcencc":       "CorCenCC",
    "news-narration": "News articles: narration",
    "news-quote":     "News articles: quotes",
    "news-statement": "News articles: statements",
}


def source_label(source):
    """Display name for a row's "source": a corpus name, a known channel, or
    the @handle of any other YouTube channel URL -- never the raw URL, which
    is what the figures showed for Newyddion S4C before 2026-10-01."""
    s = str(source)
    if s in CORPUS_LABELS:
        return CORPUS_LABELS[s]
    named = next((v for k, v in CHANNEL_MAP.items() if k in s), None)
    if named:
        return named
    handle = re.search(r"youtube\.com/(?:@|c/|channel/)([^/?#]+)", s)
    return handle.group(1) if handle else s


def transcript_kind(source):
    """Legend label: the source plus how its text was made."""
    s = str(source)
    if s in ("siarad", "patagonia", "corcencc"):
        return f"{source_label(s)} (human transcript)"
    if s.startswith("news-"):
        return f"{source_label(s)} (text)"
    return f"{source_label(s)} (Whisper transcript)"


# ========================= BATCH SELECTION =========================
def interactive_batch_selection(csv_files):
    """
    Prints all discovered batch files with row counts and lets the user
    select which to permanently DELETE before analysis runs.
    Deleted files are moved to a _deleted/ subfolder (recoverable)
    rather than wiped outright, unless the user confirms hard delete.

    Returns the list of csv_files that should be loaded (survivors only).
    """
    if not csv_files:
        return csv_files

    print("\n" + "="*70)
    print("BATCH SELECTION")
    print("="*70)
    print("Scanning batches...\n")

    batch_info = []
    for f in csv_files:
        try:
            pd.read_csv(f, encoding="utf-8-sig", nrows=0)          # validate readable
            n_rows = sum(1 for _ in open(f, encoding="utf-8-sig")) - 1
            df_sample = pd.read_csv(
                f, encoding="utf-8-sig",
                usecols=lambda c: c in ("source", "video_title"),
                nrows=500,
            )
            sources = df_sample["source"].dropna().unique().tolist() \
                if "source" in df_sample.columns else []
            short_sources = list({
                next((v for k, v in CHANNEL_MAP_SHORT.items()
                      if k in str(s)), "Other")
                for s in sources
            })
        except Exception:
            n_rows        = "?"
            short_sources = ["(unreadable)"]

        # PATCH: filenames are now mutations_original_<stamp>_<slug>.csv
        # or mutations_corroborated_<stamp>_<slug>.csv, not just
        # mutations_<stamp>_<slug>.csv -- the optional non-capturing group
        # skips past whichever prefix (or neither, for legacy files) is
        # present before the actual stamp digits.
        m = re.search(r"mutations_(?:original_|corroborated_)?(\d{8}_\d{6})", f.name)
        stamp = m.group(1) if m else f.stem
        try:
            dt       = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
            readable = dt.strftime("%Y-%m-%d  %H:%M:%S")
        except Exception:
            readable = stamp

        batch_info.append({
            "index":    len(batch_info) + 1,
            "file":     f,
            "stamp":    stamp,
            "readable": readable,
            "n_rows":   n_rows,
            "sources":  short_sources,
        })

    # Print menu
    col_w = 6
    print(f"  {'#':<{col_w}}{'Date / Time':<22}{'Rows':<10}Channels")
    print(f"  {'-'*col_w}{'-'*22}{'-'*10}{'-'*30}")
    for b in batch_info:
        src_str = ", ".join(b["sources"]) if b["sources"] else "—"
        print(f"  {b['index']:<{col_w}}{b['readable']:<22}{str(b['n_rows']):<10}{src_str}")

    print()
    print("Enter the NUMBER(S) of batches to DELETE (comma-separated),")
    print("or press Enter to keep all and continue.\n")
    raw = input("Delete batches: ").strip()

    if not raw:
        print("No batches deleted. Keeping all.\n")
        return [b["file"] for b in batch_info]

    # Parse selection
    to_delete_indices = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(batch_info):
                to_delete_indices.add(idx)
            else:
                print(f"  ⚠️  Index {idx} out of range -- ignored.")

    if not to_delete_indices:
        print("No valid indices entered. Keeping all.\n")
        return [b["file"] for b in batch_info]

    # Confirm
    print()
    print("Batches marked for deletion:")
    for b in batch_info:
        if b["index"] in to_delete_indices:
            print(f"  [{b['index']}] {b['readable']}  ({b['n_rows']} rows)")

    print()
    print("Options:")
    print("  m = move to _deleted/ subfolder (recoverable)")
    print("  d = permanently delete (cannot be undone)")
    print("  c = cancel, keep all")
    action = input("Action (m / d / c): ").strip().lower()

    if action not in ("m", "d"):
        print("Cancelled. Keeping all batches.\n")
        return [b["file"] for b in batch_info]

    survivors   = []
    deleted_dir = MUT_DIR / "_deleted"
    log_entries = []  # collected and written once at the end

    for b in batch_info:
        if b["index"] not in to_delete_indices:
            survivors.append(b["file"])
            continue

        f = b["file"]
        if action == "m":
            deleted_dir.mkdir(exist_ok=True)
            dest = deleted_dir / f.name
            if dest.exists():
                dest = deleted_dir / f"{f.stem}__{b['stamp']}_dup{f.suffix}"
            # shutil.move() handles cross-device moves safely (Path.rename()
            # raises OSError if source and destination are on different filesystems).
            shutil.move(str(f), str(dest))
            print(f"  Moved:   {f.name}  →  _deleted/{dest.name}")
            log_entries.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  MOVED    "
                f"{f.name}  ({b['n_rows']} rows)  →  {dest.name}\n"
            )
        elif action == "d":
            f.unlink()
            print(f"  Deleted: {f.name}")
            log_entries.append(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  DELETED  "
                f"{f.name}  ({b['n_rows']} rows)\n"
            )

    # Write audit log so deletions are traceable without archaeology.
    if log_entries:
        deleted_dir.mkdir(exist_ok=True)
        log_path = deleted_dir / "deleted_log.txt"
        with log_path.open("a", encoding="utf-8") as lf:
            lf.writelines(log_entries)
        print(f"  Deletion log updated: _deleted/deleted_log.txt")

    print(f"\n{len(to_delete_indices)} batch(es) removed. "
          f"{len(survivors)} remaining for analysis.\n")
    return survivors


# ========================= DATA LOADING =========================
def load_and_merge_mutations():
    """
    Discovers all mutations_*.csv files in MUT_DIR, offers interactive
    batch deletion, then loads and merges the survivors.
    Adds a 'batch' column derived from the timestamp in each filename so
    individual runs can be tracked even after merging.
    Deduplicates on (video_url, timestamp, trigger_word, following_word)
    to avoid double-counting if a run was re-processed.
    Returns the merged DataFrame and a list of (batch_stamp, row_count) tuples.
    """
    # PATCH: mutation CSVs now live one level deeper, in a per-video
    # subfolder (runs/<stamp>/<slug>/mutations_original_<stamp>_<slug>.csv)
    # rather than flat in MUT_DIR, so this needs to recurse. "_deleted" is
    # excluded since files moved there by interactive_batch_selection()
    # should stay excluded from future runs, not silently reappear.
    #
    # PATCH: mutations_original_ and mutations_corroborated_ are now two
    # SEPARATE files per video (fetch_captions.run_corroboration() used
    # to overwrite the original in place, backing it up once to
    # "*_precaption_backup.csv" -- that backup pattern is gone now, the
    # original simply never gets touched). Both filenames match
    # "mutations_*.csv", so for research figures this picks ONE per video:
    # the corroborated version when it exists (it's a strict superset --
    # same classification columns, plus the caption cross-check), falling
    # back to the original when a video was never corroborated (e.g. no
    # captions were available for it). Without this, every corroborated
    # video would load twice and double-count its mutation rows in the
    # merged corpus. "*_precaption_backup.csv" is still excluded for any
    # older, not-yet-migrated file that predates this split.
    all_found = [
        p for p in MUT_DIR.rglob("mutations_*.csv")
        if "_deleted" not in p.parts and not p.name.endswith("_precaption_backup.csv")
    ]
    by_video = {}
    for p in all_found:
        by_video.setdefault(p.parent, []).append(p)
    csv_files = []
    for files in by_video.values():
        corroborated = [f for f in files if f.name.startswith("mutations_corroborated_")]
        original     = [f for f in files if f.name.startswith("mutations_original_")]
        legacy       = [f for f in files if f not in corroborated and f not in original]
        csv_files.extend(corroborated or original or legacy)
    csv_files = sorted(csv_files)
    if not csv_files:
        print(f"No mutations_*.csv files found in {MUT_DIR}")
        print("Run the pipeline first to generate mutation data.")
        sys.exit(1)

    # ---- Interactive batch selection / deletion ----
    csv_files = interactive_batch_selection(csv_files)

    if not csv_files:
        print("No batches remaining after deletion. Exiting.")
        sys.exit(0)

    print(f"Loading {len(csv_files)} batch(es)...")
    frames    = []
    batch_log = []
    for f in run_progress.files(csv_files, "Loading mutations"):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            # (?:original_|corroborated_)? -- without it this never matched the
            # current per-video filenames, and "batch" held the whole stem.
            m  = re.search(r"mutations_(?:original_|corroborated_)?(\d{8}_\d{6})(?:_(.+))?\.csv$",
                           f.name)
            batch_stamp = m.group(1) if m else f.stem
            video_slug  = m.group(2) if (m and m.group(2)) else ""
            # label column: "20260625_193337 · Hansh_clip_01" for per-video files,
            # plain stamp for legacy per-run files
            batch_label = f"{batch_stamp} · {video_slug}" if video_slug else batch_stamp
            df["batch"]       = batch_stamp   # timestamp only -- used for convergence chart x-axis
            df["video_file"]  = video_slug    # slug -- available for future per-video filtering
            frames.append(df)
            batch_log.append((batch_label, len(df)))
        except Exception as e:
            run_progress.tqdm.write(f"  ⚠️ Could not read {f.name}: {e}")

    if not frames:
        print("No readable CSV files found.")
        sys.exit(1)

    merged      = pd.concat(frames, ignore_index=True)
    initial_len = len(merged)

    # One run per video: a video processed more than once keeps only its
    # most recent run's rows (stamps sort chronologically). Key-based
    # de-duplication alone can't do this -- a false row that a detection fix
    # stopped producing has no newer duplicate, so the old run's copy would
    # survive; and it used keep="first", so even matching rows kept the
    # OLDEST verdict.
    if "video_url" in merged.columns and "batch" in merged.columns:
        latest = merged.groupby("video_url", dropna=False)["batch"].transform("max")
        merged = merged[merged["batch"] == latest]
        superseded = initial_len - len(merged)
        if superseded:
            print(f"  Dropped {superseded} rows from older runs of re-processed videos.")

    dedup_keys   = ["video_url", "timestamp", "trigger_word", "following_word"]
    present_keys = [k for k in dedup_keys if k in merged.columns]
    if present_keys:
        before  = len(merged)
        merged  = merged.drop_duplicates(subset=present_keys, keep="last")
        dropped = before - len(merged)
        if dropped:
            print(f"  Removed {dropped} duplicate rows.")

    print(f"\nMerged total: {len(merged)} unique mutation contexts "
          f"from {len(frames)} batch(es).\n")
    return merged, batch_log


# ========================= ASR DIVERGENCE (original vs. corroborated) =========================
# Discrepancy between mutations_original_ (Whisper's own transcript) and
# mutations_corroborated_ (cross-checked against YouTube's official captions)
# is NOT a measure of language change -- it's a proxy for ASR reliability.
# load_and_merge_mutations() above deliberately picks ONE file per video for
# the research corpus (corroborated when available); this section instead
# loads BOTH files for every video where both exist, so the two can be
# compared directly.
DIVERGENCE_JOIN_KEYS = ["video_url", "timestamp", "trigger_word", "following_word"]


def load_original_corroborated_pairs():
    """
    Finds every video with BOTH mutations_original_*.csv and
    mutations_corroborated_*.csv on disk, and loads both. Videos with only
    one of the two (never corroborated -- no YouTube captions were available
    for that video) are skipped, since there's nothing to diverge from.
    Returns a list of (video_dir, df_original, df_corroborated) tuples.
    """
    all_found = [
        p for p in MUT_DIR.rglob("mutations_*.csv")
        if "_deleted" not in p.parts and not p.name.endswith("_precaption_backup.csv")
    ]
    by_video = {}
    for p in all_found:
        by_video.setdefault(p.parent, []).append(p)

    pairs, skipped = [], 0
    for video_dir, files in run_progress.files(by_video.items(), "Loading caption pairs",
                                               unit="folders", total=len(by_video)):
        corroborated = next((f for f in files if f.name.startswith("mutations_corroborated_")), None)
        original     = next((f for f in files if f.name.startswith("mutations_original_")), None)
        if not (corroborated and original):
            skipped += 1
            continue
        try:
            df_orig = pd.read_csv(original, encoding="utf-8-sig")
            df_corr = pd.read_csv(corroborated, encoding="utf-8-sig")
        except Exception as e:
            print(f"  ⚠️ Could not read pair in {video_dir}: {e}")
            continue
        pairs.append((video_dir, df_orig, df_corr))

    print(f"  Loaded {len(pairs)} original/corroborated pair(s) "
          f"({skipped} video(s) skipped -- never corroborated, nothing to compare).")
    return pairs


def compute_asr_divergence(pairs):
    """
    For each (original, corroborated) pair, joins rows on
    DIVERGENCE_JOIN_KEYS and flags contexts where the erosion verdict
    (is_erosion) disagrees between the two. `direction` distinguishes the
    two ways this can go:
      - "orig_erosion_only":  the raw Whisper transcript alone read as
        erosion, but the caption-corroborated version did not -- plausibly
        a transcription artefact reading a standard form as eroded.
      - "corr_erosion_only":  only the caption-corroborated version reads
        as erosion -- plausibly Whisper silently "normalising" an eroded
        form back toward the standard spelling, which would mean erosion
        rates computed from mutations_original_ alone UNDERSTATE erosion.
    Callers should run prepare() on each side (for boolean-normalised
    is_erosion and the channel column) before passing pairs in here.
    Returns one row per joined mutation context, or an empty DataFrame if
    no pairs had joinable rows.
    """
    rows = []
    for video_dir, df_orig, df_corr in pairs:
        keys = [k for k in DIVERGENCE_JOIN_KEYS if k in df_orig.columns and k in df_corr.columns]
        if "is_erosion" not in df_orig.columns or "is_erosion" not in df_corr.columns or not keys:
            continue

        # Safety dedup -- each file should already be one row per context,
        # but a duplicate key would silently fan out the merge otherwise.
        d_orig = df_orig.drop_duplicates(subset=keys, keep="first")
        d_corr = df_corr.drop_duplicates(subset=keys, keep="first")

        merged = d_orig.merge(d_corr, on=keys, how="inner", suffixes=("_orig", "_corr"))
        if merged.empty:
            continue

        merged["diverges"] = merged["is_erosion_orig"] != merged["is_erosion_corr"]

        def _direction(row):
            o, c = row["is_erosion_orig"], row["is_erosion_corr"]
            if pd.isna(o) or pd.isna(c) or o == c:
                return pd.NA
            return "orig_erosion_only" if o else "corr_erosion_only"
        merged["direction"] = merged.apply(_direction, axis=1)

        merged["video_dir"] = str(video_dir)
        # PATCH (Phase 3): "channel_register" dropped from this loop --
        # that hand-assigned label is gone; "channel" (the source URL/
        # slug, unaffected by the removal) is all that's carried through
        # here now.
        for col in ("channel",):
            orig_c, corr_c = f"{col}_orig", f"{col}_corr"
            if orig_c in merged.columns:
                merged[col] = merged[orig_c].fillna(merged.get(corr_c))
            elif col in merged.columns:
                pass  # unsuffixed column already present (only one side had it)

        rows.append(merged)

    if not rows:
        print("  No original/corroborated pairs with joinable rows found.")
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def print_asr_divergence_summary(divergence_df):
    """
    Prints divergence rate overall, by direction, and broken down by
    channel and register -- the register breakdown is the one most likely
    to matter for the thesis argument (ASR instability concentrated in
    informal/casual speech vs. formal/scripted).
    """
    if divergence_df.empty:
        return
    total     = len(divergence_df)
    n_diverge = int(divergence_df["diverges"].fillna(False).sum())

    print("\n" + "="*70)
    print("ASR DIVERGENCE  (mutations_original_ vs. mutations_corroborated_)")
    print("="*70)
    print(f"Joined mutation contexts (both versions present) : {total:,}")
    print(f"Divergent erosion classification                  : {n_diverge:,} "
          f"({n_diverge/total:.1%})" if total else "")

    if "direction" in divergence_df.columns:
        print("\nDivergence direction:")
        for d, n in divergence_df["direction"].value_counts(dropna=True).items():
            print(f"  {d:<20} {n:>6,}")

    if "channel" in divergence_df.columns:
        print("\nDivergence rate by channel:")
        for ch, group in divergence_df.groupby("channel"):
            n    = len(group)
            rate = group["diverges"].fillna(False).mean()
            print(f"  {ch:<20} n={n:<6} divergence={rate:.1%}")

    # PATCH (Phase 3): "Divergence rate by register" removed -- that
    # hand-assigned label is gone. See corpus_formality.py for the
    # grounded, per-video replacement.
    print("="*70)


def validate_columns(df):
    """
    Warn loudly if expected columns are missing -- pipeline version mismatches
    or partial runs can produce CSVs with different schemas.
    """
    expected = [
        "status", "is_erosion", "is_code_switch", "expected_mutation",
        "mutation_found", "trigger_word", "following_word", "source",
        "tagger_agreement", "rule", "register_class",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"⚠️  Missing columns (may be from an older pipeline version): {missing}")
        print("    Some figures may be incomplete or skipped.\n")

    # Detect legacy boolean tagger_agreement schema and warn.
    if "tagger_agreement" in df.columns:
        sample = df["tagger_agreement"].dropna().map(
            lambda v: str(v).strip().lower() in ("true", "false", "1", "0")
        )
        if sample.any():
            n = int(sample.sum())
            print(f"⚠️  {n} rows have legacy boolean tagger_agreement values.")
            print("    These will be remapped in prepare() but consider deleting")
            print("    stale batches via the batch selection menu to avoid mixing schemas.\n")

    return missing


# ========================= DATA PREP =========================
def prepare(df):
    """
    Normalise dtypes and add derived columns used across multiple figures.
    """
    # PATCH: preserve NaN for missing values rather than coercing to False.
    # The previous lambda mapped NaN → False, which caused rows with missing
    # is_erosion values (truncated runs, schema mismatches) to be silently
    # counted as non-erosion, understating the erosion rate.  NaN rows are now
    # left as NaN and excluded naturally by pandas groupby/mean operations.
    def _to_bool_or_nan(x):
        s = str(x).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return pd.NA  # NaN, None, empty string → excluded from aggregation

    for col in ("is_erosion", "is_code_switch"):
        if col in df.columns:
            df[col] = df[col].map(_to_bool_or_nan).astype("boolean")

    # Normalise tagger_agreement: legacy CSVs stored this as a boolean
    # (True = agree, False = disagree). Current pipeline writes categorical
    # strings (all_three / cysill+heuristic / parser+heuristic / heuristic_only
    # / disputed / n/a). Remap booleans so old batches don't break the charts.
    if "tagger_agreement" in df.columns:
        def _normalise_tagger_agreement(v):
            s = str(v).strip().lower()
            if s in ("true", "1"):
                return "cysill+heuristic"   # best approximation of old True
            if s in ("false", "0"):
                return "disputed"            # best approximation of old False
            return v  # already a string label -- leave untouched
        legacy_mask = df["tagger_agreement"].map(
            lambda v: str(v).strip().lower() in ("true", "false", "1", "0")
        )
        if legacy_mask.any():
            n_legacy = int(legacy_mask.sum())
            print(f"  ⚠️  Remapping {n_legacy} legacy boolean tagger_agreement "
                  f"values from old CSV(s) -- consider deleting stale batches.")
            df.loc[legacy_mask, "tagger_agreement"] =                 df.loc[legacy_mask, "tagger_agreement"].map(_normalise_tagger_agreement)

    if "status" in df.columns:
        df["status_label"] = df["status"].map(
            lambda s: STATUS_LABELS.get(str(s), str(s))
        )

    if "source" in df.columns:
        df["channel"] = df["source"].map(source_label)

    return df


# ========================= FIGURE HELPERS =========================
# Print sizing (2026-10-09): the figures go into an A4 report, placed at the
# text width (~16 cm). A figure drawn 13 inches wide is shrunk to about half
# there, so its 7-9 pt labels printed at 3-5 pt. _fit_text_for_print raises
# every text to a floor that reads as these sizes ON PAPER, by role.
PRINT_WIDTH_IN = 6.3
PRINT_MIN_PT = {"title": 10.0, "axis": 8.5, "note": 6.5}


def _fit_text_for_print(fig):
    """Raises small text so that, once the figure is shrunk to PRINT_WIDTH_IN,
    titles print at >= 10 pt, axis labels / ticks / legends at >= 8.5 pt and
    annotations at >= 6.5 pt. Never shrinks anything."""
    from matplotlib.text import Text
    scale = max(1.0, fig.get_size_inches()[0] / PRINT_WIDTH_IN)
    roles = {}
    if fig._suptitle is not None:
        roles[id(fig._suptitle)] = "title"
    for ax in fig.axes:
        for t in (ax.title, ax._left_title, ax._right_title):
            roles[id(t)] = "title"
        for t in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            roles[id(t)] = "axis"
        leg = ax.get_legend()
        if leg is not None:
            for t in [*leg.get_texts(), leg.get_title()]:
                roles[id(t)] = "axis"
    for leg in fig.legends:
        for t in [*leg.get_texts(), leg.get_title()]:
            roles[id(t)] = "axis"
    for t in fig.findobj(Text):
        if not t.get_text():
            continue
        floor = PRINT_MIN_PT[roles.get(id(t), "note")] * scale
        if t.get_fontsize() < floor:
            t.set_fontsize(floor)
    try:
        fig.tight_layout()
    except Exception:
        pass


# ---- Korean copies of every figure (2026-10-09, for the Korean report) ----
# After the English figure is saved, its text is translated by KO_PHRASES
# (applied in order; regexes, so numbers inside a title survive) and the
# figure is saved again to figures/ko/<same name>. Corpus and channel names
# (CorCenCC, Siarad, Patagonia, Hansh, Newyddion S4C...) and Welsh examples
# stay as they are. Needs a Korean font (Malgun Gothic ships with Windows);
# without one the Korean copies are skipped with a note.
KO_FIG_DIR = FIG_DIR / "ko"
KO_PHRASES = [
    # ---- whole titles and axis labels ----
    (r"Mutation erosion by source and mutation type\n\(sources from most to least formal\)",
     "자료별·변이 유형별 자음 변이 침식률\n(격식적인 자료부터 순서대로)"),
    (r"Erosion rate \(95% Wilson CI\)\. Number = contexts; bars under (\d+) contexts left out",
     r"침식률 (95% Wilson 신뢰구간). 숫자 = 문맥 수, 문맥 \1개 미만 제외"),
    (r"Mutation erosion by mutation type, per corpus\n\(soft-limited counted as soft; ([\d,]+) contexts "
     r"where two types are possible left out\)",
     r"코퍼스별·변이 유형별 자음 변이 침식률\n(제한 연변이는 연변이로 집계; 두 유형이 모두 가능한 "
     r"문맥 \1개 제외)"),
    (r"Mutation erosion by grammatical environment, per corpus", "문법 환경별·코퍼스별 자음 변이 침식률"),
    (r"Erosion rate \(%, 95% Wilson CI; bars with < (\d+) contexts omitted\)",
     r"침식률 (%, 95% Wilson 신뢰구간; 문맥 \1개 미만 제외)"),
    (r"Mutation Erosion Rate by Expected Mutation Type", "기대 변이 유형별 자음 변이 침식률"),
    (r"Expected Mutation Type", "기대 변이 유형"),
    (r"Code-Switching Rate by Channel\n\(% of all words spoken that are English, whole transcript;\n"
     r"n = total words\)", "자료별 코드스위칭률\n(전체 단어 중 영어 단어의 비율; n = 전체 단어 수)"),
    (r"Code-Switch Rate \(%, whole transcript\)", "코드스위칭률 (%, 전사본 전체)"),
    (r"Code-Switch Rate \(%\)", "코드스위칭률 (%)"),
    (r"^Channel$", "자료"),
    (r"Evaluable Mutation Outcome Distribution by Channel, least eroded first\n.*",
     "자료별 변이 판정 결과의 구성 (침식률이 낮은 자료부터)\n(막대 오른쪽 κ = spaCy와 Cysill의 코헨의 카파)"),
    (r"% of evaluable mutation outcomes \(Correct / Erosion / Wrong type\)",
     "판정 가능한 변이 환경 중 비율 (올바른 변이 / 기본형 변이 침식 / 대체형 변이 침식)"),
    (r"([\d.]+)% eroded, n=", r"침식 \1%, n="),
    (r"Overall Heuristic vs Tagger Agreement", "코드 판독과 태거의 전체 일치"),
    (r"Heuristic–Tagger Agreement by Rule", "규칙별 일치율"),
    (r"Agreement Rate \(%\)", "일치율 (%)"),
    (r"Pipeline Validation: Heuristic vs Techiaith Tagger", "판정 코드와 태거(Cysill·spaCy)의 교차 검증"),
    (r"^Agreement type$", "일치 유형"),
    (r"^all_three$", "셋 모두 일치"), (r"^cysill\+heuristic$", "Cysill + 코드"),
    (r"^parser\+heuristic$", "spaCy + 코드"), (r"^disputed$", "불일치"),
    (r"^heuristic_only$", "코드 단독"),
    (r"Mutation Erosion Rate vs\. Formality Score", "격식 정도(F-점수)와 자음 변이 침식률"),
    (r"Mutation Erosion Rate vs\. Code-Switch Rate", "코드스위칭률과 자음 변이 침식률"),
    (r"\(per video, n=(\d+), Spearman ρ=([-\d.n/a ()]+); point size = context count\)",
     r"(문서 \1건, Spearman ρ=\2; 점의 크기 = 문맥 수)"),
    (r"logistic fit, weighted by contexts: ", "로지스틱 회귀(문맥 수 가중): "),
    (r"OR per 10 points", "10점당 오즈비"),
    (r"not fitted \(fewer than (\d+) erosions, or under 5 (videos|speakers)\)",
     r"추정하지 않음 (침식 \1건 미만 또는 단위 5개 미만)"),
    (r"not fitted", "추정하지 않음"),
    (r"Formality Score \(Heylighen-Dewaele F, higher = more formal\)",
     "격식 정도 (Heylighen–Dewaele F-점수, 높을수록 격식적)"),
    (r"F-score \(higher = more formal\)", "F-점수 (높을수록 격식적)"),
    (r"Erosion [Rr]ate \(%\)", "침식률 (%)"),
    (r"Erosion rate vs\. formality, per video \(point size = contexts\)",
     "규칙별 격식 정도와 침식률 (문서 단위; 점의 크기 = 문맥 수)"),
    (r"Erosion rate vs\. formality, per speaker \(point size = contexts\)",
     "규칙별 격식 정도와 침식률 (화자 단위; 점의 크기 = 문맥 수)"),
    (r"\(predicted to erode\)", "(침식 예측)"), (r"\(predicted to resist\)", "(유지 예측)"),
    (r"videos=(\d+), Spearman", r"문서 \1건, Spearman"),
    (r"speakers=(\d+), Spearman", r"화자 \1명, Spearman"),
    (r"no curve \(fewer than (\d+) erosions\)", r"곡선 없음 (침식 \1건 미만)"),
    (r"too few (video|speaker)s: (\d+) with >= (\d+) contexts", r"단위 부족: 문맥 \3개 이상인 단위 \2개"),
    (r"Odds ratio of erosion per 10 F-points, 95% CI \(log scale; left of 1 = less erosion "
     r"when more formal\)", "F-점수 10점당 침식 오즈비, 95% 신뢰구간\n(로그 스케일; 1보다 왼쪽 = 격식적일수록 침식 감소)"),
    (r"Formality effect within each corpus, per video\n.*",
     "코퍼스 내부의 격식 효과 (문서 단위)\n(코퍼스마다 따로 적합하여 코퍼스 간 차이가 섞이지 않음)"),
    (r"Formality effect within each corpus, per speaker\n.*",
     "코퍼스 내부의 격식 효과 (화자 단위)\n(코퍼스마다 따로 적합하여 코퍼스 간 차이가 섞이지 않음)"),
    (r" -- (.+?) \((\d+) videos\)", r" -- \1 (문서 \2건)"),
    (r" -- (.+?) \((\d+) speakers\)", r" -- \1 (화자 \2명)"),
    (r"Erosion by formality band, per branch \((\d+) videos split into thirds\)",
     r"격식 정도 3구간별·규칙별 침식률 (문서 \1건을 3등분)"),
    (r"Erosion by formality band, per branch \((\d+) speakers split into thirds\)",
     r"격식 정도 3구간별·규칙별 침식률 (화자 \1명을 3등분)"),
    (r"Erosion rate \(%, pooled, 95% Wilson CI\)", "침식률 (%, 합산, 95% Wilson 신뢰구간)"),
    (r"\(least formal third\)", "(하위 33%)"), (r"\(middle third\)", "(중간 33%)"),
    (r"\(most formal third\)", "(상위 33%)"),
    (r"\(erode\)", "(침식 예측)"), (r"\(resist\)", "(유지 예측)"),
    # ---- speaker background ----
    (r"Mutation erosion by speaker background\n\(numbers above bars = speakers\)",
     "화자 배경별 자음 변이 침식률\n(막대 위 숫자 = 화자 수)"),
    (r"Erosion rate \(95% CI\)", "침식률 (95% 신뢰구간)"),
    (r"^When Welsh was acquired$", "습득 시기"),
    (r"^Speaker age \(apparent time\)$", "나이"),
    (r"^Language the mother spoke$", "어머니의 언어"),
    (r"^Welsh by age 4$", "만 4세까지"), (r"^Welsh later\n\(school /\nadult\)$", "그 이후\n(학교·성인기)"),
    # line breaks: Korean labels are wider and ran into each other (2026-10-29)
    (r"^<30$", "30세\n미만"), (r"^30-49$", "30–\n49세"), (r"^50-69$", "50–\n69세"), (r"^70\+$", "70세\n이상"),
    (r"^Welsh$", "웨일스어"), (r"^both$", "두 언어\n모두"), (r"^English /\nSpanish$", "접촉 언어\n(영어·\n스페인어)"),
    # ---- legends, categories ----
    (r"^Mutation$", "변이 유형"),
    (r"^Soft$", "연변이"), (r"^Nasal$", "비음 변이"), (r"^Aspirate$", "기음 변이"),
    (r"^h-prothesis$", "h 첨가"),
    (r"^soft$", "연변이"), (r"^soft_limited$", "제한 연변이"), (r"^nasal$", "비음 변이"),
    (r"^aspirate$", "기음 변이"), (r"^h-mutation$", "h 첨가"),
    (r"^aspirate\|soft$", "기음/연변이"), (r"^soft_limited\|nasal$", "제한 연변이/비음"),
    (r"^nasal\|soft_limited$", "비음/제한 연변이"),
    (r"^Correct$", "올바른 변이"), (r"^Erosion \(radical\)$", "기본형 변이 침식"),
    (r"^Erosion \(wrong type\)$", "대체형 변이 침식"),
    (r"Trigger word: preposition, possessive, yn, a\.\.\.", "촉발어: 전치사, 소유격, yn, a 등"),
    (r"Article \+ feminine noun", "정관사 + 여성 명사"),
    (r"Feminine noun \+ adjective", "여성 명사 + 형용사"),
    (r"un \+ feminine noun", "un + 여성 명사"), (r"dau/dwy \+ noun", "dau/dwy + 명사"),
    (r"Numeral \+ blynedd/diwrnod", "수사 뒤 비음 변이"),   # the rule chart adds "(pum mlynedd)"
    (r"Direct object of inflected verb", "활용 동사의 직접 목적어"), (r"Vocative", "호격"),
    (r"Negative particle \+ verb", "부정 소사 뒤 혼합 변이"),   # the rule chart adds "(ni chaf, …)"
    (r"h- before a vowel", "모음 앞 h 첨가"),
    (r"Feminine 'ei'", "여성 소유격 ei"), (r"Preposed adjective \+ noun", "전치 형용사 + 명사"),
    (r"Feminine ordinal \+ noun", "여성 서수 + 명사"),
    (r"^word_trigger$", "촉발어"), (r"^fem_noun\+adjective$", "여성 명사 + 형용사"),
    (r"^mixed_mutation$", "혼합 변이 (부정 소사 뒤)"), (r"^numeral\+fem_noun$", "un + 여성 명사"),
    (r"^feminine_ordinal\+noun$", "여성 서수 + 명사"), (r"^preposed_adjective\+noun$", "전치 형용사 + 명사"),
    (r"^feminine_ei$", "여성 소유격 ei"), (r"^spacy_obj_soft$", "동사의 직접 목적어"),
    (r"^numeral_general\+noun$", "dau/dwy + 명사"), (r"^definite_article\+fem_noun$", "정관사 + 여성 명사"),
    (r"^decompounded_noun$", "분리된 복합명사"), (r"^phantom_check$", "불필요한 변이 점검"),
    (r"^restricted_nasal_numeral$", "수사 뒤 비음 변이 (blynedd/diwrnod)"), (r"^h_mutation$", "모음 앞 h 첨가"),
    (r"Conjugated prepositions", "전치사의 인칭 활용"), (r"Numeral \+ singular noun", "수사 + 단수 명사"),
    (r"rhai \+ plural noun", "rhai + 복수 명사"), (r"Quantifier \+ o \+ plural", "수량사 + o + 복수"),
    (r"Controls combined", "대조 규칙 통합"), (r"^Mutation\b", "자음 변이"),
    (r"News narration \(noise floor\)", "뉴스 본문 (측정 잡음의 기준선)"),
    (r"News articles: narration", "뉴스 본문"), (r"News articles: quotes", "뉴스 인용문"),
    (r"News articles: statements", "뉴스 성명"),
    (r"News\nnarration", "뉴스\n본문"), (r"News\nstatements", "뉴스\n성명"), (r"News\nquotes", "뉴스\n인용문"),
    (r"News narration", "뉴스 본문"), (r"News quotes", "뉴스 인용문"), (r"News statements", "뉴스 성명"),
    (r"YouTube videos", "유튜브 영상"),
]
_KO_COMPILED = [(re.compile(p, re.S), r) for p, r in KO_PHRASES]
# The report's own terms (read from the .hwp draft, 2026-10-11), applied
# after KO_PHRASES so the figures say exactly what the text says.
KO_REPORT_TERMS = [
    ("제한 연변이", "제한 연음 변이"), ("연변이", "연음 변이"),
    ("문맥", "변이 환경"),
    ("전치사의 인칭 활용", "전치사의 활용"), ("수사 + 단수 명사", "수사 단수 규칙"),
    ("rhai + 복수 명사", "rhai 규칙"), ("수량사 + o + 복수", "수량어 규칙"),
    ("대조 규칙 통합", "대조군 규칙 통합"),
    ("(침식 예측)", "(예측: 무너짐)"), ("(유지 예측)", "(예측: 유지됨)"),
    ("격식 정도 (Heylighen–Dewaele F-점수, 높을수록 격식적)",
     "격식 점수 F (Heylighen–Dewaele, 높을수록 격식적)"),
    ("격식 정도(F-점수)와", "격식 점수 F와"), ("F-점수 (높을수록 격식적)", "격식 점수 F (높을수록 격식적)"),
    ("F-점수", "격식 점수 F"),
    # 2026-10-29, matched to the final report (OneDrive copy saved 10-28)
    ("기대 변이 유형", "예상 변이 유형"),
    ("3구간별", "3분위별"), ("을 3등분)", "을 3분위로 나눔)"),
    ("코드 판독과 태거의 전체 일치", "코드와 태거의 전체 일치"),
    ("유튜브 영상", "영상·팟캐스트"),
    ("촉발어: 전치사, 소유격, yn, a 등", "촉발어 뒤 (전치사, 소유격, yn, a 등)"),
    ("정관사 + 여성 명사", "정관사 뒤 여성 명사"), ("여성 명사 + 형용사", "여성 명사 뒤 형용사"),
    ("여성 서수 + 명사", "여성 서수 뒤 명사"),
    ("활용 동사의 직접 목적어", "활용하는 동사의 목적어"), ("동사의 직접 목적어", "활용하는 동사의 목적어"),
]
_ko_font = None


def _korean_font():
    """Name of an installed Korean font, or "" when there is none."""
    global _ko_font
    if _ko_font is None:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        _ko_font = next((n for n in ("Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                                     "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR")
                         if n in names), "")
        if not _ko_font:
            print("  ⚠️ No Korean font found -- Korean figure copies (figures/ko/) skipped.")
    return _ko_font


def _ko(text):
    for pattern, repl in _KO_COMPILED:
        text = pattern.sub(repl, text)
    for old, new in KO_REPORT_TERMS:
        text = text.replace(old, new)
    return text


def _koreanize(fig, font):
    """Translates every text of fig in place (KO_PHRASES) and sets the font."""
    from matplotlib.text import Text
    from matplotlib.ticker import FixedFormatter, FixedLocator
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Tick labels are redrawn from the formatter, so editing their Text
            # objects does nothing. Read the labels the current formatter gives
            # (FixedFormatter from set_ticklabels, seaborn's category formatter,
            # ...), translate them, and pin them with a fixed locator/formatter.
            # Numeric axes come back unchanged and are left alone (2026-10-29:
            # category labels were still English in five figures).
            try:
                locs = list(axis.get_majorticklocs())
                old = list(axis.get_major_formatter().format_ticks(locs))
            except Exception:
                continue
            new = [_ko(s) for s in old]
            if new != old:
                axis.set_major_locator(FixedLocator(locs))
                axis.set_major_formatter(FixedFormatter(new))
        try:
            ax.tick_params(labelfontfamily=font)
        except (ValueError, TypeError):       # matplotlib < 3.8
            for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
                t.set_fontfamily(font)
    for t in fig.findobj(Text):
        orig = getattr(t, "_orig_text", None)
        if orig is not None:                  # a wrapped title: translate the original
            t._orig_text = _ko(orig)
            t.set_text(t._orig_text)
        else:
            s = t.get_text()
            if s:
                new = _ko(s)
                if new != s:
                    t.set_text(new)
        t.set_fontfamily(font)
    _wrap_titles(fig)
    try:
        fig.tight_layout()
    except Exception:
        pass


def _text_width_in(s, fontsize):
    """Rough printed width of s in inches: Hangul/CJK about one em per
    character, Latin about half an em (bold)."""
    em = sum(1.0 if ord(ch) > 0x2E80 else 0.56 for ch in s)
    return em * fontsize / 72


def _wrap_line(line, fontsize, width_in):
    words, out, cur = line.split(" "), [], ""
    for w in words:
        trial = f"{cur} {w}" if cur else w
        if cur and _text_width_in(trial, fontsize) > width_in:
            out.append(cur)
            cur = w
        else:
            cur = trial
    out.append(cur)
    return "\n".join(out)


def _wrap_titles(fig):
    """Wraps every title (axes titles, suptitle) to the width it sits over,
    so no title is wider than its chart (2026-10-10): saving with
    bbox_inches="tight" made the image as wide as the longest title, and the
    chart shrank when the image was fitted to the page. The unwrapped text is
    kept on the Text object (_orig_text) so the Korean copy translates the
    original and is wrapped again afterwards."""
    fig_w = fig.get_size_inches()[0]
    titles = []
    if fig._suptitle is not None:
        titles.append((fig._suptitle, fig_w * 0.96))
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        width = ax.get_position().width * fig_w
        for t in (ax.title, ax._left_title, ax._right_title):
            titles.append((t, max(width, 2.0)))
    for t, width in titles:
        orig = getattr(t, "_orig_text", None)
        if orig is None:
            orig = t.get_text()
            t._orig_text = orig
        if not orig:
            continue
        fs = t.get_fontsize()
        t.set_text("\n".join(_wrap_line(line, fs, width) for line in orig.split("\n")))


def _save(fig, name):
    """Saves a figure. If Windows won't let the old file be overwritten --
    usually because it's open in an image viewer, which gave "OSError:
    [Errno 22] Invalid argument" and stopped the whole analyzer on
    2026-10-01 -- the figure goes to a second name instead. Then saves the
    Korean copy to figures/ko/ (see KO_PHRASES)."""
    path = FIG_DIR / name
    _fit_text_for_print(fig)
    _wrap_titles(fig)
    try:
        fig.tight_layout()
    except Exception:
        pass
    try:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path.name}")
    except OSError as e:
        alt = path.with_name(f"{path.stem}_{datetime.now():%H%M%S}{path.suffix}")
        try:
            fig.savefig(alt, dpi=150, bbox_inches="tight")
            print(f"  ⚠️ Couldn't overwrite {path.name} ({e.strerror}) -- is it open in an "
                  f"image viewer? Saved as {alt.name} instead.")
        except OSError as e2:
            print(f"  ⚠️ Couldn't save {path.name}: {e2}")
    try:
        font = _korean_font()
        if font:
            KO_FIG_DIR.mkdir(parents=True, exist_ok=True)
            _koreanize(fig, font)
            fig.savefig(KO_FIG_DIR / name, dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"  ⚠️ Korean copy of {name} not saved: {e}")
    finally:
        plt.close(fig)


def _bar_with_counts(ax, x, y, counts, color="#4C72B0", fmt="{:.1f}%"):
    """
    Draws a horizontal bar chart with percentage labels and sample-size
    annotations (n=...) so figures are self-documenting for a reader
    unfamiliar with corpus size.

    n= labels are drawn inside the bar in white when the bar is wide enough
    to contain them; otherwise drawn just outside in a dark colour so they
    remain legible for short bars (e.g. rare mutation types).
    """
    bars = ax.barh(x, y, color=color, edgecolor="white", linewidth=0.5)
    max_val = max(y) if len(y) > 0 else 1
    for bar, val, n in zip(bars, y, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                fmt.format(val), va="center", ha="left", fontsize=9)
        # Use white text inside the bar only when bar is wide enough (>12% of
        # the axis range) -- otherwise fall back to a dark label outside.
        bar_w = bar.get_width()
        if max_val > 0 and (bar_w / max_val) > 0.12:
            ax.text(0.3, bar.get_y() + bar.get_height() / 2,
                    f"n={n}", va="center", ha="left",
                    fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(bar_w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"n={n}", va="center", ha="left",
                    fontsize=8, color="#333333")
    ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 1)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.invert_yaxis()


# ========================= INDIVIDUAL FIGURES =========================
def fig_erosion_by_type(df, filename_prefix=""):
    """
    Horizontal bar chart: erosion rate (%) per expected mutation type.
    Excludes code-switch rows and phantom rows (no expected mutation type).
    Colour-coded per mutation type for consistency with other figures.

    PATCH (Phase 3): the per-`register` split (formal/informal/casual,
    called 3x from main()) is gone -- channel_register was a hand-
    assigned, per-channel label. See corpus_formality.py for the grounded,
    per-video, continuous replacement; the formal/informal/casual
    comparison this project is built around now runs as an erosion-vs-
    formality-score regression instead of three separately-filtered
    copies of this figure. `filename_prefix` stays as a plain parameter
    (unused by main() currently, kept for any future caller that wants a
    differently-scoped copy of this figure without duplicating the
    function).
    """
    if "expected_mutation" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_by_type -- missing columns")
        return

    sub = df
    sub = sub[(sub["is_code_switch"].isin([False])) & (sub["trigger_word"] != "[OMITTED]")] \
        if "is_code_switch" in sub.columns else sub
    # Restrict to evaluable contexts only -- phantom, selective invariant,
    # and unverified rows inflate the denominator and distort the erosion rate.
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]

    grouped = sub.groupby("expected_mutation").agg(
        contexts=("is_erosion", "count"),
        erosion_rate=("is_erosion", "mean"),
    ).reset_index()
    grouped = grouped[grouped["contexts"] >= 5].sort_values("erosion_rate", ascending=True)

    if grouped.empty:
        print(f"  [skip] {filename_prefix}erosion_by_type -- insufficient data")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.6)))
    colors = [MUTATION_COLORS.get(m, MUTATION_COLORS["other"])
              for m in grouped["expected_mutation"]]
    _bar_with_counts(ax, grouped["expected_mutation"],
                     grouped["erosion_rate"] * 100,
                     grouped["contexts"], color=colors)
    ax.set_title("Mutation Erosion Rate by Expected Mutation Type",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Erosion Rate (%)")
    ax.set_ylabel("Expected Mutation Type")
    fig.tight_layout()
    _save(fig, f"{filename_prefix}erosion_by_type.png")


# ---- per-corpus comparisons (erosion_by_rule, erosion_by_mutation_family) ----
# Pooling every source into one bar mixes edited news text (~2%) with
# conversation (~20%), so these two figures show one bar per corpus group.
# The text bar is news NARRATION only: edited copy, so its rate in each
# environment is that environment's detector noise floor (2026-10-02: article +
# feminine noun 0.9%, trigger words 1.1%, but feminine noun + adjective 6.6% --
# "arolwg barn diweddar", where the adjective belongs to the first noun).
# Quotes and press statements are partly speech, so they're left out here.
CORPUS_GROUP_ORDER = ["News narration (noise floor)", "YouTube videos", "CorCenCC",
                      "Siarad", "Patagonia"]
CORPUS_GROUP_COLORS = {"News narration (noise floor)": "#9E9E9E", "YouTube videos": "#4C72B0",
                       "CorCenCC": "#8172B3", "Siarad": "#DD8452", "Patagonia": "#55A868"}
# Scatter plots colour every point by its source, with the same colours in
# every figure (asked for 2026-10-06) -- the corpus colours above, plus the
# two kinds of news document the bar charts leave out.
SCATTER_GROUP_ORDER = ["News narration", "News quotes", "News statements", "YouTube videos",
                       "CorCenCC", "Siarad", "Patagonia"]
SCATTER_GROUP_COLORS = {"News narration": CORPUS_GROUP_COLORS["News narration (noise floor)"],
                        "News quotes": "#937860", "News statements": "#DA8BC3",
                        "YouTube videos": CORPUS_GROUP_COLORS["YouTube videos"],
                        "CorCenCC": CORPUS_GROUP_COLORS["CorCenCC"],
                        "Siarad": CORPUS_GROUP_COLORS["Siarad"],
                        "Patagonia": CORPUS_GROUP_COLORS["Patagonia"]}
MIN_CELL_CONTEXTS = 20   # a corpus x category bar needs this many contexts

# Readable names for the mutation rules, with an example of the environment.
RULE_LABELS = {
    "word_trigger":               "Trigger word: preposition, possessive, yn, a...\n(i Gaerdydd, ei gath, yn fawr)",
    "definite_article+fem_noun":  "Article + feminine noun (y ferch)",
    "fem_noun+adjective":         "Feminine noun + adjective (merch dda)",
    "numeral+fem_noun":           "un + feminine noun (un ferch)",
    "numeral_general+noun":       "dau/dwy + noun (dau gi)",
    "restricted_nasal_numeral":   "Numeral + blynedd/diwrnod (pum mlynedd)",
    "spacy_obj_soft":             "Direct object of inflected verb (welais i gi)",
    "spacy_vocative_soft":        "Vocative (bore da, blant)",
    "mixed_mutation":             "Negative particle + verb (ni chaf, na fydd)",
    "h_mutation":                 "h- before a vowel (ei hafal, eu henwau)",
    "feminine_ei":                "Feminine 'ei' (ei chath, ei henw)",
    "preposed_adjective+noun":    "Preposed adjective + noun (hen ddyn)",
    "feminine_ordinal+noun":      "Feminine ordinal + noun (ail waith)",
}

# Four mutation families. soft_limited is soft mutation minus ll/rh, so it is
# soft; a context whose expectation spans two families (yn: soft or nasal;
# a + inflected verb: aspirate or soft) has no single family and is left out
# -- placing it by the mutation actually FOUND would send every correct case
# to a family and every eroded one nowhere, which biases every family down.
MUTATION_FAMILIES = {"soft": "Soft", "soft_limited": "Soft", "nasal": "Nasal",
                     "aspirate": "Aspirate", "h-mutation": "h-prothesis"}
FAMILY_ORDER = ["Soft", "Nasal", "Aspirate", "h-prothesis"]


def _corpus_group(source):
    s = str(source)
    if s == "news-narration":
        return "News narration (noise floor)"
    if s.startswith("news-"):
        return None     # quotes / statements: see CORPUS_GROUP_ORDER
    if s in ("siarad", "patagonia", "corcencc"):
        return CORPUS_LABELS[s]
    return "YouTube videos"


def _scatter_group(source):
    s = str(source)
    if s == "news-narration":
        return "News narration"
    if s == "news-quote":
        return "News quotes"
    if s == "news-statement":
        return "News statements"
    if s in ("siarad", "patagonia", "corcencc"):
        return CORPUS_LABELS[s]
    return "YouTube videos"


def _scatter_by_source(ax, pts, x, y, sizes):
    """Scatter of pts[x] vs pts[y], one colour per source (SCATTER_GROUP_COLORS),
    added in a fixed order so every figure's legend reads the same. sizes is a
    Series on pts' index. The caller draws the legend."""
    groups = pts["source"].map(_scatter_group)
    for name in SCATTER_GROUP_ORDER:
        sel = groups == name
        if sel.any():
            ax.scatter(pts.loc[sel, x], pts.loc[sel, y], s=sizes[sel], alpha=0.6,
                       color=SCATTER_GROUP_COLORS[name], label=f"{name} ({int(sel.sum())})",
                       clip_on=False)


def _mutation_family(expected):
    families = {MUTATION_FAMILIES.get(p.strip()) for p in str(expected).split("|")} - {None}
    return families.pop() if len(families) == 1 else None


def _evaluable_mutation_rows(df):
    """Mutation rows that count toward a rate, with a corpus-group column."""
    sub = df
    if "is_code_switch" in sub.columns:
        sub = sub[sub["is_code_switch"].isin([False])]
    if "rule" in sub.columns:
        sub = sub[sub["rule"] != "phantom_check"]
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]
    sub = sub.dropna(subset=["is_erosion"])
    sub = sub.assign(corpus=sub["source"].map(_corpus_group))
    return sub.dropna(subset=["corpus"]).assign(is_erosion=lambda d: d["is_erosion"].astype(bool))


def _corpus_comparison_chart(sub, cat_col, cat_order, cat_labels, title, filename):
    """Grouped horizontal bars: one group per category, one bar per corpus,
    erosion rate with a Wilson 95% interval and n. Cells under
    MIN_CELL_CONTEXTS contexts are left blank."""
    groups = [g for g in CORPUS_GROUP_ORDER if g in set(sub["corpus"])]
    # Only categories where at least one corpus has a bar -- an all-blank row
    # ("decompounded_noun") is just clutter.
    cell_sizes = sub.groupby(["corpus", cat_col]).size()
    cats = [c for c in cat_order
            if any(cell_sizes.get((g, c), 0) >= MIN_CELL_CONTEXTS for g in groups)]
    if not groups or not cats:
        print(f"  [skip] {filename} -- no data")
        return
    height = 0.8 / len(groups)
    # 8 in wide. 0.19 in per bar only for long charts, so the 60-bar rule
    # chart still fits one A4 page at text width (2026-10-09); shorter ones
    # (the 20-bar mutation-type chart printed too small) get 0.34 in.
    n_bars = len(cats) * len(groups)
    per_bar = 0.19 if n_bars > 40 else 0.34
    fig, ax = plt.subplots(figsize=(8, max(5, per_bar * n_bars + 1.8)))
    labelled, x_max = set(), 0.0
    for gi, group in enumerate(groups):
        for ci, cat in enumerate(cats):
            cell = sub[(sub["corpus"] == group) & (sub[cat_col] == cat)]
            n = len(cell)
            if n < MIN_CELL_CONTEXTS:
                continue
            k = int(cell["is_erosion"].sum())
            p = k / n
            low, high = _wilson_interval(k, n)
            y = ci + (gi - (len(groups) - 1) / 2) * height
            ax.barh(y, 100 * p, height=height * 0.9, color=CORPUS_GROUP_COLORS[group],
                    label=group if group not in labelled else None)
            labelled.add(group)
            ax.errorbar(100 * p, y, xerr=[[max(0.0, 100 * (p - low))], [max(0.0, 100 * (high - p))]],
                        fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
            ax.text(100 * high + 0.6, y, f"{100 * p:.1f}% (n={n:,})", va="center", fontsize=7)
            x_max = max(x_max, 100 * high)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([cat_labels.get(c, c) for c in cats], fontsize=9)
    ax.invert_yaxis()
    x_lim = min(x_max * 1.3 + 2, 112)   # room for the labels, but never a 120% axis
    ax.set_xlim(0, x_lim)
    if x_lim > 100:
        ax.xaxis.set_major_locator(mticker.FixedLocator(range(0, 101, 20)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel(f"Erosion rate (%, 95% Wilson CI; bars with < {MIN_CELL_CONTEXTS} contexts omitted)")
    ax.set_title(title, fontweight="bold", pad=12)
    # Short charts (mutation types): top right -- at lower right the legend
    # covered Patagonia's h-prothesis label (2026-10-10). The long rule chart
    # keeps lower right, where its bottom rows are short.
    # 2026-10-29: in the long rule chart the legend covered the bottom rows
    # (feminine ordinal), so long charts put it under the axis instead.
    if len(cats) <= 5:
        ax.legend(fontsize=9, loc="upper right")
    else:
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    fig.tight_layout()
    _save(fig, filename)


def fig_erosion_by_rule(df, filename_prefix=""):
    """
    Erosion rate per grammatical environment (detection rule), one bar per
    corpus -- which environments erode, and whether the same ones erode in
    every corpus. Rules are ordered by their Siarad rate (the main dataset),
    highest first. "word_trigger" bundles every trigger WORD (prepositions,
    possessives, yn, a/â...); the others are structural environments.
    """
    if "rule" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_by_rule -- missing columns")
        return
    sub = _evaluable_mutation_rows(df)
    siarad = sub[sub["corpus"] == "Siarad"].groupby("rule")["is_erosion"].agg(["mean", "size"])
    ranked = siarad[siarad["size"] >= MIN_CELL_CONTEXTS].sort_values("mean", ascending=False).index.tolist()
    order = ranked + sorted(set(sub["rule"]) - set(ranked))
    _corpus_comparison_chart(sub, "rule", order, RULE_LABELS,
                             "Mutation erosion by grammatical environment, per corpus",
                             f"{filename_prefix}erosion_by_rule.png")


def fig_erosion_by_mutation_family(df, filename_prefix=""):
    """
    Erosion rate per mutation family -- soft, nasal, aspirate, h-prothesis
    -- one bar per corpus. erosion_by_type keeps the raw expected_mutation
    labels (soft_limited, "nasal|soft_limited", ...); this one merges them:
    soft_limited is soft, and contexts where two families are possible are
    left out (see MUTATION_FAMILIES) and counted in the title.
    """
    if "expected_mutation" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_by_mutation_family -- missing columns")
        return
    sub = _evaluable_mutation_rows(df)
    sub = sub.assign(family=sub["expected_mutation"].map(_mutation_family))
    n_two = int(sub["family"].isna().sum())
    sub = sub.dropna(subset=["family"])
    _corpus_comparison_chart(
        sub, "family", FAMILY_ORDER, {},
        f"Mutation erosion by mutation type, per corpus\n"
        f"(soft-limited counted as soft; {n_two:,} contexts where two types are possible left out)",
        f"{filename_prefix}erosion_by_mutation_family.png")


# Sources in erosion_by_channel, most formal first, with short two-line
# labels; any source not listed here is appended after them.
CHANNEL_ORDER = ["News articles: narration", "News articles: statements",
                 "News articles: quotes", "Newyddion S4C", "CorCenCC", "Siarad",
                 "Patagonia", "Hansh", "Haclediad"]
CHANNEL_SHORT = {"News articles: narration": "News\nnarration",
                 "News articles: statements": "News\nstatements",
                 "News articles: quotes": "News\nquotes",
                 "Newyddion S4C": "Newyddion\nS4C"}
FAMILY_COLORS = {"Soft": MUTATION_COLORS["soft"], "Nasal": MUTATION_COLORS["nasal"],
                 "Aspirate": MUTATION_COLORS["aspirate"],
                 "h-prothesis": MUTATION_COLORS["h-mutation"]}


def _count_label(n):
    """Context count for a bar label: 429, 6.1k, 61k."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k" if n < 10000 else f"{n / 1000:.0f}k"


def fig_erosion_by_channel(df):
    """
    Erosion rate per source (channel), split by mutation family -- soft,
    nasal, aspirate, h-prothesis, as in erosion_by_mutation_family (contexts
    where two families are possible left out). Sources run from most formal
    (news narration) to least. The number over each bar is its context count;
    bars with fewer than MIN_CELL_CONTEXTS contexts are left out, so a missing
    bar means "too few to say", not 0%. Redrawn 2026-10-08 (was eight raw
    mutation labels per source, with rotated "n=" labels).
    """
    if "channel" not in df.columns or "is_erosion" not in df.columns \
            or "expected_mutation" not in df.columns:
        print("  [skip] erosion_by_channel -- missing columns")
        return
    # Same filter as _evaluable_mutation_rows, minus its corpus grouping
    # (which drops news quotes/statements -- they get their own bars here).
    sub = df
    if "is_code_switch" in sub.columns:
        sub = sub[sub["is_code_switch"].isin([False])]
    if "rule" in sub.columns:
        sub = sub[sub["rule"] != "phantom_check"]
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]
    sub = sub.dropna(subset=["is_erosion", "channel"])
    sub = sub.assign(is_erosion=sub["is_erosion"].astype(bool),
                     family=sub["expected_mutation"].map(_mutation_family)).dropna(subset=["family"])
    if sub.empty:
        print("  [skip] erosion_by_channel -- insufficient data")
        return

    present = set(sub["channel"])
    channels = [c for c in CHANNEL_ORDER if c in present] + sorted(present - set(CHANNEL_ORDER))
    families = [f for f in FAMILY_ORDER if f in set(sub["family"])]
    stats = sub.groupby(["channel", "family"])["is_erosion"].agg(["sum", "size"])

    # Horizontal bars (2026-10-09, for A4 print): one group per source, top =
    # most formal; a group's four bars run soft -> h-prothesis downwards. The
    # count sits right of each bar's interval, where there is room for it.
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(channels) * len(families) / 2 + 2))
    height = 0.8 / len(families)
    labelled = set()
    for fi, fam in enumerate(families):
        offset = (fi - (len(families) - 1) / 2) * height
        for ci, ch in enumerate(channels):
            if (ch, fam) not in stats.index:
                continue
            k, n = int(stats.loc[(ch, fam), "sum"]), int(stats.loc[(ch, fam), "size"])
            if n < MIN_CELL_CONTEXTS:
                continue
            p = k / n
            low, high = _wilson_interval(k, n)
            y = ci + offset
            ax.barh(y, 100 * p, height=height * 0.9, color=FAMILY_COLORS[fam],
                    label=fam if fam not in labelled else None)
            labelled.add(fam)
            ax.errorbar(100 * p, y, xerr=[[max(0.0, 100 * (p - low))], [max(0.0, 100 * (high - p))]],
                        fmt="none", ecolor="#555555", capsize=1.5, linewidth=0.7)
            ax.text(100 * high + 1.2, y, _count_label(n), ha="left", va="center",
                    fontsize=8, color="#444444")

    for ci in range(1, len(channels)):
        ax.axhline(ci - 0.5, color="#e3e3e3", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels([CHANNEL_SHORT.get(c, c) for c in channels])
    ax.set_ylim(len(channels) - 0.5, -0.5)          # most formal on top
    ax.set_xlim(0, 112)
    ax.xaxis.set_major_locator(mticker.FixedLocator(range(0, 101, 20)))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Erosion rate (95% Wilson CI). Number = contexts; "
                  f"bars under {MIN_CELL_CONTEXTS} contexts left out")
    ax.set_title("Mutation erosion by source and mutation type\n(sources from most to least formal)",
                 fontweight="bold")
    ax.legend(title="Mutation", loc="lower right", frameon=True)
    fig.tight_layout()
    _save(fig, "erosion_by_channel.png")


def fig_codeswitch_by_channel(df):
    """
    Horizontal bar chart: code-switch rate per channel -- share of all
    words spoken (not just mutation-trigger contexts) that are English.
    """
    if "channel" not in df.columns:
        print("  [skip] codeswitch_by_channel -- missing columns")
        return

    # video_word_count and video_codeswitch_word_count are stamped once
    # per video onto every row from that video (same pattern as
    # video_duration_seconds), computed from the classifier
    # is_english_code_switch() already runs on every word in the
    # transcript (see corpus_ops.py's analyze()) -- so this is a genuine
    # whole-transcript rate, not limited to mutation-trigger sites. Both
    # dedupe by video_url per channel before summing -- otherwise a video
    # with more trigger-context rows would silently inflate its own
    # channel's totals.
    if not {"video_word_count", "video_codeswitch_word_count", "video_url"} <= set(df.columns):
        print("  [skip] codeswitch_by_channel -- missing video_word_count/"
              "video_codeswitch_word_count/video_url (older data predating "
              "these fields, or rerun_rules.py output, which doesn't carry "
              "them through yet -- reprocess to get this figure)")
        return

    video_level = df.dropna(subset=["video_url"]).drop_duplicates("video_url")
    grouped = video_level.groupby("channel").agg(
        total_words=("video_word_count", "sum"),
        cs_words=("video_codeswitch_word_count", "sum"),
    ).reset_index()
    grouped = grouped[grouped["total_words"] > 0]
    grouped["cs_rate"] = grouped["cs_words"] / grouped["total_words"]
    grouped = grouped.sort_values("cs_rate", ascending=True)

    if grouped.empty:
        print("  [skip] codeswitch_by_channel -- insufficient data")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.6)))
    _bar_with_counts(ax, grouped["channel"], grouped["cs_rate"] * 100,
                     grouped["total_words"], color=MUTATION_COLORS["code_switch"],
                     fmt="{:.2f}%")
    ax.set_title("Code-Switching Rate by Channel\n"
                 "(% of all words spoken that are English, whole transcript;\n"
                 "n = total words)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Code-Switch Rate (%)")
    ax.set_ylabel("Channel")
    fig.tight_layout()
    _save(fig, "codeswitch_by_channel.png")


def _cohens_kappa(cat_a, cat_b):
    """
    Cohen's Kappa: chance-corrected inter-rater agreement between two
    independent categorical raters on the same items.

        kappa = (p_o - p_e) / (1 - p_e)

    p_o = observed proportion of agreement (raw % of rows where both
          taggers assigned the same category).
    p_e = proportion of agreement expected by chance alone, computed from
          each rater's own marginal label distribution -- i.e. how often
          they'd agree if each were independently guessing according to
          how often it uses each label, with no actual signal between
          them.

    A plain agreement percentage can look high just because one category
    (e.g. "no mutation") dominates and both taggers default to it often --
    that's not evidence of real corroboration. Kappa corrects for exactly
    that: kappa=1 is perfect agreement, kappa=0 is no better than the
    chance level implied by their own label frequencies, negative is
    worse than chance (systematic disagreement).

    cat_a, cat_b: paired sequences (same length, same row order) of
    category labels, one entry per item both raters scored.
    Returns (kappa, n) -- n is the number of paired observations kappa
    was computed over, since kappa on a small n is a noisy estimate and
    the caller should show it alongside the number, not alone.
    """
    n = len(cat_a)
    if n == 0:
        return None, 0
    cat_a = list(cat_a)
    cat_b = list(cat_b)
    labels = set(cat_a) | set(cat_b)
    po = sum(a == b for a, b in zip(cat_a, cat_b)) / n
    freq_a = {l: cat_a.count(l) / n for l in labels}
    freq_b = {l: cat_b.count(l) / n for l in labels}
    pe = sum(freq_a[l] * freq_b[l] for l in labels)
    if pe >= 1.0:
        # Every rater used exactly one label between them, in lockstep --
        # no variance to correct for; observed agreement IS the answer.
        return (1.0 if po == 1.0 else 0.0), n
    return (po - pe) / (1 - pe), n


def fig_status_distribution(df):
    """
    Stacked horizontal bar chart: evaluable mutation-outcome breakdown per
    channel (Correct / Erosion (radical) / Erosion (wrong type) only).

    PATCH: previously stacked all 9 statuses together (including
    phantom_mutation, code_switch, selective_invariancy, colloquial_variant,
    mutation_mismatch), which mixed genuinely-evaluable erosion outcomes
    with structurally different categories into one 100%-stacked bar.
    Now restricted to the 3 statuses where an erosion outcome was actually
    possible:
      - phantom_mutation is dropped entirely, not merged into "Correct" --
        it's a different detection path (no trigger word at all) that can
        only ever represent a correctly-produced mutation by construction,
        so it carries no erosion-risk information for this chart.
      - mutation_mismatch (tagger disagreement) is dropped from the bar
        too, but not discarded -- it's still real data about whether the
        classification itself is trustworthy, just a different kind of
        signal than "did erosion happen." Shown instead as a separate
        per-channel Cohen's Kappa annotation (spaCy vs Cysill agreement,
        chance-corrected -- see _cohens_kappa()) to the right of each bar,
        purple, rather than folded into the bar as a raw percentage.
      - code_switch, selective_invariancy, colloquial_variant,
        mutation_absent_expected_none stay excluded, same as before (none
        of these were ever includable erosion outcomes).
    Draws bars via mpatches.Rectangle to avoid a matplotlib/numpy linewidth
    scalar bug that affects ax.barh in some version combinations.
    """
    if "status" not in df.columns or "channel" not in df.columns:
        print("  [skip] status_distribution -- missing columns")
        return

    BAR_STATUSES = ["correct_mutation", "erosion", "wrong_mutation_type"]
    bar_df = df[df["status"].isin(BAR_STATUSES)]
    if bar_df.empty:
        print("  [skip] status_distribution -- no evaluable bar-eligible rows")
        return

    pivot = bar_df.groupby(["channel", "status"]).size().unstack(fill_value=0)
    pivot = pivot.groupby(level=0).sum()
    for s in BAR_STATUSES:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot     = pivot[BAR_STATUSES]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    # Least eroded at the top, most eroded at the bottom (user, 2026-10-02):
    # the bars then read as a formality ladder, not an alphabetical list.
    pivot_pct = pivot_pct.sort_values("correct_mutation", ascending=False)
    pivot     = pivot.loc[pivot_pct.index]

    # PATCH: was a plain mismatch-rate percentage (mismatch count / total
    # classification-attempted count). That's not a real credibility
    # measure -- it doesn't correct for the fact that two taggers can
    # agree (or "disagree" via one dominant label) by chance alone. Now
    # uses Cohen's Kappa between spaCy and Cysill directly (see
    # _cohens_kappa() docstring for the formula and rationale) --
    # normalizes each tagger's raw mutation-type column (spacy_mutation:
    # SM/NM/AM morph tags via SPACY_MUTATION_MAP; cysill_mutation_type:
    # already plain-text soft/nasal/aspirate/none) onto the same category
    # space, restricted to rows where BOTH taggers were actually
    # consulted (cysill_pos and spacy_pos both present -- rows where one
    # tagger never weighed in aren't a disagreement, they're missing
    # data, and including them would understate agreement for the wrong
    # reason) and to EVALUABLE_STATUSES, same universe as the bar itself.
    kappa_eligible = df[
        df["status"].isin(EVALUABLE_STATUSES)
        & df.get("cysill_pos", pd.Series(dtype=object)).notna()
        & ~df.get("cysill_pos", pd.Series(dtype=object)).isin(["none", "n/a"])
        & df.get("spacy_pos", pd.Series(dtype=object)).notna()
        & ~df.get("spacy_pos", pd.Series(dtype=object)).isin(["none", "n/a"])
    ] if {"cysill_pos", "spacy_pos", "cysill_mutation_type", "spacy_mutation"} <= set(df.columns) else df.iloc[0:0]

    channel_kappa = {}
    for ch, g in kappa_eligible.groupby("channel"):
        cysill_cat = g["cysill_mutation_type"].fillna("none").replace({"n/a": "none"})
        spacy_cat  = g["spacy_mutation"].map(lambda m: SPACY_MUTATION_MAP.get(m, "none"))
        channel_kappa[ch] = _cohens_kappa(cysill_cat.tolist(), spacy_cat.tolist())

    all_cols = BAR_STATUSES
    channels = list(pivot_pct.index)
    n        = len(channels)
    bar_h    = 0.6
    fig, ax  = plt.subplots(figsize=(9, max(3, n * 0.8)))

    legend_handles = [mpatches.Patch(facecolor=STATUS_COLORS.get(STATUS_LABELS[s], "#CCCCCC"),
                                      label=STATUS_LABELS[s])
                       for s in all_cols]

    for row_idx in range(n):
        left = 0.0
        for col_idx, status in enumerate(all_cols):
            # .iat[row, col] always returns a scalar by integer position --
            # immune to duplicate index/column issues that cause .at/.loc to
            # return a Series instead of a scalar.
            val   = float(pivot_pct.iat[row_idx, col_idx])
            color = STATUS_COLORS.get(STATUS_LABELS[status], "#CCCCCC")
            rect  = mpatches.Rectangle(
                (left, row_idx - bar_h / 2), val, bar_h,
                facecolor=color, edgecolor="white", linewidth=0.3
            )
            ax.add_patch(rect)
            left += val
        # Credibility annotation, outside the 100%-stacked area: Cohen's
        # Kappa between spaCy and Cysill, not a stacked segment.
        ch = channels[row_idx]
        kappa, kappa_n = channel_kappa.get(ch, (None, 0))
        label = f"κ = {kappa:.2f} (n={kappa_n})" if kappa is not None else "κ = n/a"
        ax.text(102, row_idx, label,
                va="center", ha="left", fontsize=8, color="#8E44AD", fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    eroded = 100 - pivot_pct["correct_mutation"]
    ax.set_yticklabels([f"{ch}\n{eroded[ch]:.1f}% eroded, n={int(pivot.loc[ch].sum()):,}"
                        for ch in channels])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel("% of evaluable mutation outcomes (Correct / Erosion / Wrong type)")
    ax.set_title("Evaluable Mutation Outcome Distribution by Channel, least eroded first\n"
                 "(phantom omissions excluded -- correct by construction; "
                 "\u03ba = Cohen's Kappa, spaCy vs Cysill, shown separately, right of each bar)",
                 fontweight="bold", pad=12)
    # Below the axis: at the top right it covered the first bar's kappa label.
    ax.legend(handles=legend_handles, bbox_to_anchor=(0.5, -0.09),
              loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    _save(fig, "status_distribution.png")


def fig_tagger_agreement(df):
    """
    Pie chart + bar chart: rate at which the pipeline's own heuristic
    agrees with Techiaith's tagger on mutation type.
    """
    if "tagger_agreement" not in df.columns:
        print("  [skip] tagger_agreement -- missing column")
        return

    sub    = df[df["tagger_agreement"].notna()]
    counts = sub["tagger_agreement"].value_counts()

    if counts.empty:
        print("  [skip] tagger_agreement -- no data")
        return

    # Stacked, not side by side (2026-10-09): readable at A4 text width.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [1, 1.25]})

    # Colour palette for the current pipeline vocabulary.
    # all_three = all three sources agree (best)
    # cysill+heuristic / parser+heuristic = two-way agreement
    # heuristic_only = tagger(s) absent, heuristic ran alone
    # disputed = taggers present but disagree with heuristic
    colors_pie = {
        "all_three":        "#55A868",
        "cysill+heuristic": "#4C72B0",
        "parser+heuristic": "#8172B2",
        "heuristic_only":   "#9E9E9E",
        "disputed":         "#C44E52",
        "n/a":              "#CCCCCC",
    }
    pie_colors = [colors_pie.get(k, "#CCCCCC") for k in counts.index]
    # PATCH: inline labels (the old `labels=counts.index` kwarg) collide and
    # overlap when several slices are small and adjacent (e.g. disputed,
    # parsed+heuristic, all_three, n/a all bunched near the top). Drop inline
    # labels entirely and use a legend instead -- percentages stay on the
    # wedges via autopct, but only for slices large enough to read (>=2%).
    wedges, _, autotexts = ax1.pie(
        counts.values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        colors=pie_colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax1.legend(wedges, counts.index, title="Agreement type",
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax1.set_title("Overall Heuristic vs Tagger Agreement", fontweight="bold")

    if "rule" in df.columns:
        # Treat all_three / cysill+heuristic / parser+heuristic as agreement;
        # disputed as disagreement.  heuristic_only and n/a are excluded.
        agreement_values = {"all_three", "cysill+heuristic", "parser+heuristic", "disputed"}
        rule_agree = sub[sub["tagger_agreement"].isin(agreement_values)] \
            .groupby("rule")["tagger_agreement"] \
            .apply(lambda x: (x.isin(["all_three", "cysill+heuristic",
                                       "parser+heuristic"])).mean() * 100).reset_index()
        rule_agree.columns = ["rule", "agree_pct"]
        rule_agree = rule_agree.sort_values("agree_pct", ascending=True)

        if not rule_agree.empty:
            ax2.barh(rule_agree["rule"], rule_agree["agree_pct"],
                     color="#4C72B0", edgecolor="white")
            ax2.set_xlim(0, 105)
            ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax2.set_xlabel("Agreement Rate (%)")
            ax2.set_title("Heuristic–Tagger Agreement by Rule", fontweight="bold")
            ax2.invert_yaxis()
            for i, (_, row) in enumerate(rule_agree.iterrows()):
                ax2.text(row["agree_pct"] + 1, i, f"{row['agree_pct']:.1f}%",
                         va="center", fontsize=8)
        else:
            ax2.set_visible(False)
    else:
        ax2.set_visible(False)

    fig.suptitle("Pipeline Validation: Heuristic vs Techiaith Tagger",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "tagger_agreement.png")


# fig_collision_flags (a bar per word in mutation_tables.KNOWN_HOMOGRAPH_
# COLLISIONS) was removed 2026-10-02: the list only ever held "chi", and its
# 18 rows were all one bug -- pronoun "chi" scored as an object -- fixed at
# the source (mutation_engine._is_object_candidate). The count is still
# printed in the summary ("Homograph collision flags").


# ========================= FORMALITY (Phase 4) =========================
def _per_video_erosion_rates(df, min_contexts=5):
    """
    One row per video_url: erosion rate + context count, restricted to
    EVALUABLE_STATUSES (same discipline as every other erosion-rate
    figure in this file). min_contexts=5 matches the low-N guard already
    used elsewhere (e.g. fig_erosion_by_type's `contexts >= 5`) -- a
    video with only a handful of mutation contexts gives too noisy a
    per-video rate to plot as a single point.
    """
    sub = df
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]
    if sub.empty or "video_url" not in sub.columns or "is_erosion" not in sub.columns:
        return pd.DataFrame()
    per_video = sub.groupby("video_url").agg(
        erosion_rate=("is_erosion", "mean"),
        contexts=("is_erosion", "count"),
    ).reset_index()
    return per_video[per_video["contexts"] >= min_contexts]


def fig_erosion_vs_formality(df, formality_df):
    """
    Scatter + logistic curve (_logistic_fit): per-video erosion rate vs. formality
    score (Heylighen-Dewaele F-score, corpus_formality.py) -- the direct
    test of the formal/informal/casual comparison this project is built
    around, now against a grounded, continuous, per-video measurement
    instead of channel_register's hand-assigned per-channel label.
    Replaces the old formal_/informal_/casual_ erosion_by_type/
    erosion_by_rule/erosion_over_batches figure sets.
    """
    if formality_df.empty or "f_score" not in formality_df.columns:
        print("  [skip] erosion_vs_formality -- no formality data")
        return
    per_video = _per_video_erosion_rates(df)
    if per_video.empty:
        print("  [skip] erosion_vs_formality -- no evaluable per-video erosion rates")
        return

    cols = ["video_url", "f_score"] + (["source"] if "source" in formality_df.columns else [])
    merged = per_video.merge(formality_df[cols], on="video_url", how="inner")
    merged = merged.dropna(subset=["f_score"])
    if len(merged) < 3:
        print(f"  [skip] erosion_vs_formality -- need at least 3 videos with both "
              f"an erosion rate and a formality score (found {len(merged)})")
        return
    merged["erosion_rate_pct"] = merged["erosion_rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    if "source" in merged.columns:
        _scatter_by_source(ax, merged, "f_score", "erosion_rate_pct", merged["contexts"].clip(upper=200))
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.scatter(merged["f_score"], merged["erosion_rate_pct"],
                   s=merged["contexts"].clip(upper=200), alpha=0.6, clip_on=False)
    fit = _draw_logistic(ax, merged["f_score"], merged["erosion_rate"], merged["contexts"])
    rho = _spearman_text(merged["f_score"], merged["erosion_rate_pct"])
    _rate_axis_from_zero(ax)
    ax.set_xlabel("Formality Score (Heylighen-Dewaele F, higher = more formal)")
    ax.set_ylabel("Erosion Rate (%)")
    ax.set_title(f"Mutation Erosion Rate vs. Formality Score\n"
                 f"(per video, n={len(merged)}, Spearman ρ={rho}; point size = context count)\n"
                 f"logistic fit, weighted by contexts: {fit or 'not fitted'}",
                 fontweight="bold", pad=12, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()
    _save(fig, "erosion_vs_formality.png")


def fig_erosion_vs_codeswitch(df, formality_df):
    """
    Scatter + logistic curve (_logistic_fit): per-video erosion rate vs. whole-
    transcript code-switch rate -- reported as its OWN figure, separate
    from formality (see corpus_formality.py's module docstring for why):
    a real, independent contact-intensity signal, but mixing it into the
    formality composite would muddy whether an erosion-formality
    correlation reflects formality itself or just how much English is in
    the room.
    """
    if formality_df.empty or "codeswitch_rate" not in formality_df.columns:
        print("  [skip] erosion_vs_codeswitch -- no formality data")
        return
    per_video = _per_video_erosion_rates(df)
    if per_video.empty:
        print("  [skip] erosion_vs_codeswitch -- no evaluable per-video erosion rates")
        return

    cols = ["video_url", "codeswitch_rate"] + (["source"] if "source" in formality_df.columns else [])
    merged = per_video.merge(formality_df[cols], on="video_url", how="inner")
    merged = merged.dropna(subset=["codeswitch_rate"])
    if len(merged) < 3:
        print(f"  [skip] erosion_vs_codeswitch -- need at least 3 videos with both "
              f"an erosion rate and a code-switch rate (found {len(merged)})")
        return
    merged["erosion_rate_pct"]    = merged["erosion_rate"] * 100
    merged["codeswitch_rate_pct"] = merged["codeswitch_rate"] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    if "source" in merged.columns:
        _scatter_by_source(ax, merged, "codeswitch_rate_pct", "erosion_rate_pct",
                           merged["contexts"].clip(upper=200))
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.scatter(merged["codeswitch_rate_pct"], merged["erosion_rate_pct"],
                   s=merged["contexts"].clip(upper=200), alpha=0.6, clip_on=False)
    fit = _draw_logistic(ax, merged["codeswitch_rate_pct"], merged["erosion_rate"], merged["contexts"])
    rho = _spearman_text(merged["codeswitch_rate_pct"], merged["erosion_rate_pct"])
    _rate_axis_from_zero(ax)
    ax.set_xlabel("Code-Switch Rate (%, whole transcript)")
    ax.set_ylabel("Erosion Rate (%)")
    ax.set_title(f"Mutation Erosion Rate vs. Code-Switch Rate\n"
                 f"(per video, n={len(merged)}, Spearman ρ={rho}; point size = context count)\n"
                 f"logistic fit, weighted by contexts: {fit or 'not fitted'}",
                 fontweight="bold", pad=12, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()
    _save(fig, "erosion_vs_codeswitch.png")


# ========================= ALL BRANCHES vs FORMALITY =========================
# (label, per-video CSV prefix, prediction). The three "erode" branches have
# no English counterpart; the two "resist" branches do (English "some books",
# "lots of books") and are the controls. The quantifier branch was added
# 2026-09-28 because "rhai" + noun is too rare in speech to carry that side
# alone.
BRANCHES = [
    ("Mutation",                "mutations",            "erode"),
    ("Conjugated prepositions", "prep_mutations",       "erode"),
    ("Numeral + singular noun", "numeral_mutations",    "erode"),
    ("rhai + plural noun",      "plural_mutations",     "resist"),
    ("Quantifier + o + plural", "quantifier_mutations", "resist"),
    # Both controls test one rule -- a count noun after a quantity word is
    # plural, in Welsh and in English -- so they're also pooled into one
    # sturdier control (decision 2026-10-02). Built in load_branch_rows.
    ("Controls combined",       None,                   "resist"),
]
COMBINED_CONTROL = ("rhai + plural noun", "Quantifier + o + plural")


def _apply_quantifier_census(rows):
    """quantifier_engine.py writes every singular after a quantifier as
    erosion_unverified -- most are mass or degree readings ("llawer o
    wahaniaeth"), so on its own the branch read 0% by construction. Since
    2026-10-05 every candidate is judged by hand (the quantifier census,
    audit_sample.py, menu 6 -> d): those judged "slip" become erosions here;
    the rest stay out of the rate. Prints how far the census has got."""
    if "status" not in rows.columns:
        return rows
    cand = rows["status"] == "erosion_unverified"
    if not cand.any():
        return rows
    try:
        from audit_sample import census_verdicts, census_id
        verdicts = census_verdicts()
    except Exception as e:
        print(f"  ⚠️ Quantifier census not read ({e}) -- its singulars stay uncounted.")
        return rows
    ids = rows.loc[cand].apply(lambda r: census_id(r["video_url"], r["timestamp"], r["following_word"]),
                               axis=1)
    verdict = ids.map(lambda i: verdicts.get(i, ("",))[0])
    slips = verdict.index[verdict == "slip"]
    rows = rows.copy()
    rows.loc[slips, "status"] = "erosion"
    rows.loc[slips, "is_erosion"] = True
    print(f"  Quantifier census: {int(cand.sum()):,} singular candidates, "
          f"{int((verdict != '').sum()):,} judged -> {len(slips):,} slips counted as erosion; "
          f"{int((verdict == '').sum()):,} not judged yet (menu 6 -> d).")
    return rows


def load_branch_rows():
    """
    {branch label: evaluable rows} across every run under runs/. One run
    per video: if a video was processed more than once, only rows from its
    most recent runs/<stamp>/ folder are kept (stamps sort chronologically),
    so a re-run replaces a video's data instead of being added on top of it.
    Mutation folders use the same one-file-per-folder rule as
    load_and_merge_mutations() (corroborated, else original, else legacy).
    Every branch writes "correct_mutation"/"erosion" statuses, so
    EVALUABLE_STATUSES applies to all of them. Siarad rows keep their
    "speaker" column.
    """
    out = {}
    for label, prefix, _ in BRANCHES:
        if prefix is None:      # the combined control, built below
            continue
        files = [p for p in MUT_DIR.rglob(f"{prefix}_*.csv")
                 if "_deleted" not in p.parts and not p.name.endswith("_precaption_backup.csv")]
        if prefix == "mutations":
            by_dir = {}
            for p in files:
                by_dir.setdefault(p.parent, []).append(p)
            files = []
            for fs in by_dir.values():
                corroborated = [f for f in fs if f.name.startswith("mutations_corroborated_")]
                original     = [f for f in fs if f.name.startswith("mutations_original_")]
                files.extend(corroborated or original or fs)
        frames = []
        for p in run_progress.files(files, f"Loading {label}"):
            try:
                d = pd.read_csv(p, encoding="utf-8-sig")
            except Exception as e:
                run_progress.tqdm.write(f"  ⚠️ Could not read {p.name}: {e}")
                continue
            if d.empty or "video_url" not in d.columns or "is_erosion" not in d.columns:
                continue
            stamp = re.search(r"\d{8}_\d{6}", p.name)
            frames.append(d.assign(_run=stamp.group(0) if stamp else ""))
        if not frames:
            out[label] = pd.DataFrame()
            continue
        rows = pd.concat(frames, ignore_index=True)
        latest = rows.groupby("video_url")["_run"].transform("max")
        rows = rows[rows["_run"] == latest]
        if prefix == "quantifier_mutations":
            rows = _apply_quantifier_census(rows)
        if "status" in rows.columns:
            rows = rows[rows["status"].isin(EVALUABLE_STATUSES)]
        rows = rows.assign(is_erosion=rows["is_erosion"].map(lambda v: str(v).strip().lower() == "true"))
        out[label] = rows
    parts = [out[l] for l in COMBINED_CONTROL if l in out and not out[l].empty]
    out["Controls combined"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


UNVERSIONED = "unversioned (before 2026-09-26)"


def report_pipeline_versions(df, label):
    """Prints a warning when rows come from more than one pipeline_version
    (corpus_io.pipeline_version()): numbers pooled across detection-code
    versions aren't one measurement. Returns the version counts."""
    if df is None or df.empty:
        return pd.Series(dtype=int)
    versions = (df["pipeline_version"].fillna(UNVERSIONED) if "pipeline_version" in df.columns
                else pd.Series(UNVERSIONED, index=df.index))
    # Versions recorded as detection-identical (corpus_io.VERSION_EQUIVALENTS)
    # count as the current one -- they are one measurement.
    current = pipeline_version()
    versions = versions.map(lambda v: current if is_current_version(v) else v)
    counts = versions.value_counts()
    if len(counts) > 1:
        print(f"  ⚠️ {label}: rows from {len(counts)} different pipeline versions -- "
              f"re-process the older videos before reporting these numbers:")
        for version, n in counts.items():
            print(f"      {version}: {n:,} rows")
    return counts


def _unit_rates(rows, keys, min_contexts):
    """Erosion rate + context count per unit (keys: video, or video+speaker)."""
    if rows is None or rows.empty or any(k not in rows.columns for k in keys):
        return pd.DataFrame()
    rates = rows.groupby(keys).agg(erosion_rate=("is_erosion", "mean"),
                                   contexts=("is_erosion", "size")).reset_index()
    return rates[rates["contexts"] >= min_contexts]


def _wilson_interval(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


MIN_FIT_EVENTS = 10   # erosions (and non-erosions) a slope needs -- see _logistic_fit


def _logistic_fit(x, rate, contexts):
    """Binomial logistic regression of erosion on x, every unit weighted by
    its contexts: P(erosion) = 1 / (1 + exp(-(a + b*x))). Replaces the
    straight regplot line (2026-10-02), which ran below 0% at the formal end
    -- a rate can't. Fitted by IRLS in numpy (no statsmodels needed).

    The 95% band is widened by the quasi-binomial dispersion: contexts from
    one video (one speaker, one topic) aren't independent, so a plain
    binomial band would be far too narrow.

    Not fitted below MIN_FIT_EVENTS erosions (2026-10-04): a slope's
    precision comes from the number of erosions, not of contexts, and the
    usual floor is about 10 per predictor (Peduzzi et al. 1996). Siarad's
    numeral branch (4 erosions in 572 contexts) gave OR 10.1 [0.3-323] and
    its combined control (1 erosion) [0.002-3042] -- correct arithmetic,
    no information, and on the forest plot's log axis they squashed the
    real results into dots.

    Returns (predict(x_grid) -> (rate, low, high) in %, odds ratio per 10
    points of x, its 95% CI), or None when there's nothing to fit (fewer
    than MIN_FIT_EVENTS erosions or non-erosions)."""
    x = np.asarray(x, float)
    n = np.asarray(contexts, float)
    k = np.asarray(rate, float) * n
    events = round(float(k.sum()))
    if len(x) < 3 or min(events, round(float(n.sum())) - events) < MIN_FIT_EVENTS or np.ptp(x) == 0:
        return None
    centre = x.mean()
    X = np.column_stack([np.ones_like(x), x - centre])
    beta = np.zeros(2)
    y = k / n
    try:
        for _ in range(50):
            p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-9, 1 - 1e-9)
            w = n * p * (1 - p)
            z = X @ beta + (y - p) / (p * (1 - p))
            new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
            if np.max(np.abs(new - beta)) < 1e-8:
                beta = new
                break
            beta = new
        p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-9, 1 - 1e-9)
        w = n * p * (1 - p)
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    dispersion = max(1.0, float(np.sum((k - n * p) ** 2 / (n * p * (1 - p)))) / max(len(x) - 2, 1))
    cov = cov * dispersion

    def predict(grid):
        G = np.column_stack([np.ones_like(grid), np.asarray(grid, float) - centre])
        eta = G @ beta
        se = np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))
        sig = lambda v: 100 / (1 + np.exp(-v))
        return sig(eta), sig(eta - 1.96 * se), sig(eta + 1.96 * se)

    b, se_b = beta[1], math.sqrt(cov[1, 1])
    odds10 = (math.exp(10 * b), math.exp(10 * (b - 1.96 * se_b)), math.exp(10 * (b + 1.96 * se_b)))
    return predict, odds10


def _draw_logistic(ax, x, rate, contexts, color="#C44E52"):
    """Draws the _logistic_fit curve + band on ax; returns a short label
    ("OR per 10 F-points 0.41 [0.33-0.50]") or "" when nothing was fitted."""
    fit = _logistic_fit(x, rate, contexts)
    if fit is None:
        return ""
    predict, (odds, low, high) = fit
    grid = np.linspace(np.min(x), np.max(x), 200)
    mid, lo, hi = predict(grid)
    ax.plot(grid, mid, color=color, linewidth=2)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.15, linewidth=0)
    return f"OR per 10 points {odds:.2f} [{low:.2f}-{high:.2f}]"


def _rate_axis_from_zero(ax):
    """Erosion-rate y-axis that starts exactly at 0% -- never "-0%" / "-2%"
    ticks. A branch with no erosion at all (quantifier) got matplotlib's
    tiny symmetric range around 0, labelled -2% ... 0% (2026-10-02); it now
    shows 0-5%. Points at 0% sit on the axis line: draw them with
    clip_on=False so they aren't cut in half."""
    ax.set_ylim(0, max(ax.get_ylim()[1], 5))
    # whole-number steps only: with 2.5-point steps the "%.0f%%" labels read
    # 2%, 5%, 8%, 12%... (2026-10-29)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))


def _spearman_text(x, y):
    """Rank correlation (no straight-line assumption), or n/a when flat."""
    s = pd.DataFrame({"x": x, "y": y})
    if s["x"].nunique() < 2 or s["y"].nunique() < 2:
        return "n/a (no variation)"
    # Spearman = Pearson on ranks; done by hand because pandas' own
    # method="spearman" needs scipy, which this laptop doesn't have.
    return f"{s['x'].rank().corr(s['y'].rank()):.2f}"


def fig_branches_vs_formality(branch_rows, formality_df, keys, filename, unit, min_contexts=5):
    """
    Scatter + logistic curve, one panel per branch: erosion rate per unit
    (video, or Siarad speaker) against its F-score. Points are coloured by
    source (_scatter_by_source: one colour per corpus, the same in every
    figure), so an offset between sources shows up as separate clouds, not
    as a slope. A panel with fewer
    than 3 units reaching min_contexts says so instead of fitting a line;
    the sparse branches usually need fig_branches_by_formality_band().
    """
    if formality_df.empty or "f_score" not in formality_df.columns:
        print(f"  [skip] {filename} -- no formality data")
        return
    fscores = formality_df[keys + ["f_score", "source"]].dropna(subset=["f_score"])
    # Two columns, portrait (2026-10-09): a 2x3 grid printed at A4 text
    # width left each panel ~5 cm, too narrow for its title.
    nrows = math.ceil(len(BRANCHES) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(9.5, 4.4 * nrows), sharex=True)
    for ax in axes.flat[len(BRANCHES):]:
        ax.set_visible(False)
    for ax, (label, _prefix, prediction) in zip(axes.flat, BRANCHES):
        pts = _unit_rates(branch_rows.get(label), keys, min_contexts)
        pts = pts.merge(fscores, on=keys, how="inner") if not pts.empty else pts
        title = f"{label} (predicted to {prediction})"
        if len(pts) < 3:
            ax.set_title(title, fontweight="bold")
            ax.text(0.5, 0.5, f"too few {unit}s: {len(pts)} with >= {min_contexts} contexts",
                    ha="center", va="center", transform=ax.transAxes, color="grey")
            continue
        pts["erosion_rate_pct"] = pts["erosion_rate"] * 100
        _scatter_by_source(ax, pts, "f_score", "erosion_rate_pct", pts["contexts"].clip(upper=200))
        # Logistic curve (see _logistic_fit); a branch with no erosion at all
        # (quantifier: 0% everywhere) gets no curve and rho "n/a".
        fit = _draw_logistic(ax, pts["f_score"], pts["erosion_rate"], pts["contexts"])
        rho = _spearman_text(pts["f_score"], pts["erosion_rate_pct"])
        _rate_axis_from_zero(ax)
        ax.set_title(f"{title}\n{unit}s={len(pts)}, Spearman ρ={rho}"
                     + (f"\n{fit}" if fit else f"\nno curve (fewer than {MIN_FIT_EVENTS} erosions)"),
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("F-score (higher = more formal)")
        ax.set_ylabel("Erosion rate (%)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Erosion rate vs. formality, per {unit} (point size = contexts)",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)


def fig_within_corpus_slopes(branch_rows, formality_df, keys, filename, unit, min_contexts=5):
    """
    The formality slope fitted WITHIN each corpus (2026-10-02). A curve
    across all units mixes "which corpus" with "how formal": Siarad (casual,
    northern) has low conjugated-preposition erosion and CorCenCC (a little
    more formal, mostly southern) higher, so the pooled preposition curve
    rose with formality (OR 1.35) while every corpus-internal comparison may
    not; and the mutation curve's formal end is the news texts. Here each
    corpus gets its own logistic fit (_logistic_fit: weighted by contexts,
    quasi-binomial interval) -- the corpus can no longer stand in for
    formality. Forest plot of the odds ratio per 10 F-points (below 1 =
    erosion falls as formality rises), plus a CSV and a printed table.
    """
    if formality_df.empty or "f_score" not in formality_df.columns:
        print(f"  [skip] {filename} -- no formality data")
        return
    fscores = formality_df[keys + ["f_score", "source"]].dropna(subset=["f_score"])
    records = []
    for label, _prefix, prediction in BRANCHES:
        pts = _unit_rates(branch_rows.get(label), keys, min_contexts)
        if pts.empty:
            continue
        pts = pts.merge(fscores, on=keys, how="inner")
        pts["group"] = pts["source"].map(_corpus_group)
        for group in CORPUS_GROUP_ORDER:
            g = pts[pts["group"] == group]
            fit = _logistic_fit(g["f_score"], g["erosion_rate"], g["contexts"]) if len(g) >= 5 else None
            rec = {"branch": label, "prediction": prediction, "corpus": group, f"{unit}s": len(g),
                   "contexts": int(g["contexts"].sum()),
                   "erosions": int(round((g["erosion_rate"] * g["contexts"]).sum())),
                   "f_range": f"{g['f_score'].min():.0f}-{g['f_score'].max():.0f}" if len(g) else "",
                   "odds_ratio_per_10": None, "ci_low": None, "ci_high": None}
            if fit is not None:
                _predict, (odds, low, high) = fit
                rec.update({"odds_ratio_per_10": round(odds, 3), "ci_low": round(low, 3),
                            "ci_high": round(high, 3)})
            records.append(rec)
    table = pd.DataFrame(records)
    if table.empty:
        print(f"  [skip] {filename} -- no data")
        return
    table.to_csv(OUT_DIR / filename.replace(".png", ".csv"), index=False, encoding="utf-8-sig")

    print(f"\nFormality slope WITHIN each corpus, per {unit} (odds ratio per 10 F-points; "
          f"< 1 = erosion falls as formality rises):")
    for label in table["branch"].unique():
        print(f"  {label}")
        for _, r in table[table["branch"] == label].iterrows():
            if r[f"{unit}s"] == 0:
                continue
            slope = (f"not fitted (fewer than {MIN_FIT_EVENTS} erosions, or under 5 {unit}s)"
                     if pd.isna(r["odds_ratio_per_10"])
                     else f"OR {r['odds_ratio_per_10']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}]")
            print(f"    {r['corpus']:<30} {r[f'{unit}s']:>5} {unit}s, {r['erosions']:>6,}/"
                  f"{r['contexts']:<7,} F {r['f_range']:<7}  {slope}")

    fitted = table.dropna(subset=["odds_ratio_per_10"])
    if fitted.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(fitted) + 1.5)))
    y, ticks, labels, labelled = 0, [], [], set()
    for label in fitted["branch"].unique():
        rows = fitted[fitted["branch"] == label]
        for _, r in rows.iterrows():
            color = CORPUS_GROUP_COLORS.get(r["corpus"], "#555555")
            ax.errorbar(r["odds_ratio_per_10"], y,
                        xerr=[[r["odds_ratio_per_10"] - r["ci_low"]], [r["ci_high"] - r["odds_ratio_per_10"]]],
                        fmt="o", color=color, ecolor=color, capsize=3,
                        label=r["corpus"] if r["corpus"] not in labelled else None)
            labelled.add(r["corpus"])
            ticks.append(y)
            labels.append(f"{label} -- {r['corpus']} ({r[f'{unit}s']} {unit}s)")
            y += 1
        y += 0.6
    ax.axvline(1, color="#333333", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio of erosion per 10 F-points, 95% CI (log scale; left of 1 = less erosion "
                  "when more formal)")
    ax.set_title(f"Formality effect within each corpus, per {unit}\n"
                 f"(each corpus fitted on its own, so the corpus can't stand in for formality)",
                 fontweight="bold", pad=12)
    # Under the axis: at lower right it covered the rhai/quantifier rows (2026-10-29).
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    _save(fig, filename)


def fig_branches_by_formality_band(branch_rows, formality_df, keys, filename, unit):
    """
    The hypothesis in one chart: units (videos, or Siarad speakers) split
    into F-score thirds, every branch's contexts pooled within each third,
    erosion rate plotted with a Wilson 95% interval. Pooling is what lets
    the sparse numeral/rhai branches show at all -- a single video rarely
    has 5 of them. Prediction: the "erode" lines rise toward the informal
    end, the "resist" line stays flat.
    """
    if formality_df.empty or "f_score" not in formality_df.columns:
        print(f"  [skip] {filename} -- no formality data")
        return
    units = formality_df[keys + ["f_score"]].dropna(subset=["f_score"]).copy()
    if len(units) < 3:
        print(f"  [skip] {filename} -- need at least 3 {unit}s with an F-score (found {len(units)})")
        return
    # rank first so tied F-scores can't produce duplicate band edges
    units["band"] = pd.qcut(units["f_score"].rank(method="first"), 3, labels=False)
    band_labels = [f"F {g['f_score'].min():.0f}-{g['f_score'].max():.0f}\n"
                   f"({['least formal', 'middle', 'most formal'][b]} third)"
                   for b, g in units.groupby("band")]

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    offsets = [0.09 * (k - (len(BRANCHES) - 1) / 2) for k in range(len(BRANCHES))]
    for offset, (label, _prefix, prediction) in zip(offsets, BRANCHES):
        rows = branch_rows.get(label)
        if rows is None or rows.empty or any(k not in rows.columns for k in keys):
            continue
        banded = rows.merge(units[keys + ["band"]], on=keys, how="inner")
        if banded.empty:
            continue
        xs, ys, lo, hi, ns = [], [], [], [], []
        for band, grp in banded.groupby("band"):
            n, k = len(grp), int(grp["is_erosion"].sum())
            low, high = _wilson_interval(k, n)
            xs.append(band + offset)
            ys.append(100 * k / n)
            # clamp: at k=0 (or k=n) float rounding can leave the bound a
            # hair past the point estimate, and errorbar rejects negatives
            lo.append(max(0.0, 100 * (k / n - low)))
            hi.append(max(0.0, 100 * (high - k / n)))
            ns.append(n)
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4,
                    linestyle="--" if prediction == "resist" else "-",
                    label=f"{label} ({prediction})")
        for x, y, n in zip(xs, ys, ns):
            # slanted and a little smaller: the near-0% points of the formal
            # third sit close together and their labels ran into each other
            ax.annotate(f"n={n:,}", (x, y), textcoords="offset points", xytext=(3, 5),
                        fontsize=6.5, rotation=40)
        plotted += 1
    if not plotted:
        plt.close(fig)
        print(f"  [skip] {filename} -- no branch rows matched a {unit} with an F-score")
        return
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels)
    ax.set_ylabel("Erosion rate (%, pooled, 95% Wilson CI)")
    # headroom so the slanted n= label of the top point doesn't run into the title
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_title(f"Erosion by formality band, per branch ({len(units)} {unit}s split into thirds)",
                 fontweight="bold", pad=12)
    # under the axis: inside, it covered a data point (speaker version, 2026-10-29)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    _save(fig, filename)


# ========================= SUMMARY TABLE =========================
def print_corpus_summary(df, batch_log):
    total     = len(df)
    cs_cases  = int(df["is_code_switch"].sum()) if "is_code_switch" in df.columns else 0
    non_cs    = total - cs_cases
    erosion   = int(df["is_erosion"].sum()) if "is_erosion" in df.columns else 0
    collision = int(df["collision_flag"].notna().sum()) if "collision_flag" in df.columns else 0

    # PATCH: the headline erosion-rate figure below used to divide by
    # non_cs (total - code_switch only), which still includes
    # phantom_mutation / selective_invariancy / erosion_unverified /
    # colloquial_variant rows -- none of those are evaluable erosion
    # contexts, and every OTHER erosion-rate calculation in this file
    # (by mutation type, by channel, both explicitly labeled "evaluable
    # contexts only") already excludes them via EVALUABLE_STATUSES. That
    # meant the single headline number at the top of this summary was
    # computed on a noisier, larger denominator than the breakdowns
    # directly below it -- same status column, two different answers.
    # evaluable_n/evaluable_erosion now match that same filter so the
    # headline is consistent with the rest of the report; total/non_cs
    # are kept as-is for informational corpus-size context only.
    if "status" in df.columns:
        evaluable_df    = df[df["status"].isin(EVALUABLE_STATUSES)]
        evaluable_n      = len(evaluable_df)
        evaluable_erosion = int(evaluable_df["is_erosion"].sum()) if "is_erosion" in evaluable_df.columns else 0
    else:
        evaluable_n, evaluable_erosion = non_cs, erosion

    print("\n" + "="*70)
    print("CORPUS SUMMARY - WELSH MUTATION ANALYSIS")
    print("="*70)
    print(f"Batches loaded                    : {len(batch_log)}")
    for stamp, count in batch_log:
        print(f"  {stamp}  ({count} rows)")

    # PATCH: video-level counts (distinct videos analyzed, total minutes of
    # audio covered) alongside the existing row-level mutation-context count.
    # "video_url" is the uniqueness key rather than "video_title" since two
    # different videos could plausibly share a title, and it's set for every
    # source type (YouTube URL for queue/discovery, local file path for
    # option 1). One row per video is taken via drop_duplicates before
    # summing -- video_duration_seconds is stamped onto every mutation row
    # from that video, so summing across all rows would massively overcount.
    # Rows from analyze_phrase() (menu option 4) carry no video_url at all
    # (there's no video involved) and are excluded here via dropna, though
    # they still count toward "Total mutation contexts" below.
    if "video_url" in df.columns:
        video_level = df.dropna(subset=["video_url"]).drop_duplicates("video_url")
        n_videos = len(video_level)
        has_duration = "video_duration_seconds" in video_level.columns
        if has_duration:
            total_minutes = video_level["video_duration_seconds"].fillna(0).sum() / 60
            print(f"Videos analyzed                    : {n_videos:,}")
            print(f"Total video analyzed                : {total_minutes:,.1f} min "
                  f"({total_minutes/60:.1f} hr)")
        else:
            print(f"Videos analyzed                    : {n_videos:,}")
            print("Total video analyzed                : unavailable "
                  "(older CSVs predate duration tracking)")

        # PATCH (Phase 3): the "by register" breakdown that used to sit
        # here is gone -- channel_register was a hand-assigned, per-
        # channel label. See corpus_formality.py for the grounded, per-
        # video replacement.

    # PATCH: unfiltered row counts (every status -- correct, erosion,
    # phantom, code-switch, mismatch, everything) per channel and per
    # video. Every other count in this function (evaluable_n, the
    # erosion-rate breakdowns below, every figure in the corresponding
    # PNGs) is scoped to EVALUABLE_STATUSES or narrower -- there was
    # previously no single place that answered the plain question
    # "how many rows exist for X, full stop", which meant answering it
    # required a one-off manual query against merged_mutations.csv.
    if "channel" in df.columns:
        print("\nRows per channel (UNFILTERED -- every status, incl. "
              "phantom/code-switch/mismatch/etc.):")
        for ch, n in df["channel"].value_counts().items():
            print(f"  {ch:<24} {n:>6,}")

    if "video_title" in df.columns:
        print("\nRows per video (UNFILTERED -- every status):")
        video_row_counts = df.groupby("video_title", sort=False).size() \
            .sort_values(ascending=False)
        for title, n in video_row_counts.items():
            short_title = (title[:58] + "...") if isinstance(title, str) and len(title) > 61 else title
            print(f"  {short_title:<61} {n:>6,}")

    print(f"\nTotal mutation contexts           : {total:,}")
    print(f"Code-switch cases                  : {cs_cases:,} ({cs_cases/total:.1%})"
          if total > 0 else f"Code-switch cases                  : {cs_cases:,}")
    print(f"Non-CS contexts (analysable)       : {non_cs:,}")
    print(f"  of which evaluable (excl. phantom/selective-invariancy/"
          f"unverified/colloquial-variant): {evaluable_n:,}")
    print(f"Erosion cases                      : {evaluable_erosion:,} "
          f"({evaluable_erosion/evaluable_n:.1%} of evaluable contexts)"
          if evaluable_n > 0 else f"Erosion cases                      : {evaluable_erosion:,}")
    print(f"Homograph collision flags          : {collision:,}")

    if "expected_mutation" in df.columns and "is_erosion" in df.columns:
        print("\nErosion rate by mutation type (evaluable contexts only):")
        sub = df[(df["is_code_switch"].isin([False])) & (df["trigger_word"] != "[OMITTED]")] \
            if "is_code_switch" in df.columns else df
        if "status" in sub.columns:
            sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]
        for mut_type, group in sub.groupby("expected_mutation"):
            n = len(group)
            if n < 5:
                continue
            rate = group["is_erosion"].mean()
            print(f"  {mut_type:<20} n={n:<6} erosion={rate:.1%}")

    if "channel" in df.columns and "is_erosion" in df.columns:
        print("\nErosion rate by channel (evaluable contexts only):")
        sub = df[df["is_code_switch"].isin([False])] if "is_code_switch" in df.columns else df
        if "status" in sub.columns:
            sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]
        for ch, group in sub.groupby("channel"):
            n    = len(group)
            rate = group["is_erosion"].mean()
            print(f"  {ch:<20} n={n:<6} erosion={rate:.1%}")

    print("="*70)


# ========================= UTTERANCE EXPORT =========================
def export_utterance_level(df):
    """
    Writes a clean, flat CSV of all mutation contexts, preserving the
    fields most useful for joining with future syntactic/phonetic datasets.
    """
    export_cols = [
        "video_url", "video_title", "source", "channel", "batch",
        "timestamp", "trigger_word", "following_word", "lemma",
        "rule", "expected_mutation", "mutation_found",
        "tagger_mutation_type", "tagger_agreement",
        "status", "is_erosion", "is_code_switch",
        "collision_flag", "register_class",
        "trigger_confidence", "following_confidence", "pos",
    ]
    present   = [c for c in export_cols if c in df.columns]
    export_df = df[present].copy()
    out_path  = OUT_DIR / "utterance_export.csv"
    export_df.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"\nUtterance-level export: {out_path.name} ({len(export_df):,} rows)")
    print("  Use this file for future joins with syntactic/phonetic datasets.")


# ========================= MAIN =========================
# ========================= SPEAKER BACKGROUND (supplementary) =========================
# Added 2026-10-09, after Margaret Deuchar pointed to the BangorTalk
# questionnaire data (bangortalk.org.uk/downloads.php). A supplementary
# analysis: it joins the speakers' questionnaire answers to the FROZEN
# mutation results (nothing is re-detected), for Siarad and Patagonia.
#
# Input: BASE_DIR/speaker_background/speakers_<corpus>.csv -- cleaned copies
# of the questionnaire sheets with only the coded background columns (the
# Patagonia sheet's names/addresses/emails were never copied). Codes, from
# the Siarad questionnaire (Q6) and the data itself:
#   welsh_since   1 by age 2, 2 by age 4, 3 primary school, 4 secondary, 5 adult
#   mother_spoke  1 Welsh, 2 the other language (English / Spanish), 0 both,
#                 3 other -- inferred: every adult learner has 2, and the
#                 Patagonia "first_lang" column uses the same 1/2/0.
SPEAKER_BG_DIR = BASE_DIR / "speaker_background"
BG_CORPORA = ("siarad", "patagonia")
BG_MIN_CONTEXTS = 20          # a speaker's own rate needs this many contexts
BG_BOOT = 2000
AGE_BANDS = [(0, 30, "<30"), (30, 50, "30-49"), (50, 70, "50-69"), (70, 200, "70+")]


def _load_speaker_background():
    """One row per (video_url, speaker) with the coded answers."""
    frames = []
    for corpus in BG_CORPORA:
        path = SPEAKER_BG_DIR / f"speakers_{corpus}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"files": str}, encoding="utf-8-sig"))
    if not frames:
        return pd.DataFrame()
    bg = pd.concat(frames, ignore_index=True)
    bg = bg.dropna(subset=["files"]).assign(file=lambda d: d["files"].str.split())
    bg = bg.explode("file")
    bg["video_url"] = bg["corpus"].str.lower() + ":" + bg["file"].str.lower()
    bg["speaker"] = bg["speaker"].astype(str).str.upper()
    for c in ("age", "welsh_since", "mother_spoke"):
        bg[c] = pd.to_numeric(bg[c], errors="coerce")
    bg["acquisition"] = np.where(bg["welsh_since"].isna(), None,
                                 np.where(bg["welsh_since"] <= 2, "Welsh by age 4",
                                          "Welsh later (school / adult)"))
    home = {1: "Welsh", 0: "both", 2: "English / Spanish"}
    bg["mother_language"] = bg["mother_spoke"].map(home)
    bg["age_band"] = None
    for lo, hi, name in AGE_BANDS:
        bg.loc[(bg["age"] >= lo) & (bg["age"] < hi), "age_band"] = name
    return bg.drop(columns=["files"]).drop_duplicates(["video_url", "speaker"])


def _boot_group_rate(sp, rng):
    """Pooled erosion rate of a group of speakers and its 95% interval from
    resampling SPEAKERS (contexts of one speaker aren't independent)."""
    e, n = sp["erosions"].to_numpy(float), sp["contexts"].to_numpy(float)
    if len(sp) < 2 or n.sum() == 0:
        return e.sum() / n.sum() if n.sum() else float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(sp), size=(BG_BOOT, len(sp)))
    rates = e[idx].sum(axis=1) / n[idx].sum(axis=1)
    return e.sum() / n.sum(), *np.percentile(rates, [2.5, 97.5])


def _logit_irls(X, k, n):
    """Binomial logistic regression on aggregated counts (k of n), by IRLS.
    Returns the coefficient vector, or None if it doesn't converge."""
    beta = np.zeros(X.shape[1])
    y = k / n
    try:
        for _ in range(60):
            p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-9, 1 - 1e-9)
            w = n * p * (1 - p)
            z = X @ beta + (y - p) / (p * (1 - p))
            new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
            if np.max(np.abs(new - beta)) < 1e-8:
                return new if np.all(np.isfinite(new)) else None
            beta = new
    except np.linalg.LinAlgError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


def speaker_background_analysis(branch_rows, speaker_formality):
    """Mutation erosion by the speakers' language background (Siarad,
    Patagonia): groups by when Welsh was acquired, by age band (apparent
    time) and by the mother's language, each with a speaker-bootstrap 95%
    interval; and per corpus one logistic model, erosion ~ later acquirer +
    age (per 10 years) + formality (per 10 F-points), its odds ratios with
    speaker-bootstrap intervals. Writes speaker_background_*.csv and
    figures/speaker_background.png."""
    bg = _load_speaker_background()
    rows = branch_rows.get("Mutation")
    if bg.empty or rows is None or rows.empty or "speaker" not in rows.columns:
        print("  [skip] speaker background -- no questionnaire data or no speaker column")
        return
    rows = rows[rows["source"].isin(BG_CORPORA)].copy()
    rows["video_url"] = rows["video_url"].str.lower()
    rows["speaker"] = rows["speaker"].astype(str).str.upper()
    per = (rows.groupby(["source", "video_url", "speaker"])["is_erosion"]
               .agg(erosions="sum", contexts="size").reset_index())
    sp = per.merge(bg, on=["video_url", "speaker"], how="inner")
    if speaker_formality is not None and not speaker_formality.empty:
        f = speaker_formality[["video_url", "speaker", "f_score"]].copy()
        f["video_url"] = f["video_url"].str.lower()
        f["speaker"] = f["speaker"].astype(str).str.upper()
        sp = sp.merge(f, on=["video_url", "speaker"], how="left")
    else:
        sp["f_score"] = np.nan
    covered = per["contexts"].sum()
    print(f"\nSpeaker background (supplementary): {len(sp)} speakers matched, "
          f"{sp['contexts'].sum():,} of {covered:,} Siarad/Patagonia mutation contexts "
          f"({sp['contexts'].sum() / covered:.0%})")
    sp.drop(columns=["file"], errors="ignore").to_csv(
        OUT_DIR / "speaker_background_speakers.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20261009)
    groups, panels = [], {"acquisition": [], "age_band": [], "mother_language": []}
    orders = {"acquisition": ["Welsh by age 4", "Welsh later (school / adult)"],
              "age_band": [b for _, _, b in AGE_BANDS],
              "mother_language": ["Welsh", "both", "English / Spanish"]}
    for var, order in orders.items():
        for corpus in BG_CORPORA:
            for level in order:
                g = sp[(sp["source"] == corpus) & (sp[var] == level)]
                if g.empty:
                    continue
                rate, lo, hi = _boot_group_rate(g, rng)
                groups.append({"variable": var, "corpus": CORPUS_LABELS[corpus], "group": level,
                               "speakers": len(g), "contexts": int(g["contexts"].sum()),
                               "erosions": int(g["erosions"].sum()), "erosion_rate": round(rate, 4),
                               "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    gdf = pd.DataFrame(groups)
    gdf.to_csv(OUT_DIR / "speaker_background_groups.csv", index=False, encoding="utf-8-sig")
    for var in orders:
        print(f"  by {var}:")
        for _, r in gdf[gdf["variable"] == var].iterrows():
            print(f"    {r['corpus']:<10} {r['group']:<30} {r['erosion_rate']:6.1%} "
                  f"[{r['ci_low']:.1%}-{r['ci_high']:.1%}]  {r['speakers']:>3} speakers, "
                  f"{r['contexts']:,} contexts")

    # Model per corpus: all three predictors together, so each is net of the others.
    models = []
    for corpus in BG_CORPORA:
        m = sp[(sp["source"] == corpus)].dropna(subset=["welsh_since", "age", "f_score"])
        m = m[m["contexts"] >= 1]
        if len(m) < 20 or m["erosions"].sum() < MIN_FIT_EVENTS:
            continue
        def design(d):
            return np.column_stack([np.ones(len(d)), (d["welsh_since"] >= 3).astype(float),
                                    (d["age"] - 50) / 10, (d["f_score"] - 60) / 10])
        beta = _logit_irls(design(m), m["erosions"].to_numpy(float), m["contexts"].to_numpy(float))
        if beta is None:
            continue
        boots = []
        for _ in range(1000):
            b = m.iloc[rng.integers(0, len(m), len(m))]
            bb = _logit_irls(design(b), b["erosions"].to_numpy(float), b["contexts"].to_numpy(float))
            if bb is not None:
                boots.append(bb)
        boots = np.array(boots)
        for j, name in enumerate(["later acquirer (vs by age 4)", "age (per 10 years)",
                                  "formality (per 10 F-points)"], start=1):
            lo, hi = np.percentile(np.exp(boots[:, j]), [2.5, 97.5]) if len(boots) else (np.nan, np.nan)
            models.append({"corpus": CORPUS_LABELS[corpus], "predictor": name, "speakers": len(m),
                           "odds_ratio": round(float(np.exp(beta[j])), 3),
                           "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3)})
    mdf = pd.DataFrame(models)
    mdf.to_csv(OUT_DIR / "speaker_background_model.csv", index=False, encoding="utf-8-sig")
    if not mdf.empty:
        print("  model (erosion ~ later acquirer + age + formality; speaker-bootstrap 95% CI):")
        for _, r in mdf.iterrows():
            print(f"    {r['corpus']:<10} {r['predictor']:<30} OR {r['odds_ratio']:.2f} "
                  f"[{r['ci_low']:.2f}-{r['ci_high']:.2f}]  ({r['speakers']} speakers)")

    # Figure: three panels, one per background variable.
    titles = {"acquisition": "When Welsh was acquired", "age_band": "Speaker age (apparent time)",
              "mother_language": "Language the mother spoke"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    width = 0.38
    for ax, (var, order) in zip(axes, orders.items()):
        sub = gdf[gdf["variable"] == var]
        for ci, corpus in enumerate(BG_CORPORA):
            label = CORPUS_LABELS[corpus]
            color = CORPUS_GROUP_COLORS.get(label, "#888888")
            for li, level in enumerate(order):
                r = sub[(sub["corpus"] == label) & (sub["group"] == level)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = li + (ci - 0.5) * width
                ax.bar(x, 100 * r["erosion_rate"], width=width * 0.92, color=color,
                       label=label if li == 0 or (var != "acquisition" and level == order[0]) else None)
                ax.errorbar(x, 100 * r["erosion_rate"],
                            yerr=[[100 * (r["erosion_rate"] - r["ci_low"])],
                                  [100 * (r["ci_high"] - r["erosion_rate"])]],
                            fmt="none", ecolor="#444444", capsize=2, linewidth=0.8)
                ax.text(x, 100 * r["ci_high"] + 1, str(int(r["speakers"])), ha="center",
                        va="bottom", fontsize=7, color="#555555")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace(" (", "\n(").replace(" / ", " /\n") for o in order], fontsize=9)
        ax.set_title(titles[var], fontweight="bold")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Erosion rate (95% CI)")
    axes[0].set_ylim(0, max(axes[0].get_ylim()[1], 100 * gdf["ci_high"].max() + 9))
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes[0].legend(seen.values(), seen.keys(), loc="upper left", fontsize=9)
    fig.suptitle("Mutation erosion by speaker background\n"
                 "(numbers above bars = speakers)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "speaker_background.png")


# ---- Quick redraw from the last full run (2026-10-10) ----
# A full run reads ~5,000 output folders and recomputes every formality
# score (about 15 minutes). It saves what the figures are drawn from to
# FIG_CACHE; "redraw" mode loads that and draws only the figures you pick,
# with the CURRENT figure code. The cache is only as new as the last full
# run -- after new data (or menu 4 / 6 -> d), do a full run.
FIG_CACHE = OUT_DIR / "_figure_cache.pkl"


def _figure_registry(c):
    """{figure name: function drawing it} over a cache dict c."""
    df, fdf, br, sdf = c["df"], c["formality_df"], c["branch_rows"], c["speaker_df"]
    sp = sdf[sdf["pos_tagged_word_count"] >= corpus_formality.MIN_SPEAKER_WORDS] \
        if sdf is not None and not sdf.empty else sdf
    has_f = fdf is not None and not fdf.empty
    has_s = sp is not None and not sp.empty
    reg = {
        "erosion_by_type": lambda: fig_erosion_by_type(df),
        "erosion_by_mutation_family": lambda: fig_erosion_by_mutation_family(df),
        "erosion_by_rule": lambda: fig_erosion_by_rule(df),
        "erosion_by_channel": lambda: fig_erosion_by_channel(df),
        "codeswitch_by_channel": lambda: fig_codeswitch_by_channel(df),
        "status_distribution": lambda: fig_status_distribution(df),
        "tagger_agreement": lambda: fig_tagger_agreement(df),
    }
    if has_f:
        reg.update({
            "erosion_vs_formality": lambda: fig_erosion_vs_formality(df, fdf),
            "erosion_vs_codeswitch": lambda: fig_erosion_vs_codeswitch(df, fdf),
            "branches_vs_formality": lambda: fig_branches_vs_formality(
                br, fdf, ["video_url"], "branches_vs_formality.png", "video"),
            "branches_by_formality_band": lambda: fig_branches_by_formality_band(
                br, fdf, ["video_url"], "branches_by_formality_band.png", "video"),
            "within_corpus_slopes": lambda: fig_within_corpus_slopes(
                br, fdf, ["video_url"], "within_corpus_slopes.png", "video"),
        })
    if has_s:
        reg.update({
            "speaker_branches_vs_formality": lambda: fig_branches_vs_formality(
                br, sp, ["video_url", "speaker"], "speaker_branches_vs_formality.png", "speaker"),
            "speaker_branches_by_formality_band": lambda: fig_branches_by_formality_band(
                br, sp, ["video_url", "speaker"], "speaker_branches_by_formality_band.png", "speaker"),
            "speaker_within_corpus_slopes": lambda: fig_within_corpus_slopes(
                br, sp, ["video_url", "speaker"], "speaker_within_corpus_slopes.png", "speaker"),
        })
    reg["speaker_background"] = lambda: speaker_background_analysis(br, sdf)
    return reg


def _save_figure_cache(df, formality_df, branch_rows, speaker_df):
    try:
        pd.to_pickle({"df": df, "formality_df": formality_df, "branch_rows": branch_rows,
                      "speaker_df": speaker_df, "saved": datetime.now().isoformat(timespec="minutes")},
                     FIG_CACHE)
        print(f"Figure cache saved ({FIG_CACHE.name}) -- menu 5 can now redraw single figures quickly.")
    except Exception as e:
        print(f"  ⚠️ Figure cache not saved ({e}) -- quick redraw will need a full run first.")


def _quick_redraw():
    """Asks which figures to redraw and draws them from FIG_CACHE.
    Returns False when there is no usable cache (caller does a full run)."""
    try:
        c = pd.read_pickle(FIG_CACHE)
    except Exception as e:
        print(f"  No usable figure cache ({e}) -- doing a full run.")
        return False
    reg = _figure_registry(c)
    names = list(reg)
    print(f"\nRedraw from the run of {c.get('saved', '?')}:")
    for i, n in enumerate(names, 1):
        print(f"  {i:>2}  {n}")
    raw = input("Figures to redraw (e.g. 2,5-7), 'all', or q to cancel: ").strip().lower()
    if raw in ("q", ""):
        print("Nothing redrawn.")
        return True
    picked = set()
    if raw == "all":
        picked = set(range(1, len(names) + 1))
    else:
        for part in raw.replace(" ", "").split(","):
            try:
                if "-" in part:
                    a, b = part.split("-", 1)
                    picked.update(range(int(a), int(b) + 1))
                elif part:
                    picked.add(int(part))
            except ValueError:
                print(f"  couldn't read {part!r}, skipped")
    for i in sorted(picked):
        if 1 <= i <= len(names):
            print(f"\n{names[i - 1]}:")
            try:
                reg[names[i - 1]]()
            except Exception as e:
                print(f"  ⚠️ {names[i - 1]} failed: {e}")
    print(f"\nFigures in: {FIG_DIR}  (Korean copies in {KO_FIG_DIR})")
    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Welsh Mutation Corpus Analyzer")
    if FIG_CACHE.exists():
        mode = input("Full run (f), or redraw chosen figures from the last full run (r)? [f]: ") \
            .strip().lower() or "f"
        if mode == "r" and _quick_redraw():
            return
    print(f"Reading from: {MUT_DIR}\n")

    df, batch_log = load_and_merge_mutations()
    report_pipeline_versions(df, "Mutation corpus")
    validate_columns(df)
    df = prepare(df)

    merged_path = OUT_DIR / "merged_mutations.csv"
    df.to_csv(merged_path, index=False, encoding="utf-8-sig", quoting=1)
    print(f"Merged CSV saved: {merged_path.name}\n")

    print_corpus_summary(df, batch_log)

    print("\nComputing ASR divergence (original vs. corroborated)...")
    pairs         = load_original_corroborated_pairs()
    pairs         = [(vd, prepare(do), prepare(dc)) for vd, do, dc in pairs]
    divergence_df = compute_asr_divergence(pairs)
    if not divergence_df.empty:
        divergence_path = OUT_DIR / "asr_divergence.csv"
        divergence_df.to_csv(divergence_path, index=False, encoding="utf-8-sig",
                              quoting=csv.QUOTE_MINIMAL)
        print(f"ASR divergence export saved: {divergence_path.name} "
              f"({len(divergence_df):,} rows)")
    print_asr_divergence_summary(divergence_df)

    print("\nGenerating figures...")
    # PATCH (Phase 3): the 3x-per-register calls that used to sit here
    # (formal/informal/casual) are gone along with channel_register
    # itself -- each figure is now generated once, unfiltered. The
    # formal-vs-informal comparison this project is built around now runs
    # against the grounded, per-video formality score instead (see
    # corpus_formality.py and its erosion-vs-formality figure, called
    # below).
    fig_erosion_by_type(df)
    fig_erosion_by_mutation_family(df)
    fig_erosion_by_rule(df)
    fig_erosion_by_channel(df)
    fig_codeswitch_by_channel(df)
    fig_status_distribution(df)
    fig_tagger_agreement(df)

    print("\nComputing per-video formality scores (corpus_formality.py)...")
    formality_df = corpus_formality.build_video_formality_table()
    if not formality_df.empty:
        formality_path = OUT_DIR / "video_formality.csv"
        formality_df.to_csv(formality_path, index=False, encoding="utf-8-sig")
        print(f"Formality scores saved: {formality_path.name} ({len(formality_df):,} videos)")
        fig_erosion_vs_formality(df, formality_df)
        fig_erosion_vs_codeswitch(df, formality_df)
    else:
        print("  [skip] no formality data computed -- see corpus_formality.py output above")

    print("\nAll branches vs. formality...")
    branch_rows = load_branch_rows()
    for label, rows in branch_rows.items():
        n_video = rows["video_url"].nunique() if not rows.empty else 0
        rate = f"{rows['is_erosion'].mean():.1%}" if not rows.empty else "n/a"
        print(f"  {label:<26} {len(rows):>6,} evaluable contexts, {n_video:>4} videos, erosion {rate}")
        # the pooled line mixes corpora with very different speakers
        # (Patagonia = mostly learners/heritage speakers), so report each too
        if not rows.empty and "source" in rows.columns:
            kind = rows["source"].map(lambda s: CORPUS_LABELS.get(str(s), "YouTube (Whisper)"))
            for name, grp in rows.groupby(kind):
                k, n = int(grp["is_erosion"].sum()), len(grp)
                low, high = _wilson_interval(k, n)
                print(f"      {name:<26} {n:>6,} contexts, {grp['video_url'].nunique():>4} videos, "
                      f"erosion {k / n:.1%} (95% CI {max(0.0, low):.1%}-{high:.1%})")
        if label != "Controls combined":
            report_pipeline_versions(rows, label)
    if not formality_df.empty:
        fig_branches_vs_formality(branch_rows, formality_df, ["video_url"],
                                  "branches_vs_formality.png", "video")
        fig_branches_by_formality_band(branch_rows, formality_df, ["video_url"],
                                       "branches_by_formality_band.png", "video")
        fig_within_corpus_slopes(branch_rows, formality_df, ["video_url"],
                                 "within_corpus_slopes.png", "video")

    # Siarad only: the same two figures with each SPEAKER as the unit.
    speaker_df = corpus_formality.build_speaker_formality_table()
    if not speaker_df.empty:
        speaker_path = OUT_DIR / "speaker_formality.csv"
        speaker_df.to_csv(speaker_path, index=False, encoding="utf-8-sig")
        speakers = speaker_df[speaker_df["pos_tagged_word_count"] >= corpus_formality.MIN_SPEAKER_WORDS]
        print(f"Speaker formality saved: {speaker_path.name} ({len(speaker_df):,} speakers, "
              f"{len(speakers):,} with >= {corpus_formality.MIN_SPEAKER_WORDS} tagged words)")
        fig_branches_vs_formality(branch_rows, speakers, ["video_url", "speaker"],
                                  "speaker_branches_vs_formality.png", "speaker")
        fig_branches_by_formality_band(branch_rows, speakers, ["video_url", "speaker"],
                                       "speaker_branches_by_formality_band.png", "speaker")
        fig_within_corpus_slopes(branch_rows, speakers, ["video_url", "speaker"],
                                 "speaker_within_corpus_slopes.png", "speaker")

    # Supplementary: speakers' language background (BangorTalk questionnaires).
    try:
        speaker_background_analysis(branch_rows, speaker_df)
    except Exception as e:
        print(f"  ⚠️ Speaker background analysis failed ({e}) -- everything else is saved.")

    _save_figure_cache(df, formality_df, branch_rows, speaker_df)
    export_utterance_level(df)

    print(f"\nAll outputs written to: {OUT_DIR}")
    print(f"Figures in:             {FIG_DIR}")


if __name__ == "__main__":
    main()