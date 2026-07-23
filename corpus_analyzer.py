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
    Register axis is three-tier, most to least formal: formal (e.g. BBC
    Radio Cymru) > informal (e.g. Hansh/S4C, produced-but-informal content)
    > casual (fully spontaneous, unscripted peer conversation, e.g.
    podcasts like Haclediad). One erosion_by_type/erosion_by_rule/
    erosion_over_batches figure set is produced per tier:
    - figures/formal_erosion_by_type.png   : erosion rate per mutation type (formal register only)
    - figures/informal_erosion_by_type.png : erosion rate per mutation type (informal register only)
    - figures/casual_erosion_by_type.png   : erosion rate per mutation type (casual register only)
    - figures/formal_erosion_by_rule.png   : erosion rate per detection rule (formal register only)
    - figures/informal_erosion_by_rule.png : erosion rate per detection rule (informal register only)
    - figures/casual_erosion_by_rule.png   : erosion rate per detection rule (casual register only)
    - figures/erosion_by_channel.png: erosion rate per source channel
    - figures/codeswitch_by_channel.png: code-switch rate per channel
    - figures/status_distribution.png  : full status breakdown (stacked bar)
    - figures/tagger_agreement.png  : heuristic vs tagger agreement
    - figures/formal_erosion_over_batches.png   : erosion rate trend across runs (formal register only)
    - figures/informal_erosion_over_batches.png : erosion rate trend across runs (informal register only)
    - figures/casual_erosion_over_batches.png   : erosion rate trend across runs (casual register only)
    - utterance_export.csv          : flat utterance-level rows for future joining
"""

import re
import sys
import csv
import shutil
from pathlib import Path
from datetime import datetime

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
from mutation_engine import BASE_DIR, MUT_DIR, OUT_DIR, FIG_DIR

# Seaborn theme -- clean, publication-friendly.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

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
]

CHANNEL_MAP       = {url: name  for url, _slug, name in _CHANNEL_ENTRIES}
CHANNEL_MAP_SHORT = {slug: name for _url, slug, name in _CHANNEL_ENTRIES}


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

        m = re.search(r"mutations_(\d{8}_\d{6})", f.name)
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
    # subfolder (mutations/<stamp>_<slug>/mutations_<stamp>_<slug>.csv)
    # rather than flat in MUT_DIR, so this needs to recurse. "_deleted" is
    # excluded since files moved there by interactive_batch_selection()
    # should stay excluded from future runs, not silently reappear.
    #
    # PATCH: also exclude "*_precaption_backup.csv". fetch_captions.py's
    # run_corroboration() corroborates a video's mutations file IN PLACE
    # (adds the caption_corroboration columns directly onto the real
    # mutations_<stamp>_<slug>.csv) and, before doing that, writes a
    # one-time backup of the pre-corroboration state to
    # mutations_<stamp>_<slug>_precaption_backup.csv -- so re-running
    # corroboration later can't double-corroborate an already-corroborated
    # file. That backup's name still starts with "mutations_" and ends in
    # ".csv", so it matched this glob too: every corroborated video was
    # being loaded twice and every mutation row double-counted in the
    # merged corpus. The backup is intentional and should stay on disk as
    # a safety net; it just shouldn't be treated as a second video's worth
    # of data.
    csv_files = sorted(
        p for p in MUT_DIR.rglob("mutations_*.csv")
        if "_deleted" not in p.parts and not p.name.endswith("_precaption_backup.csv")
    )
    if not csv_files:
        print(f"No mutations_*.csv files found in {MUT_DIR}")
        print("Run the pipeline first to generate mutation data.")
        sys.exit(1)

    # ---- Interactive batch selection / deletion ----
    csv_files = interactive_batch_selection(csv_files)

    if not csv_files:
        print("No batches remaining after deletion. Exiting.")
        sys.exit(0)

    print(f"Loading {len(csv_files)} batch(es):")
    frames    = []
    batch_log = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            m  = re.search(r"mutations_(\d{8}_\d{6})(?:_(.+))?\.csv$", f.name)
            batch_stamp = m.group(1) if m else f.stem
            video_slug  = m.group(2) if (m and m.group(2)) else ""
            # label column: "20260625_193337 · Hansh_clip_01" for per-video files,
            # plain stamp for legacy per-run files
            batch_label = f"{batch_stamp} · {video_slug}" if video_slug else batch_stamp
            df["batch"]       = batch_stamp   # timestamp only -- used for convergence chart x-axis
            df["video_file"]  = video_slug    # slug -- available for future per-video filtering
            frames.append(df)
            batch_log.append((batch_label, len(df)))
            print(f"  {f.name}: {len(df)} rows")
        except Exception as e:
            print(f"  ⚠️ Could not read {f.name}: {e}")

    if not frames:
        print("No readable CSV files found.")
        sys.exit(1)

    merged      = pd.concat(frames, ignore_index=True)
    initial_len = len(merged)

    dedup_keys   = ["video_url", "timestamp", "trigger_word", "following_word"]
    present_keys = [k for k in dedup_keys if k in merged.columns]
    if present_keys:
        merged  = merged.drop_duplicates(subset=present_keys, keep="first")
        dropped = initial_len - len(merged)
        if dropped:
            print(f"  Removed {dropped} duplicate rows across runs.")

    print(f"\nMerged total: {len(merged)} unique mutation contexts "
          f"from {len(frames)} batch(es).\n")
    return merged, batch_log


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
        df["channel"] = df["source"].map(
            lambda s: next((v for k, v in CHANNEL_MAP.items() if k in str(s)), str(s))
        )

    return df


# ========================= FIGURE HELPERS =========================
def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


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
def fig_erosion_by_type(df, register=None, filename_prefix=""):
    """
    Horizontal bar chart: erosion rate (%) per expected mutation type.
    Excludes code-switch rows and phantom rows (no expected mutation type).
    Colour-coded per mutation type for consistency with other figures.

    PATCH: takes an optional `register` filter ("formal" / "informal" /
    "casual"), same reasoning as fig_erosion_by_rule -- formal (BBC Radio
    Cymru), informal (Hansh / Rownd a Rownd / S4C), and casual (fully
    spontaneous unscripted speech, e.g. podcasts) have different erosion
    baselines, so this is called once per register from main().
    """
    if "expected_mutation" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_by_type -- missing columns")
        return

    sub = df
    if register is not None:
        if "channel_register" not in df.columns:
            print(f"  [skip] {filename_prefix}erosion_by_type -- missing channel_register column")
            return
        sub = sub[sub["channel_register"] == register]

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

    title_suffix = f" -- {register.capitalize()} register" if register else ""
    fig, ax = plt.subplots(figsize=(8, max(3, len(grouped) * 0.6)))
    colors = [MUTATION_COLORS.get(m, MUTATION_COLORS["other"])
              for m in grouped["expected_mutation"]]
    _bar_with_counts(ax, grouped["expected_mutation"],
                     grouped["erosion_rate"] * 100,
                     grouped["contexts"], color=colors)
    ax.set_title(f"Mutation Erosion Rate by Expected Mutation Type{title_suffix}",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Erosion Rate (%)")
    ax.set_ylabel("Expected Mutation Type")
    fig.tight_layout()
    _save(fig, f"{filename_prefix}erosion_by_type.png")


def fig_erosion_by_rule(df, register=None, filename_prefix=""):
    """
    Horizontal bar chart: erosion rate per detection rule
    (word_trigger, definite_article+fem_noun, fem_noun+adjective,
    numeral+fem_noun, h_mutation, feminine_ei, phantom_check).
    This is the figure that most directly shows which grammatical
    environments are most at risk -- key for the assimilation argument.

    PATCH: takes an optional `register` filter ("formal" / "informal" /
    "casual"). Formal (BBC Radio Cymru, scripted/edited), informal (Hansh,
    Rownd a Rownd, S4C -- produced but colloquial), and casual (fully
    spontaneous unscripted peer speech, e.g. podcasts) have structurally
    different erosion baselines, so pooling them into one chart blurs
    exactly the register contrast the research is trying to measure.
    Called once per register from main(), instead of once on the mixed pool.
    """
    if "rule" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_by_rule -- missing columns")
        return

    sub = df
    if register is not None:
        if "channel_register" not in df.columns:
            print(f"  [skip] {filename_prefix}erosion_by_rule -- missing channel_register column")
            return
        sub = sub[sub["channel_register"] == register]

    sub = sub[sub["is_code_switch"].isin([False])] if "is_code_switch" in sub.columns else sub
    sub = sub[sub["rule"] != "phantom_check"]
    # Restrict to evaluable contexts only
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]

    grouped = sub.groupby("rule").agg(
        contexts=("is_erosion", "count"),
        erosion_rate=("is_erosion", "mean"),
    ).reset_index()
    grouped = grouped[grouped["contexts"] >= 5].sort_values("erosion_rate", ascending=True)

    if grouped.empty:
        print(f"  [skip] {filename_prefix}erosion_by_rule -- insufficient data")
        return

    title_suffix = f" -- {register.capitalize()} register" if register else ""
    fig, ax = plt.subplots(figsize=(9, max(3, len(grouped) * 0.65)))
    _bar_with_counts(ax, grouped["rule"], grouped["erosion_rate"] * 100,
                     grouped["contexts"], color="#4C72B0")
    ax.set_title(f"Mutation Erosion Rate by Detection Rule{title_suffix}\n"
                 "(grammatical environment)", fontweight="bold", pad=12)
    ax.set_xlabel("Erosion Rate (%)")
    ax.set_ylabel("Detection Rule")
    fig.tight_layout()
    _save(fig, f"{filename_prefix}erosion_by_rule.png")


def fig_erosion_by_channel(df):
    """
    Grouped bar chart: erosion rate per channel, broken down by mutation type.
    The channel axis is the proxy for register/formality -- S4C formal
    broadcast vs Hansh youth-oriented content vs Sgorio sports commentary.
    """
    if "channel" not in df.columns or "is_erosion" not in df.columns \
            or "expected_mutation" not in df.columns:
        print("  [skip] erosion_by_channel -- missing columns")
        return

    sub = df[(df["is_code_switch"].isin([False])) & (df["trigger_word"] != "[OMITTED]")] \
        if "is_code_switch" in df.columns else df
    # Restrict to evaluable contexts only
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]

    type_counts = sub.groupby("expected_mutation")["is_erosion"].count()
    valid_types = type_counts[type_counts >= 10].index.tolist()
    sub = sub[sub["expected_mutation"].isin(valid_types)]

    if sub.empty:
        print("  [skip] erosion_by_channel -- insufficient data")
        return

    pivot        = sub.groupby(["channel", "expected_mutation"])["is_erosion"].mean().unstack(fill_value=0)
    pivot_counts = sub.groupby(["channel", "expected_mutation"])["is_erosion"].count().unstack(fill_value=0)

    fig, ax   = plt.subplots(figsize=(11, 5))
    n_types   = len(pivot.columns)
    n_channels = len(pivot)
    bar_width = 0.8 / n_types
    x         = range(n_channels)

    for idx, mut_type in enumerate(pivot.columns):
        offset = (idx - n_types / 2 + 0.5) * bar_width
        color  = MUTATION_COLORS.get(mut_type, MUTATION_COLORS["other"])
        xpos   = [xi + offset for xi in x]
        bars   = ax.bar(xpos, pivot[mut_type] * 100,
                        width=bar_width, label=mut_type, color=color,
                        edgecolor="white", linewidth=0.4)
        # Annotate each bar with its context count so a 0% bar is clearly
        # distinguishable from "no contexts" (n=0) vs "contexts but no erosion".
        for bar, xi, ch in zip(bars, xpos, pivot.index):
            n = int(pivot_counts.loc[ch, mut_type]) if ch in pivot_counts.index else 0
            if n > 0:
                ax.text(xi, bar.get_height() + 0.5, f"n={n}",
                        ha="center", va="bottom", fontsize=6, rotation=90,
                        color="#555555")

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylabel("Erosion Rate (%)")
    ax.set_title("Mutation Erosion Rate by Channel and Mutation Type\n"
                 "(register proxy: formal broadcast vs informal content)",
                 fontweight="bold", pad=12)
    ax.legend(title="Mutation Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
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
    fig, ax  = plt.subplots(figsize=(12, max(3, n * 0.9)))

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
    ax.set_yticklabels(channels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlabel("% of evaluable mutation outcomes (Correct / Erosion / Wrong type)")
    ax.set_title("Evaluable Mutation Outcome Distribution by Channel\n"
                 "(phantom omissions excluded -- correct by construction; "
                 "\u03ba = Cohen's Kappa, spaCy vs Cysill, shown separately, right of each bar)",
                 fontweight="bold", pad=12)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
               loc="upper center", bbox_to_anchor=(0.5, -0.05),
               ncol=2, fontsize=9)
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


def fig_erosion_over_batches(df, batch_log, register=None, filename_prefix=""):
    """
    Line chart: overall erosion rate per batch run, ordered chronologically.
    Lets you track whether successive corpus runs converge on a stable estimate.

    PATCH: takes an optional `register` filter ("formal" / "informal" /
    "casual"). A single batch/run can mix videos from multiple registers
    (e.g. a queue that pulled BBC Radio Cymru and Hansh videos together),
    so filtering by
    register happens on the rows within each batch, not by excluding whole
    batches. Needs its own >=2-batches-with-data check per register, since a
    register that's only ever appeared in one run so far can't show a trend
    yet even if the unfiltered pool has plenty of batches.
    """
    if "batch" not in df.columns or "is_erosion" not in df.columns:
        print(f"  [skip] {filename_prefix}erosion_over_batches -- missing columns")
        return

    sub = df
    if register is not None:
        if "channel_register" not in df.columns:
            print(f"  [skip] {filename_prefix}erosion_over_batches -- missing channel_register column")
            return
        sub = sub[sub["channel_register"] == register]

    sub = sub[sub["is_code_switch"].isin([False])] if "is_code_switch" in sub.columns else sub
    # Restrict to evaluable contexts only
    if "status" in sub.columns:
        sub = sub[sub["status"].isin(EVALUABLE_STATUSES)]

    n_batches = sub["batch"].nunique() if "batch" in sub.columns else 0
    if n_batches < 2:
        print(f"  [skip] {filename_prefix}erosion_over_batches -- need at least 2 batches "
              f"with {register or 'any'} data to plot a trend (found {n_batches})")
        return

    batch_rates = sub.groupby("batch").agg(
        contexts=("is_erosion", "count"),
        erosion_rate=("is_erosion", "mean"),
    ).reset_index().sort_values("batch")

    fig, ax = plt.subplots(figsize=(max(6, len(batch_rates) * 0.8), 4))
    ax.plot(batch_rates["batch"], batch_rates["erosion_rate"] * 100,
            marker="o", linewidth=2, color="#4C72B0", markersize=6)
    ax.fill_between(batch_rates["batch"], batch_rates["erosion_rate"] * 100,
                    alpha=0.15, color="#4C72B0")

    for _, row in batch_rates.iterrows():
        ax.annotate(f"{row['erosion_rate']*100:.1f}%\n(n={row['contexts']})",
                    (row["batch"], row["erosion_rate"] * 100),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)

    title_suffix = f" -- {register.capitalize()} register" if register else ""
    ax.set_xlabel("Batch (run timestamp)")
    ax.set_ylabel("Erosion Rate (%)")
    ax.set_title(f"Overall Erosion Rate Across Successive Batches{title_suffix}\n"
                 "(convergence check)", fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, f"{filename_prefix}erosion_over_batches.png")


def fig_collision_flags(df):
    """
    Simple count bar: how many rows were flagged as known homograph
    collisions, broken down by the collision word.
    """
    if "collision_flag" not in df.columns:
        print("  [skip] collision_flags -- missing column")
        return

    flagged = df[df["collision_flag"].notna()]
    if flagged.empty:
        print("  [skip] collision_flags -- no collisions flagged in this corpus")
        return

    counts = flagged["following_word"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.4)))
    ax.barh(counts.index, counts.values, color="#937860", edgecolor="white")
    ax.set_xlabel("Occurrences")
    ax.set_title("Known Homograph Collision Cases\n"
                 "(mutation status unreliable -- manual review recommended)",
                 fontweight="bold", pad=12)
    ax.invert_yaxis()
    for i, (word, count) in enumerate(counts.items()):
        ax.text(count + 0.2, i, str(count), va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, "collision_flags.png")


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

        if "channel_register" in video_level.columns:
            print("  by register:")
            for reg, group in video_level.groupby("channel_register"):
                reg_n = len(group)
                if has_duration:
                    reg_min = group["video_duration_seconds"].fillna(0).sum() / 60
                    print(f"    {reg:<12} {reg_n:>4} videos, {reg_min:>8,.1f} min")
                else:
                    print(f"    {reg:<12} {reg_n:>4} videos")

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
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Welsh Mutation Corpus Analyzer")
    print(f"Reading from: {MUT_DIR}\n")

    df, batch_log = load_and_merge_mutations()
    validate_columns(df)
    df = prepare(df)

    merged_path = OUT_DIR / "merged_mutations.csv"
    df.to_csv(merged_path, index=False, encoding="utf-8-sig", quoting=1)
    print(f"Merged CSV saved: {merged_path.name}\n")

    print_corpus_summary(df, batch_log)

    print("\nGenerating figures...")
    fig_erosion_by_type(df, register="formal",   filename_prefix="formal_")
    fig_erosion_by_type(df, register="informal", filename_prefix="informal_")
    fig_erosion_by_type(df, register="casual",   filename_prefix="casual_")
    fig_erosion_by_rule(df, register="formal",   filename_prefix="formal_")
    fig_erosion_by_rule(df, register="informal", filename_prefix="informal_")
    fig_erosion_by_rule(df, register="casual",   filename_prefix="casual_")
    fig_erosion_by_channel(df)
    fig_codeswitch_by_channel(df)
    fig_status_distribution(df)
    fig_tagger_agreement(df)
    fig_erosion_over_batches(df, batch_log, register="formal",   filename_prefix="formal_")
    fig_erosion_over_batches(df, batch_log, register="informal", filename_prefix="informal_")
    fig_erosion_over_batches(df, batch_log, register="casual",   filename_prefix="casual_")
    fig_collision_flags(df)

    export_utterance_level(df)

    print(f"\nAll outputs written to: {OUT_DIR}")
    print(f"Figures in:             {FIG_DIR}")


if __name__ == "__main__":
    main()