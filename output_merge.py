"""
output_merge.py
================
Keeps ONE folder per video (or Siarad/Patagonia conversation) under runs/.
When something that already has output is processed again -- after a
pipeline update, or to cover a different stretch of the video -- the new
run's folder absorbs the earlier output, and the earlier folder is moved to
runs/_deleted/<session>/<video>/ (recoverable; logged in
runs/_deleted/deleted_log.txt, which every analysis tool already skips).

Time windows stack. A folder's coverage is a list of [start, end] intervals
in video seconds: coverage.json once this module has written one; for older
folders, the sample-window audio name ("..._sample60-600s.mp3" holds the
true window), else the span of the saved segments. Merging a 180-300s run
into a folder covering 0-240s gives 0-300s:
  - every row of the new run is kept;
  - an earlier segment is dropped if it lies inside the new coverage or
    overlaps any new segment in time -- where both runs cover the same
    audio, the new pipeline's output wins, including what its filters
    threw away;
  - an earlier segment outside that is kept, with every word/lemma/POS/
    detection row that starts inside it. A sampled run drops segments that
    cross its window edge (corpus_ops._shift_and_trim_padded_segments), so
    the earlier run's copy of such a segment is kept unless a new segment
    overlaps it: no gap and no duplicate at the seam.

Kept earlier rows keep their own pipeline_version: a merge extends coverage,
it does not re-run old detection. To bring a stretch up to date, process a
window that covers it.

Per-video totals on every row describe the merged folder afterwards:
video_duration_seconds = total covered length; video_word_count and
video_codeswitch_word_count = the new run's count plus the earlier count
scaled by the share of its words that were kept (an estimate -- per-segment
counts were never saved).

Manual review (mutation_manual_editing.py) is never dropped silently: a
reviewed mutation row the new run replaces hands its review to the new row
with the same rule, trigger, target and status within 1s; otherwise it is
written to superseded_reviews_mutations.csv in the new folder and reported.

An earlier run's tagged word stream (tagged_*.json.gz) is copied along when
any of its rows are kept, so mutation_rerun_rules.py can re-run every row
of the folder; it stacks the regenerated runs with the same _select_old /
absorb / finalize steps as a merge.

Captions are whole-video files, so copy_previous_captions() copies an
earlier .vtt into a new folder before captions are fetched -- the pipeline
then uses it instead of asking YouTube again.

Not part of pipeline_version: this changes where rows are stored, not how
they are detected.
"""
import csv
import json
import shutil
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

from corpus_io import RUNS_DIR, _replace_with_retry, data_folders, _segments_stamp

# vpaths key -> file-name prefix, as corpus_io._video_slug() names them
DATA_FILES = {
    "segments": "segments", "words": "words", "lemmas": "lemmas", "pos": "pos",
    "mutations": "mutations_original", "prep_mutations": "prep_mutations",
    "plural_mutations": "plural_mutations", "numeral_mutations": "numeral_mutations",
    "quantifier_mutations": "quantifier_mutations",
}
CORROBORATED_PREFIX = "mutations_corroborated"
SKIP_SUFFIXES = ("_rerun_candidate.csv", "_precaption_backup.csv")
COVERAGE_FILE = "coverage.json"
REVIEWS_FILE = "superseded_reviews_mutations.csv"
DELETED_DIR = RUNS_DIR / "_deleted"

TOL = 0.05                  # seconds: absorbs timestamp rounding
JOIN_GAP = 0.5              # windows closer than this count as one interval
REVIEW_MATCH_SECONDS = 1.0
REVIEW_COLUMNS = ("manual_reviewed", "is_erosion", "is_erosion_original", "flagged",
                  "flag_note", "review_count", "review_log")
REVIEW_KEY = ("timestamp", "trigger_word", "following_word", "rule")
REVIEW_MATCH = ("rule", "trigger_word", "following_word", "status")
COUNT_COLUMNS = ("video_word_count", "video_codeswitch_word_count")

SAMPLE_RE = re.compile(r"_sample(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s\.mp3$", re.IGNORECASE)
YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})")


# ========================= SMALL HELPERS =========================
def video_key(url):
    """Identifies the same video across runs: the YouTube id when there is
    one (URL spellings differ), else the url/identifier itself
    ("siarad:davies1", "patagonia:01", a local file path)."""
    url = str(url or "").strip()
    m = YOUTUBE_ID_RE.search(url)
    return m.group(1) if m else url


def _is_true(series):
    return series.astype(str).str.strip().str.lower() == "true"


def _read_csv(path):
    """Everything as text, so a rewrite keeps every cell exactly as it was."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return None


def _find_file(folder, prefix):
    for p in sorted(Path(folder).glob(f"{prefix}_*.csv")):
        if not p.name.endswith(SKIP_SUFFIXES):
            return p
    return None


def _load(folder, paths=None):
    """{vpaths key: DataFrame} for each data file present. `paths` (the new
    run's vpaths) names the files exactly; otherwise they're found by prefix."""
    frames = {}
    for key, prefix in DATA_FILES.items():
        p = Path(paths[key]) if paths else _find_file(folder, prefix)
        if p is not None and p.exists():
            df = _read_csv(p)
            if df is not None:
                frames[key] = df
    return frames


def _first(df, col):
    if df is None or df.empty or col not in df.columns:
        return None
    return df[col].iloc[0]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_seconds(x):
    return str(int(round(x))) if abs(x - round(x)) < 1e-6 else f"{x:.3f}".rstrip("0").rstrip(".")


def _start_times(df, key):
    """Start time (s) of each row: segment_start / word_start / the first
    half of a detection row's "12.3s - 13.4s" timestamp. NaN where unknown."""
    if key == "segments":
        col = df["segment_start"] if "segment_start" in df.columns else None
    elif "word_start" in df.columns:
        col = df["word_start"]
    elif "timestamp" in df.columns:
        col = df["timestamp"].astype(str).str.split("s - ").str[0]
    else:
        col = None
    if col is None:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(col, errors="coerce")


def _segment_spans(df):
    if df is None or df.empty or "segment_start" not in df.columns:
        return []
    starts = pd.to_numeric(df["segment_start"], errors="coerce")
    ends = pd.to_numeric(df["segment_end"], errors="coerce")
    return [(a, b) for a, b in zip(starts, ends) if pd.notna(a) and pd.notna(b)]


def _prefix_max_ends(spans):
    out, running = [], float("-inf")
    for _, b in spans:
        running = max(running, b)
        out.append(running)
    return out


# ========================= COVERAGE =========================
def _union(intervals):
    merged = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1] + JOIN_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _covered_seconds(intervals):
    return sum(b - a for a, b in intervals)


def _inside(coverage, a, b):
    return any(lo - TOL <= a and b <= hi + TOL for lo, hi in coverage)


def _describe(coverage):
    return ", ".join(f"{_fmt_seconds(a)}-{_fmt_seconds(b)}s" for a, b in coverage)


def read_coverage(folder, frames):
    """(intervals, history) for a folder written earlier -- see the module
    docstring for where the intervals come from."""
    folder = Path(folder)
    manifest = folder / COVERAGE_FILE
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("intervals"):
                return _union(data["intervals"]), list(data.get("history") or [])
        except (OSError, ValueError):
            pass
    windows = []
    for p in folder.glob("*.mp3"):
        m = SAMPLE_RE.search(p.name)
        if m:
            windows.append((float(m.group(1)), float(m.group(2))))
    if not windows:
        spans = _segment_spans(frames.get("segments"))
        if spans:
            windows = [(min(a for a, _ in spans), max(b for _, b in spans))]
    coverage = _union(windows)
    return coverage, [{"session": folder.parent.name, "intervals": coverage,
                       "pipeline_version": _first(frames.get("segments"), "pipeline_version"),
                       "inferred": True}]


def _write_manifest(folder, key, coverage, history):
    data = {"video_key": key, "intervals": coverage,
            "covered_seconds": round(_covered_seconds(coverage), 3), "history": history}
    tmp = Path(folder) / (COVERAGE_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(tmp, Path(folder) / COVERAGE_FILE)


# ========================= FINDING EARLIER OUTPUT =========================
def _folder_key(folder):
    manifest = folder / COVERAGE_FILE
    if manifest.exists():
        try:
            key = json.loads(manifest.read_text(encoding="utf-8")).get("video_key")
            if key:
                return key
        except (OSError, ValueError):
            pass
    segments = _find_file(folder, "segments")
    if segments is None:
        return None
    try:
        with open(segments, encoding="utf-8-sig", newline="") as f:
            row = next(csv.DictReader(f), None)
    except OSError:
        return None
    return video_key(row.get("video_url")) if row else None


def find_previous_folders(key, exclude=None):
    """Earlier folders holding output for this video, newest first. Skips
    runs/_deleted/ and `exclude` (the folder being written now)."""
    exclude = Path(exclude).resolve() if exclude else None
    found = set()
    # any depth: home folders (runs/<source>/.../<doc>/) and staged ones
    # (runs/_sessions/<stamp>/<doc>/) -- corpus_io "SOURCE LAYOUT"
    for folder in data_folders(RUNS_DIR):
        if folder in found or folder.resolve() == exclude:
            continue
        if _folder_key(folder) == key:
            found.add(folder)
    # newest run first: the stamp in the file names (folder names no
    # longer carry one)
    return sorted(found, key=lambda f: (_segments_stamp(f), f.stat().st_mtime), reverse=True)


def copy_previous_captions(video_url, dest_dir):
    """Copies the caption track(s) of this video's most recent earlier
    folder into `dest_dir`, unless it already has one. Returns how many."""
    dest_dir = Path(dest_dir)
    if any(dest_dir.glob("*.vtt")):
        return 0
    for folder in find_previous_folders(video_key(video_url), exclude=dest_dir):
        vtts = sorted(folder.glob("*.vtt"))
        if vtts:
            for vtt in vtts:
                shutil.copy2(vtt, dest_dir / vtt.name)
            tqdm.write(f"  📎 Reusing captions saved in {folder.parent.name}/ -- not fetched again.")
            return len(vtts)
    return 0


# ========================= MANUAL REVIEWS =========================
def _ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def _overlay_reviews(original, corroborated):
    """mutation_manual_editing.py saves reviews in the corroborated file when a
    folder has one. Copies them onto the matching rows of the original file,
    which is the one merged. Returns (original, reviewed rows with no match)."""
    if corroborated is None or "manual_reviewed" not in corroborated.columns:
        return original, None
    reviewed = corroborated[_is_true(corroborated["manual_reviewed"])]
    if reviewed.empty:
        return original, None
    if original is None:
        return original, reviewed
    original = _ensure_columns(original.copy(), REVIEW_COLUMNS)
    index = {tuple(str(r.get(c, "")) for c in REVIEW_KEY): i for i, r in original.iterrows()}
    unmatched = []
    for _, r in reviewed.iterrows():
        i = index.get(tuple(str(r.get(c, "")) for c in REVIEW_KEY))
        if i is None:
            unmatched.append(r)
            continue
        for c in REVIEW_COLUMNS:
            if c in r.index:
                original.at[i, c] = r[c]
    return original, (pd.DataFrame(unmatched) if unmatched else None)


def _transfer_reviews(current, dropped):
    """Hands the review of each dropped, reviewed row to the new row with the
    same rule/trigger/target/status closest in time (within 1s). Returns
    (current, reviewed rows with no match)."""
    if dropped is None or dropped.empty or "manual_reviewed" not in dropped.columns:
        return current, None
    reviewed = dropped[_is_true(dropped["manual_reviewed"])]
    if reviewed.empty:
        return current, None
    if current is None or current.empty:
        return current, reviewed
    current = _ensure_columns(current.copy(), REVIEW_COLUMNS)
    times = _start_times(current, "mutations")
    taken = set(current.index[_is_true(current["manual_reviewed"]).values])
    unmatched = []
    for _, r in reviewed.iterrows():
        t0 = _to_float(str(r.get("timestamp", "")).split("s - ")[0])
        same = pd.Series(True, index=current.index)
        for c in REVIEW_MATCH:
            if c in current.columns:
                same &= current[c] == str(r.get(c, ""))
        candidates = [i for i in current.index[same.values]
                      if i not in taken and t0 is not None and pd.notna(times[i])
                      and abs(times[i] - t0) <= REVIEW_MATCH_SECONDS]
        if not candidates:
            unmatched.append(r)
            continue
        best = min(candidates, key=lambda i: abs(times[i] - t0))
        for c in REVIEW_COLUMNS:
            if c in r.index:
                current.at[best, c] = r[c]
        taken.add(best)
    return current, (pd.DataFrame(unmatched) if unmatched else None)


def save_stray_reviews(folder, stray_reviews):
    """Appends reviewed rows that no longer match any row to the folder's
    superseded_reviews_mutations.csv."""
    stray_reviews = [s for s in stray_reviews if s is not None and not s.empty]
    if not stray_reviews:
        return
    reviews_path = Path(folder) / REVIEWS_FILE
    existing = _read_csv(reviews_path) if reviews_path.exists() else None
    parts = ([existing] if existing is not None else []) + stray_reviews
    pd.concat(parts, ignore_index=True, sort=False).fillna("").to_csv(
        reviews_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


# ========================= THE MERGE =========================
def _select_old(current, old, coverage):
    """Which rows of an earlier folder survive (see the module docstring).
    Returns ({key: kept rows}, dropped mutation rows, #segments kept,
    #segments dropped)."""
    new_spans = sorted(_segment_spans(current.get("segments")))
    new_starts = [a for a, _ in new_spans]
    new_max_end = _prefix_max_ends(new_spans)

    def overlaps_new(a, b):
        i = bisect_left(new_starts, b - TOL)      # new segments starting before b
        return i > 0 and new_max_end[i - 1] > a + TOL

    old_segments = old.get("segments")
    if old_segments is None or old_segments.empty:
        return {}, old.get("mutations"), 0, 0
    starts = pd.to_numeric(old_segments["segment_start"], errors="coerce")
    ends = pd.to_numeric(old_segments["segment_end"], errors="coerce")
    keep = pd.Series([pd.notna(a) and pd.notna(b) and not _inside(coverage, a, b)
                      and not overlaps_new(a, b) for a, b in zip(starts, ends)],
                     index=old_segments.index, dtype=bool)

    kept_spans = sorted((a, b) for a, b, k in zip(starts, ends, keep) if k)
    kept_starts = [a for a, _ in kept_spans]
    kept_max_end = _prefix_max_ends(kept_spans)

    def in_kept_segment(t):
        if pd.isna(t):
            return False
        j = bisect_right(kept_starts, t + TOL)
        return j > 0 and kept_max_end[j - 1] >= t - TOL

    kept = {"segments": old_segments[keep]}
    dropped_mutations = None
    for key, df in old.items():
        if key == "segments":
            continue
        mask = _start_times(df, key).map(in_kept_segment).astype(bool)
        kept[key] = df[mask]
        if key == "mutations":
            dropped_mutations = df[~mask]
    return kept, dropped_mutations, int(keep.sum()), int((~keep).sum())


def _share_kept(old, kept):
    for key in ("words", "segments"):
        total = len(old[key]) if old.get(key) is not None else 0
        if total:
            return len(kept.get(key, ())) / total
    return 0.0


def absorb(current, counts, old, kept):
    """Appends the kept earlier rows (from _select_old) to `current`, and adds
    the earlier run's per-video counts, scaled by the share of its words that
    were kept, to `counts`. Returns the pipeline versions of the kept rows."""
    share = _share_kept(old, kept)
    for c in COUNT_COLUMNS:
        old_count = _to_float(_first(old.get("segments"), c))
        if counts.get(c) is not None and old_count is not None:
            counts[c] += old_count * share
    versions = set()
    for k, df in kept.items():
        if df is None or df.empty:
            continue
        if "pipeline_version" in df.columns:
            versions.update(v for v in df["pipeline_version"].unique() if v)
        current[k] = (pd.concat([current[k], df], ignore_index=True, sort=False)
                      if k in current else df.copy())
    return versions


def finalize(current, coverage, counts, title):
    """Stacked frames made consistent: rows in time order, and every row's
    per-video columns describing the whole folder."""
    duration = _covered_seconds(coverage)
    out = {}
    for k, df in current.items():
        df = df.fillna("")
        df = (df.assign(_t=_start_times(df, k))
                .sort_values("_t", kind="mergesort", na_position="last")
                .drop(columns="_t")
                .reset_index(drop=True))
        if "video_duration_seconds" in df.columns:
            df["video_duration_seconds"] = _fmt_seconds(duration)
        for c in COUNT_COLUMNS:
            if c in df.columns and counts.get(c) is not None:
                df[c] = str(int(round(counts[c])))
        if title and "video_title" in df.columns:
            df["video_title"] = title
        out[k] = df
    return out


def write_frames(frames, paths):
    """Writes {key: DataFrame} to paths[key]. Every file is written before any
    is replaced, so a failure leaves the existing files as they were."""
    temp_files = []
    for k, df in frames.items():
        path = Path(paths[k])
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        temp_files.append((tmp, path))
    for tmp, path in temp_files:
        _replace_with_retry(tmp, path)


def _rel(path):
    try:
        return Path(path).resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        return Path(path)


def _retire(old_dir, new_dir, note):
    """Moves an absorbed folder to runs/_deleted/<session>/<video>/ and logs it."""
    dest = DELETED_DIR / old_dir.parent.name / old_dir.name
    n = 2
    while dest.exists():
        dest = DELETED_DIR / old_dir.parent.name / f"{old_dir.name}_{n}"
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_dir), str(dest))
    session = old_dir.parent
    try:
        if session.exists() and not any(session.iterdir()):
            session.rmdir()
    except OSError:
        pass
    with open(DELETED_DIR / "deleted_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  MERGED   "
                f"{_rel(dest)}  →  {_rel(new_dir)}  ({note})\n")


def merge_with_previous(vpaths, video_url, window=None):
    """Merges every earlier folder of this video into the folder just written
    (vpaths), newest first, rewrites its CSVs, writes coverage.json, and moves
    the earlier folders to runs/_deleted/. `window` = [start, end] seconds this
    run covered (inferred from the folder when None). Call it after the run's
    CSVs are written and before caption corroboration. Returns a summary dict."""
    new_dir = Path(vpaths["segments"]).parent
    current = _load(new_dir, vpaths)
    if not current:
        return {"merged": 0}
    key = video_key(video_url)
    new_coverage = _union([window]) if window else read_coverage(new_dir, current)[0]
    version = _first(current.get("segments"), "pipeline_version")
    history = [{"session": new_dir.parent.name, "intervals": new_coverage,
                "pipeline_version": version,
                "processed_at": datetime.now().isoformat(timespec="seconds")}]
    coverage = list(new_coverage)
    previous = find_previous_folders(key, exclude=new_dir)
    if not previous:
        _write_manifest(new_dir, key, coverage, history)
        return {"merged": 0}

    newest_segments = current.get("segments")
    counts = {c: _to_float(_first(newest_segments, c)) for c in COUNT_COLUMNS}
    title = _first(newest_segments, "video_title")
    stray_reviews, absorbed, old_versions = [], [], set()

    for old_dir in previous:
        old = _load(old_dir)
        corroborated_path = _find_file(old_dir, CORROBORATED_PREFIX)
        corroborated = _read_csv(corroborated_path) if corroborated_path else None
        merged_original, stray = _overlay_reviews(old.get("mutations"), corroborated)
        if merged_original is not None:
            old["mutations"] = merged_original
        if stray is not None:
            stray_reviews.append(stray)

        old_coverage, old_history = read_coverage(old_dir, old)
        kept, dropped_mutations, n_kept, n_dropped = _select_old(current, old, coverage)
        updated, stray = _transfer_reviews(current.get("mutations"), dropped_mutations)
        if updated is not None:
            current["mutations"] = updated
        if stray is not None:
            stray_reviews.append(stray)

        old_versions |= absorb(current, counts, old, kept)
        if n_kept:
            # the kept rows' tagged words, so mutation_rerun_rules.py can
            # still re-run them from this folder
            for tagged in old_dir.glob("tagged_*.json.gz"):
                if not (new_dir / tagged.name).exists():
                    shutil.copy2(tagged, new_dir / tagged.name)
        coverage = _union(coverage + old_coverage)
        history.extend(old_history)
        absorbed.append((old_dir, n_kept, n_dropped))

    duration = _covered_seconds(coverage)
    write_frames(finalize(current, coverage, counts, title), vpaths)
    save_stray_reviews(new_dir, stray_reviews)
    _write_manifest(new_dir, key, coverage, history)

    tqdm.write(f"  🔗 Merged {len(absorbed)} earlier folder(s) of this video into this run:")
    for old_dir, n_kept, n_dropped in absorbed:
        try:
            _retire(old_dir, new_dir, f"kept {n_kept} segment(s) outside the new window, "
                                      f"replaced {n_dropped}")
            tqdm.write(f"     {old_dir.parent.name}/{old_dir.name}: kept {n_kept} segment(s) "
                       f"outside this run's window, replaced {n_dropped}; moved to runs/_deleted/")
        except OSError as e:
            tqdm.write(f"     ⚠️ {old_dir}: merged, but couldn't be moved ({e}). Move it to "
                       f"runs/_deleted/ yourself, or its rows will be counted twice.")
    tqdm.write(f"     coverage now {_describe(coverage)} ({_fmt_seconds(duration)}s)")
    older = sorted(old_versions - {version})
    if older:
        tqdm.write(f"  ⚠️  Kept stretches still carry pipeline version(s) {', '.join(older)} -- "
                   f"process those windows again to update them.")
    n_stray = sum(len(s) for s in stray_reviews)
    if n_stray:
        tqdm.write(f"  ⚠️  {n_stray} manually reviewed row(s) had no match in the new output -- "
                   f"saved to {REVIEWS_FILE} in this folder.")
    return {"merged": len(absorbed), "coverage": coverage, "older_versions": older,
            "unmatched_reviews": n_stray}
