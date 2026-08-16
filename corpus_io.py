"""
corpus_io.py
============
Single home for every "where does this project's state/output actually
live, and how does it get read/written" concern. Before this module
existed, that question had no single answer -- it was split across
mutation_engine.py (which claims in its own docstring to be "pure
linguistics... independent of how audio gets fed into it", despite
owning ~400 lines of directory config, output-path layout, lemma-cache
persistence, and checkpoint I/O), corpus_ops.py (queue/processed/failed
JSON logs, mixed in among download/analyze/email code), and a small
CSV-append closure defined locally inside welsh_pipeline.py's main().

That split is exactly the shape of bug this project has already been
bitten by more than once (the .parents[N] hop-count bugs in
fetch_captions.py/manual_editing.py/rerun_rules.py, the flat-vs-nested
layout confusion) -- logic about where things live, owned by whichever
file happened to need it first, rather than by one file whose whole job
is knowing that. Consolidating it here means "how is a video's output
laid out on disk" and "how does a JSON state log get written" each have
exactly one place to check, not three.

What deliberately did NOT move here, and why:
  - yt_dlp_cookie_opts() stays in mutation_engine.py -- it's yt-dlp auth
    config, not state/output persistence.
  - LEMMA_CACHE (the in-memory dict) and get_welsh_lemma() etc. stay in
    mutation_engine.py -- that's linguistic engine state that happens to
    be persisted, not a logging concern. This module only owns the
    generic "read/write this cache to disk" mechanics
    (load_lemma_cache_json/save_lemma_cache_json); mutation_engine.py's
    own load_lemma_cache()/save_lemma_cache() wrappers call into those
    with its own module-level LEMMA_CACHE dict, so this module never
    needs to import mutation_engine (which would create a circular
    import, since mutation_engine needs BASE_DIR etc. from here).
  - Checkpoint fingerprinting (_checkpoint_fingerprint) is generic enough
    (hashes a word list) to live here too, but the actual chunk-resume
    DECISION logic stays in mutation_engine.enrich_words() -- this module
    only owns reading/writing/validating the checkpoint file itself.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from youtube_access import configure as configure_youtube_access

# ========================= DIRECTORY / PATH CONFIG =========================
# Keep data outside the source tree so the program can be copied, installed,
# or run from any working directory. Set WELSH_ANALYSIS_DIR to override this
# location (for example, to use an external drive or a shared project folder).
BASE_DIR      = Path(os.environ.get("WELSH_ANALYSIS_DIR",
                                   str(Path.home() / "welsh_analysis"))).expanduser()
configure_youtube_access(BASE_DIR)

AUDIO_DIR     = BASE_DIR / "audio"
TRANS_DIR     = BASE_DIR / "transcriptions"
MUT_DIR       = BASE_DIR / "mutations"
# Dedicated home for run-level summary/rollup CSVs (research_summary,
# erosion_by_trigger_type, erosion_by_rule) so they don't get mixed in with
# the per-video transcription CSVs in TRANS_DIR.
SUMMARY_DIR   = BASE_DIR / "summaries"
VIDEO_QUEUE   = BASE_DIR / "video_queue.json"
PROCESSED_LOG = BASE_DIR / "processed_videos.json"
LOCAL_MP3_DIR = BASE_DIR / "test_audio"
# Mirrors PROCESSED_LOG but for local MP3 batches (menu option 1), which
# previously had no resume capability at all -- a crash partway through a
# folder of local files meant reprocessing everything from scratch, at
# 25-50+ min/video on this hardware. Keyed by filename, with file size
# recorded so a same-named file that's actually been swapped out gets
# reprocessed rather than incorrectly skipped.
LOCAL_PROCESSED_LOG = BASE_DIR / "processed_local_mp3s.json"
# Queue videos that fail (download error, transcription crash, etc.) used
# to get silently marked "processed" forever with no record of why -- a
# transient network blip meant losing that video permanently. This tracks
# per-video attempt count and last error so failures are retried a bounded
# number of times before being given up on, instead of either infinite-
# retrying a permanently broken video or silently dropping a good one.
FAILED_LOG         = BASE_DIR / "failed_videos.json"
FAILED_MAX_RETRIES = 3
# LOCAL_PROCESSED_LOG/FAILED_LOG above give resume at the WHOLE-VIDEO level
# -- a crash means reprocess this video from scratch, which is fine for a
# video that fails fast, but not for one that got hours into the expensive
# Cysill/spaCy tagging pass in enrich_words() before dying (an interrupted
# Cysill run genuinely lost 8 hours of work with nothing on disk to show
# for it). CHECKPOINT_DIR holds CHUNK-level progress within a single
# video's enrich_words() call, one JSON file per in-progress video, so a
# kill/crash partway through tagging resumes from the last completed chunk
# instead of re-tagging (and re-hitting Cysill's rate limit for)
# everything again.
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

CAPTIONS_DIR = BASE_DIR / "captions"
OUT_DIR      = BASE_DIR / "analysis"
FIG_DIR      = OUT_DIR / "figures"

# Persistent lemma cache path.
LEMMA_CACHE_PATH = BASE_DIR / "lemma_cache.json"

# Dedicated home for ad-hoc "test a Welsh phrase" output (menu option 4,
# via run_paths() below). Because corpus_analyzer.py and rerun_rules.py
# both aggregate the real corpus via MUT_DIR.rglob("mutations_*.csv") --
# recursive, so nesting alone doesn't help -- a throwaway typed-phrase
# test (explicitly "checking a linguistic rule against a specific
# example", not corpus contribution) would silently get swept into the
# real erosion-rate research figures alongside genuine video data. Giving
# phrase-test output its own directory entirely outside MUT_DIR's/
# TRANS_DIR's tree means the glob simply never sees it.
PHRASE_TEST_DIR = BASE_DIR / "phrase_tests"

# Dedicated home for local-MP3 "preview, don't save" output -- the Testing
# menu's save-or-preview toggle. Same rationale as PHRASE_TEST_DIR just
# above: a sibling of MUT_DIR/TRANS_DIR, outside either directory's tree,
# so a preview run is never picked up by the corpus-aggregating globs --
# something you haven't decided to keep shouldn't silently become part of
# your erosion-rate figures.
PREVIEW_DIR = BASE_DIR / "mp3_previews"


def ensure_dirs():
    """Creates every output folder up front so the whole output layout is
    visible from the very first run, regardless of which menu options get
    used afterward -- rather than each folder appearing lazily the first
    time whichever menu option needs it actually runs."""
    for p in [BASE_DIR, AUDIO_DIR, TRANS_DIR, MUT_DIR, SUMMARY_DIR,
              LOCAL_MP3_DIR, CAPTIONS_DIR, OUT_DIR, FIG_DIR, PHRASE_TEST_DIR,
              PREVIEW_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_paths(stamp):
    """
    Flat filenames living directly in PHRASE_TEST_DIR -- only ever called
    for menu option 4's ad-hoc "test a Welsh phrase" output (see
    save_analysis_outputs() in corpus_ops.py). Explicitly for checking a
    rule against one example, not for contributing to the research corpus
    -- PHRASE_TEST_DIR sits outside MUT_DIR's/TRANS_DIR's tree entirely
    (see its own comment above) so the aggregating globs never see it,
    regardless of filename.
    """
    return {
        "segments": PHRASE_TEST_DIR / f"segments_{stamp}.csv",
        "words":    PHRASE_TEST_DIR / f"words_{stamp}.csv",
        "lemmas":   PHRASE_TEST_DIR / f"lemmas_{stamp}.csv",
        "pos":      PHRASE_TEST_DIR / f"pos_{stamp}.csv",
        "mutations":PHRASE_TEST_DIR / f"mutations_{stamp}.csv",
    }


def _video_slug(meta, stamp):
    """
    Build a filesystem-safe filename slug for a single video, and organise
    that video's output CSVs into a run-then-video nested folder structure.

    Layout:
        transcriptions/<stamp>/<slug>/segments_<stamp>_<slug>.csv
        transcriptions/<stamp>/<slug>/words_<stamp>_<slug>.csv
        transcriptions/<stamp>/<slug>/lemmas_<stamp>_<slug>.csv
        transcriptions/<stamp>/<slug>/pos_<stamp>_<slug>.csv
        mutations/<stamp>/<slug>/mutations_<stamp>_<slug>.csv
        captions/<stamp>/<slug>/<video_id>.<lang>.vtt
        captions/<stamp>/<slug>/<video_id>.<lang>.csv

    `stamp` is generated once per menu-loop iteration in welsh_pipeline.py
    (see run_stamp()) and reused for every video processed in that single
    run, so every video from the same option-1 or option-3 invocation
    lands under the same <stamp> parent folder -- browsing by run, then by
    video within it. Filenames still carry both stamp and slug (not
    simplified to e.g. "words.csv") so corpus_analyzer.py's existing
    filename-based parsing, and every *.rglob("mutations_*.csv")-style
    discovery call elsewhere, keep working unchanged regardless of
    nesting depth. Falls back to the video id, then the run stamp, if
    neither is available for the slug itself.
    """
    import re
    title = meta.get("title") or meta.get("id") or stamp
    # keep only alphanumeric, spaces, hyphens; collapse whitespace; truncate
    slug = re.sub(r"[^\w\s-]", "", str(title), flags=re.UNICODE)
    slug = re.sub(r"[\s]+", "_", slug.strip())[:60]
    slug = slug or "untitled"

    folder_name = f"{stamp}_{slug}"
    video_trans_dir    = TRANS_DIR / stamp / slug
    video_mut_dir      = MUT_DIR / stamp / slug
    video_captions_dir = CAPTIONS_DIR / stamp / slug
    video_trans_dir.mkdir(parents=True, exist_ok=True)
    video_mut_dir.mkdir(parents=True, exist_ok=True)
    video_captions_dir.mkdir(parents=True, exist_ok=True)

    return {
        "segments": video_trans_dir / f"segments_{folder_name}.csv",
        "words":    video_trans_dir / f"words_{folder_name}.csv",
        "lemmas":   video_trans_dir / f"lemmas_{folder_name}.csv",
        "pos":      video_trans_dir / f"pos_{folder_name}.csv",
        "mutations":video_mut_dir   / f"mutations_{folder_name}.csv",
        "captions_dir": video_captions_dir,
    }


def _preview_video_slug(meta, stamp):
    """
    Mirrors _video_slug() above but writes under PREVIEW_DIR instead of
    TRANS_DIR/MUT_DIR -- used by the Testing menu's "preview, don't save"
    choice for local MP3 analysis. Deliberately a separate small function
    rather than parameterizing _video_slug() with a base-dir argument:
    _video_slug() is also called from the queue-processing path, where
    "preview" isn't a concept at all (queue videos are always real corpus
    data) -- keeping this separate avoids threading an always-unused
    parameter through that call site too.

    No captions_dir here -- local MP3s never fetch captions regardless of
    preview/save, same as the real path.
    """
    import re
    title = meta.get("title") or meta.get("id") or stamp
    slug = re.sub(r"[^\w\s-]", "", str(title), flags=re.UNICODE)
    slug = re.sub(r"[\s]+", "_", slug.strip())[:60]
    slug = slug or "untitled"

    folder_name = f"{stamp}_{slug}"
    video_dir = PREVIEW_DIR / stamp / slug
    video_dir.mkdir(parents=True, exist_ok=True)

    return {
        "segments": video_dir / f"segments_{folder_name}.csv",
        "words":    video_dir / f"words_{folder_name}.csv",
        "lemmas":   video_dir / f"lemmas_{folder_name}.csv",
        "pos":      video_dir / f"pos_{folder_name}.csv",
        "mutations":video_dir / f"mutations_{folder_name}.csv",
    }


# ========================= JSON STATE PERSISTENCE =========================
# Two DIFFERENT writers, deliberately not merged into one -- they look
# similar enough that collapsing them would be exactly the "two things
# that look comparable but aren't" mistake this project has already been
# bitten by (see mutation_engine.py's own comments on the tagger_agreement
# / DOM-Vnoun-conflation bugs). _write_json is for small, human-readable
# state files (queue/processed/failed/local-processed) -- pretty-printed,
# no custom serializer needed since these only ever hold plain
# str/int/bool/list/dict values. _write_json_atomic is for the LARGE
# lemma-cache and checkpoint blobs -- compact (no indent, since pretty-
# printing would meaningfully bloat a multi-thousand-entry cache) and
# tolerant of numpy/pandas scalar types (the `default=` fallback) that
# can end up in a checkpoint's "enriched_words" list.
def _write_json(path, value):
    """Atomically replace a small JSON state file to avoid corrupting it
    on a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _write_json_atomic(path, value):
    """Atomically replace a large JSON blob (lemma cache, checkpoints) --
    compact, and tolerant of non-JSON-native scalar types."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=False,
                                    default=lambda o: float(o) if hasattr(o, "__float__") else str(o)),
                         encoding="utf-8")
    tmp_path.replace(path)


# ---- queue / processed / failed / local-processed logs ----
def load_queue():
    if VIDEO_QUEUE.exists():
        try:
            data = json.loads(VIDEO_QUEUE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []
    return []


def save_queue(queue):
    _write_json(VIDEO_QUEUE, queue)


def load_processed():
    if PROCESSED_LOG.exists():
        try:
            data = json.loads(PROCESSED_LOG.read_text(encoding="utf-8"))
            return set(data) if isinstance(data, list) else set()
        except (OSError, json.JSONDecodeError):
            return set()
    return set()


def save_processed(processed):
    _write_json(PROCESSED_LOG, sorted(processed))


def load_failed():
    if not FAILED_LOG.exists():
        return {}
    try:
        data = json.loads(FAILED_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_failed(failed):
    _write_json(FAILED_LOG, failed)


def record_failure(video, error, failed):
    """Record a failed queue item and return whether it may be retried."""
    # Imported lazily to avoid a hard dependency at module-import time --
    # youtube_access is already a dependency of this module (configure()
    # above), so this is just a local name, not a new coupling.
    from youtube_access import YouTubeRateLimited

    video_id = str(video["id"])
    previous = failed.get(video_id, {})
    # A shared YouTube cooldown is not evidence that this video is bad. Do
    # not consume its finite retry budget simply because the whole service
    # asked this process to pause; keeping it queued makes the run resumable.
    if isinstance(error, YouTubeRateLimited):
        import time as _time
        failed[video_id] = {
            **previous,
            "attempts": int(previous.get("attempts", 0)),
            "last_error": str(error),
            "title": video.get("title", video_id),
            "deferred_until": _time.time() + error.retry_after,
            "last_failed_at": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_failed(failed)
        return True
    import time as _time
    attempts = int(previous.get("attempts", 0)) + 1
    failed[video_id] = {
        "attempts": attempts,
        "last_error": str(error),
        "title": video.get("title", video_id),
        "last_failed_at": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_failed(failed)
    return attempts < FAILED_MAX_RETRIES


def clear_failure(video_id, failed):
    if str(video_id) in failed:
        failed.pop(str(video_id))
        save_failed(failed)


def load_local_processed():
    if not LOCAL_PROCESSED_LOG.exists():
        return {}
    try:
        data = json.loads(LOCAL_PROCESSED_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_local_processed(processed):
    _write_json(LOCAL_PROCESSED_LOG, processed)


# ---- lemma cache persistence ----
# Generic dict-in/dict-out, deliberately NOT touching a module-level cache
# here -- mutation_engine.py owns the actual LEMMA_CACHE dict (it's engine
# runtime state, consulted mid-processing by get_welsh_lemma() etc.), and
# calls these two as thin persistence wrappers around it. See this
# module's own docstring for why that split avoids a circular import.
def load_lemma_cache_json(path=LEMMA_CACHE_PATH):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        tqdm.write(f" ⚠️ Lemma cache unreadable ({e}) -- starting with an empty cache.")
        return {}


def save_lemma_cache_json(cache, path=LEMMA_CACHE_PATH):
    _write_json_atomic(path, cache)


# ---- checkpoint system (chunk-level resume within enrich_words()) ----
def checkpoint_fingerprint(all_preprocessed_words):
    """
    Cheap fingerprint of the exact word sequence enrich_words() was given,
    so a checkpoint can be verified to still match the input before
    resuming from it -- NOT just trusted because a file happens to exist
    with the right name. Guards against the one way resuming could
    silently corrupt data: if this video gets re-transcribed with
    different Whisper settings (or Whisper's own non-determinism) between
    the interrupted attempt and this one, the word list -- and therefore
    chunk_words_for_pos()'s chunk boundaries -- could differ, and splicing
    "chunks 1-17 from the OLD word list" onto "chunks 18+ freshly computed
    from the NEW word list" would misalign every word after the resume
    point. Hashing every word (not just count) catches a same-length but
    reordered/changed transcript, not just a shorter/longer one.
    """
    h = hashlib.sha256()
    for w in all_preprocessed_words:
        h.update(w["word"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def checkpoint_path_for(checkpoint_key):
    """
    Filename is a hash of checkpoint_key (video url/id/audio path -- may
    contain characters that aren't filesystem-safe on every OS this
    project runs on) rather than a slugified version of it. The original
    key is still stored INSIDE the checkpoint JSON for anyone grepping
    CHECKPOINT_DIR by hand to figure out which file belongs to which
    video, without needing the filename itself to be readable.
    """
    digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()[:24]
    return CHECKPOINT_DIR / f"{digest}.json"


def load_enrich_checkpoint(checkpoint_path, checkpoint_key, expected_fingerprint,
                            chunk_word_counts):
    """
    Returns {"completed_chunk_count": int, "enriched_words": [...]} if a
    valid, matching checkpoint exists, else None. "Valid" means: the file
    parses, its fingerprint matches the CURRENT input words, its completed
    chunk count is in range, and its saved-word count exactly matches that
    completed chunk prefix. Any mismatch is treated as "this checkpoint
    doesn't apply anymore" and logged, not silently discarded -- so a
    fingerprint mismatch is visible instead of just quietly re-tagging
    everything with no explanation of why the resume didn't happen.
    """
    if not checkpoint_path.exists():
        return None
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        tqdm.write(f" ⚠️ Checkpoint file unreadable ({e}) -- starting this video fresh.")
        return None

    if data.get("fingerprint") != expected_fingerprint:
        tqdm.write(" ⚠️ Found a checkpoint for this video, but its fingerprint doesn't "
                   "match the current transcription (re-transcribed with different "
                   "settings, or Whisper produced a different result this time) -- "
                   "discarding it and starting this video's tagging fresh, rather than "
                   "risk misaligning words to the wrong chunk.")
        return None

    total_chunks = len(chunk_word_counts)
    completed = data.get("completed_chunk_count", 0)
    enriched  = data.get("enriched_words", [])
    if not isinstance(completed, int) or not (0 <= completed <= total_chunks):
        tqdm.write(" ⚠️ Checkpoint's completed_chunk_count is invalid for this video's "
                   "current chunk count -- starting fresh.")
        return None

    expected_word_count = sum(chunk_word_counts[:completed])
    if not isinstance(enriched, list) or len(enriched) != expected_word_count:
        tqdm.write("Checkpoint word count does not match its completed chunks; "
                   "starting this video fresh to avoid shifted alignment.")
        return None

    return {"completed_chunk_count": completed, "enriched_words": enriched}


def save_enrich_checkpoint(checkpoint_path, checkpoint_key, fingerprint,
                             total_chunks, completed_chunk_count, enriched_words):
    _write_json_atomic(checkpoint_path, {
        "source_key":            checkpoint_key,
        "fingerprint":           fingerprint,
        "total_chunks":          total_chunks,
        "completed_chunk_count": completed_chunk_count,
        "enriched_words":        enriched_words,
    })


def delete_enrich_checkpoint(checkpoint_path):
    checkpoint_path.unlink(missing_ok=True)


# ========================= CSV OUTPUT WRITER =========================
def append_output_csv(df, path, header_flags, idx):
    """
    Appends a DataFrame to `path`, writing a header only the first time
    this (stamp, key) combination is touched -- header_flags is a
    caller-owned list of bools, one per output key (segments/words/
    lemmas/pos/mutations), shared across every video in a run so each
    output CSV gets exactly one header line no matter how many videos
    get appended to it. Was a local closure inside welsh_pipeline.py's
    main() (`_append`); promoted here since it's the same "how does
    output get written to disk" concern as everything else in this
    module, and no different between the queue-processing and local-MP3
    loops that both used to define their own copy of it inline.
    """
    df.to_csv(path, mode="a", header=header_flags[idx], index=False,
              encoding="utf-8-sig", quoting=1)
    header_flags[idx] = False


# ========================= INCOMPLETE-ATTEMPT CLEANUP =========================
# _video_slug() creates a video's transcription/mutation/captions folders
# and welsh_pipeline.py's per-video loops fetch captions into them BEFORE
# calling download_audio() -- so a download failure partway through a
# video (confirmed cause, 2026-08: yt-dlp throwing "HTTP Error 403:
# Forbidden" on the actual media fetch, even though the earlier metadata/
# caption calls succeeded fine) leaves real caption files sitting in an
# otherwise-empty per-video folder tree with no mutations CSV ever
# written, forever, since that video goes back onto the retry queue and
# gets a BRAND NEW stamp/slug pair next attempt. Left alone these orphaned
# folders just accumulate silently -- they're invisible to every
# mutations_*.csv-based glob (corpus_analyzer.py, rerun_rules.py), so they
# never distort the research figures, but they do burn disk and make the
# output tree misleading to browse by hand.
#
# The only reliable signal that an attempt was genuinely incomplete (as
# opposed to a real, legitimately-zero-mutations video, which still gets
# marked "processed" on purpose) is whether vpaths["mutations"] ever got
# written. Call this from any per-video except block, right after
# computing vpaths and before the video goes back onto the retry queue.
def cleanup_incomplete_video_dirs(vpaths, video_label="video"):
    """
    If this attempt's vpaths never got a mutations CSV written, deletes
    the exact stamp/slug folders _video_slug() created for THIS attempt
    (transcription dir, mutation dir, captions dir) -- never anything
    else in TRANS_DIR/MUT_DIR/CAPTIONS_DIR. Returns True if anything was
    removed, so callers can log accordingly.

    Safe no-op if vpaths["mutations"] exists (this attempt did produce
    real data -- e.g. a failure that happened AFTER mutations were
    already written, such as the corroboration pass) or if vpaths is
    falsy/None.
    """
    if not vpaths:
        return False
    mutations_path = vpaths.get("mutations")
    if mutations_path is not None and Path(mutations_path).exists():
        return False  # this attempt DID produce mutation data -- leave it alone

    dirs_to_remove = set()
    for key, value in vpaths.items():
        if value is None:
            continue
        p = Path(value)
        # captions_dir is itself a directory; every other key is a filename
        # inside that video's transcription/mutation folder -- take .parent.
        dirs_to_remove.add(p if key == "captions_dir" else p.parent)

    removed = []
    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))

    if removed:
        tqdm.write(f"  🧹 No mutation data was produced for {video_label} -- "
                   f"removed {len(removed)} empty/partial folder(s) from this "
                   f"attempt: {', '.join(removed)}")
    return bool(removed)
