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
import re
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

# PATCH: restructured from five separate top-level folders (audio/,
# transcriptions/, mutations/, captions/, summaries/) into one runs/
# tree, requested explicitly: everything for one video (transcript,
# both mutation files, both audio files, captions) now lives together
# in ONE folder -- runs/<stamp>/<slug>/ -- with no further subfolders
# inside it, so nothing needs an extra click to reach. Session-level
# summary CSVs (research_summary etc. -- belong to the whole run, not
# one video) sit one level up, directly in runs/<stamp>/, alongside
# that run's video folders.
#
#   runs/<stamp>/research_summary_<stamp>.csv         (session-level)
#   runs/<stamp>/erosion_by_trigger_type_<stamp>.csv  (session-level)
#   runs/<stamp>/erosion_by_rule_<stamp>.csv          (session-level)
#   runs/<stamp>/<slug>/segments_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/words_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/lemmas_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/pos_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/mutations_original_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/mutations_corroborated_<stamp>_<slug>.csv
#   runs/<stamp>/<slug>/<video_id>_<title>.mp3        (raw audio)
#   runs/<stamp>/<slug>/<video_id>_<title>_norm.mp3   (normalized audio)
#   runs/<stamp>/<slug>/<video_id>.<lang>.vtt / .csv  (captions)
#
# TRANS_DIR/MUT_DIR/CAPTIONS_DIR/AUDIO_DIR/SUMMARY_DIR are kept as
# aliases, all pointing at RUNS_DIR, purely so any other module that
# still imports those names (mutation_engine.py, corpus_ops.py) doesn't
# break. Don't build new paths from them directly -- use RUNS_DIR, or
# better, the "video_dir" a call already has from _video_slug(), so
# it's obvious you mean "this video's one folder."
RUNS_DIR      = BASE_DIR / "runs"
AUDIO_DIR = TRANS_DIR = MUT_DIR = CAPTIONS_DIR = SUMMARY_DIR = RUNS_DIR

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

OUT_DIR      = BASE_DIR / "analysis"
# PATCH: figures moved out of analysis/figures/ to a plain top-level
# folder -- requested explicitly, figures live at the same level as
# runs/ and the state JSON files, not nested under "analysis".
FIG_DIR      = BASE_DIR / "figures"

# Persistent lemma cache path.
LEMMA_CACHE_PATH = BASE_DIR / "lemma_cache.json"

# Dedicated home for ad-hoc "test a Welsh phrase" output (menu option 4,
# via run_paths() below). Because corpus_analyzer.py and rerun_rules.py
# both aggregate the real corpus via discover_mutation_files() (below) --
# recursive, so nesting alone doesn't help -- a throwaway typed-phrase
# test (explicitly "checking a linguistic rule against a specific
# example", not corpus contribution) would silently get swept into the
# real erosion-rate research figures alongside genuine video data. Giving
# phrase-test output its own directory entirely outside RUNS_DIR's tree
# means the discovery walk simply never sees it.
PHRASE_TEST_DIR = BASE_DIR / "phrase_tests"

# Dedicated home for local-MP3 "preview, don't save" output -- the Testing
# menu's save-or-preview toggle. Same rationale as PHRASE_TEST_DIR just
# above: a sibling of RUNS_DIR, outside its tree, so a preview run is
# never picked up by the corpus-aggregating discovery walk -- something
# you haven't decided to keep shouldn't silently become part of your
# erosion-rate figures.
PREVIEW_DIR = BASE_DIR / "mp3_previews"


# ========================= CHANNEL CONFIGURATION =========================
# PATCH: relocated here from mutation_engine.py, where it sat despite that
# module's own docstring claiming to be "self-contained... independent of
# how audio gets fed into it" -- channel/video-discovery config isn't
# mutation linguistics, it's exactly the "where does config/state live"
# concern this module already owns everything else of. Moved as part of
# the file-naming pass (see the project's branch-prefix convention) since
# a mutation-branch file shouldn't own config every branch needs.
#
# PATCH: entries no longer carry a "channel_register" tag. That hand-
# assigned, per-channel formal/informal/casual label couldn't detect an
# unusually formal/casual episode from an otherwise-typical channel, and
# there was no way to check whether the label was actually right -- it
# wasn't derived from anything measurable. Replaced by a grounded,
# continuous, per-VIDEO formality score computed from transcript data
# already collected (see corpus_formality.py) -- computed after the fact,
# not assigned up front, so it isn't something a CURATED_CHANNELS entry
# can carry at all.
#
# Sgorio removed: nominally Welsh-language sports coverage, but verified
# (Vkq5, 2026-07) to be virtually all-English in practice -- was silently
# contaminating the corpus with non-Welsh audio.
CURATED_CHANNELS = [
    {"url": "https://www.youtube.com/c/HanshS4C/videos"},
    {"url": "https://www.youtube.com/@RowndaRownd/videos"},
    {"url": "https://www.youtube.com/@S4C/videos"},
    {"url": "https://www.youtube.com/@BBCRadio_Cymru/videos"},
    # PATCH: casual, fully-spontaneous-speech sources (not YouTube channels
    # -- see corpus_ops.py's _resolve_entry_url()/_discover_ypod_json() for
    # how discover_new_videos() handles non-YouTube sources). Each carries
    # an explicit "name" -- unlike a YouTube channel URL (where the slug in
    # the URL itself, e.g. "HanshS4C", already reads fine), an RSS path or
    # cache filename ("rss", "podcast-cwins.json?v=1") tells a human
    # nothing about the show. prompt_channel_selection() in corpus_ops.py
    # prefers this field when present.
    #
    # Haclediad -- long-running (since 2010), fully spontaneous unscripted
    # peer conversation between three friends, once a month. Direct
    # podcast RSS feed, confirmed via haclediad.cymru/subscribe.
    # PATCH: haclediad.cymru/rss (what /subscribe pointed at) wasn't
    # actually resolving through yt-dlp -- confirmed via search index that
    # the real canonical feed is hosted directly on Fireside, not served
    # reliably at the custom-domain path. Swapped to the confirmed URL.
    {"url": "https://feeds.fireside.fm/haclediad/rss", "name": "Haclediad"},
    # Colli'r Plot (Y Pod) -- four novelists chatting about books and
    # whatever else, unscripted. RSS feed via its Spreaker host.
    # PATCH: "type": "rss_feed" routes this through corpus_ops.py's
    # _discover_rss_feed() (direct XML parsing) instead of yt-dlp's
    # generic extraction -- confirmed live that yt-dlp's extract_flat
    # misresolved every entry on this specific feed to the feed's own
    # URL instead of each episode's, silently producing duplicate audio
    # across an entire batch. See _discover_rss_feed()'s own docstring.
    {"url": "https://www.spreaker.com/show/5059223/episodes/feed",
     "name": "Colli'r Plot", "type": "rss_feed"},
    # Pryd ar Dafod (Y Pod) -- casual food-and-chat interview podcast.
    # Confirmed working: Anchor/Spotify-for-Podcasters publishes a
    # standard public RSS feed for this show, so it goes through the
    # normal yt-dlp path like Haclediad/Colli'r Plot -- no need for
    # the _discover_ypod_json cache adapter after all. That adapter is
    # left in corpus_ops.py in case a future Y Pod-only show turns out
    # not to have a real feed the way this one did.
    {"url": "https://anchor.fm/s/1064d88dc/podcast/rss",
     "name": "Pryd ar Dafod"},
    # Siarad Siop efo Mari a Meilir (Y Pod) -- casual celebrity/pop-culture
    # banter between two friends, unscripted. Confirmed: this show's Y Pod
    # cache ID is "cwins" (a legacy name from when it started in 2023 as a
    # RuPaul's Drag Race UK recap podcast, before broadening into general
    # pop-culture chat) -- goes through the same ypod_json adapter as Pryd
    # ar Dafod's cache did. Episode audio URLs here are already direct
    # content.rss.com links (no Anchor-style play/redirect wrapper), so
    # _extract_direct_media_url() passes them through unchanged -- verified
    # working against this show's actual cache response, not guessed.
    {"url": "https://ypod.cymru/beta/s/cache/podcast-cwins.json?v=1",
     "name": "Siarad Siop efo Mari a Meilir", "type": "ypod_json"},
]


def ensure_dirs():
    """Creates every output folder up front so the whole output layout is
    visible from the very first run, regardless of which menu options get
    used afterward -- rather than each folder appearing lazily the first
    time whichever menu option needs it actually runs."""
    for p in [BASE_DIR, RUNS_DIR, LOCAL_MP3_DIR, OUT_DIR, FIG_DIR,
              PHRASE_TEST_DIR, PREVIEW_DIR, CHECKPOINT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# You (or migrate_to_new_structure.py's --summaries-only pass) sometimes
# rename a runs/<stamp> folder to append a personal tracking note --
# "20260720_212400 (NULL SUMMARY)" while auditing what's missing, say.
# That annotation is real and useful, but it must never leak into a
# newly WRITTEN filename: a naive f"research_summary_{stamp}.csv" built
# from that raw folder name would produce
# "research_summary_20260720_212400 (NULL SUMMARY).csv" -- a brand new
# file sitting right next to the correctly-named original, not an
# update to it. clean_stamp() strips that trailing annotation so any
# code building a stamp-based FILENAME (never a directory path -- the
# actual folder on disk keeps whatever name it currently has) gets back
# to the real stamp underneath.
_STAMP_ANNOTATION_RE = re.compile(r"\s*\([^)]*\)\s*$")


def clean_stamp(raw_stamp):
    """Strip a trailing user-added " (...)" tracking annotation off a
    stamp string before it's used to build a new filename. A stamp with
    no annotation is returned unchanged."""
    return _STAMP_ANNOTATION_RE.sub("", raw_stamp).strip()


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
        "prep_mutations": PHRASE_TEST_DIR / f"prep_mutations_{stamp}.csv",
        "plural_mutations": PHRASE_TEST_DIR / f"plural_mutations_{stamp}.csv",
        "numeral_mutations": PHRASE_TEST_DIR / f"numeral_mutations_{stamp}.csv",
    }


def _video_slug(meta, stamp):
    """
    One flat folder per video: runs/<stamp>/<slug>/ holds the transcript
    CSVs, both mutation files, both audio files, and captions together --
    no further subfolders, so nothing needs an extra click to reach.

        runs/<stamp>/<slug>/segments_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/words_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/lemmas_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/pos_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/mutations_original_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/mutations_corroborated_<stamp>_<slug>.csv
        runs/<stamp>/<slug>/<video_id>_<title>.mp3        (raw audio)
        runs/<stamp>/<slug>/<video_id>_<title>_norm.mp3   (normalized)
        runs/<stamp>/<slug>/<video_id>.<lang>.vtt / .csv  (captions)

    `stamp` is generated once per menu-loop iteration in welsh_pipeline.py
    (see run_stamp()) and reused for every video processed in that single
    run, so every video from the same invocation lands under the same
    <stamp> parent folder -- browsing by run, then by video within it.
    Filenames still carry both stamp and slug so corpus_analyzer.py's
    filename-based parsing keeps working regardless of nesting depth.

    "mutations" (original, direct pipeline output) and
    "mutations_corroborated" (written separately by
    fetch_captions.run_corroboration -- see its own docstring) are now
    two SEPARATE files, not one file overwritten in place with a backup.
    "mutations" is the key every other module already checks to decide
    whether a video actually produced data (cleanup_incomplete_video_dirs,
    etc.), so that key name is kept as-is -- only its filename changed.

    "captions_dir" and "audio_dir" both point at this same flat video_dir
    now (previously separate nested trees) -- kept as dict keys purely so
    call sites that already do vpaths["captions_dir"] / vpaths["audio_dir"]
    keep working unchanged.
    """
    import re
    title = meta.get("title") or meta.get("id") or stamp
    # keep only alphanumeric, spaces, hyphens; collapse whitespace; truncate
    slug = re.sub(r"[^\w\s-]", "", str(title), flags=re.UNICODE)
    slug = re.sub(r"[\s]+", "_", slug.strip())[:60]
    slug = slug or "untitled"

    folder_name = f"{stamp}_{slug}"
    video_dir = RUNS_DIR / stamp / slug
    video_dir.mkdir(parents=True, exist_ok=True)

    return {
        "segments": video_dir / f"segments_{folder_name}.csv",
        "words":    video_dir / f"words_{folder_name}.csv",
        "lemmas":   video_dir / f"lemmas_{folder_name}.csv",
        "pos":      video_dir / f"pos_{folder_name}.csv",
        "mutations":              video_dir / f"mutations_original_{folder_name}.csv",
        "mutations_corroborated": video_dir / f"mutations_corroborated_{folder_name}.csv",
        # PATCH (Phase 5): own file for the conjugated-preposition branch --
        # a separate schema/phenomenon from mutation erosion, not merged
        # into the mutations_*.csv files (see prep_engine.py's docstring).
        "prep_mutations": video_dir / f"prep_mutations_{folder_name}.csv",
        # PATCH (Phase 6): same pattern for the plural-marking branch (see
        # plural_engine.py's docstring).
        "plural_mutations": video_dir / f"plural_mutations_{folder_name}.csv",
        "numeral_mutations": video_dir / f"numeral_mutations_{folder_name}.csv",
        "captions_dir": video_dir,
        "audio_dir":    video_dir,
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
        "mutations":video_dir / f"mutations_original_{folder_name}.csv",
        "prep_mutations": video_dir / f"prep_mutations_{folder_name}.csv",
        "plural_mutations": video_dir / f"plural_mutations_{folder_name}.csv",
        "numeral_mutations": video_dir / f"numeral_mutations_{folder_name}.csv",
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
        # captions_dir and audio_dir are themselves directories; every
        # other key is a filename inside that video's transcription/
        # mutation folder -- take .parent for those.
        dirs_to_remove.add(p if key in ("captions_dir", "audio_dir") else p.parent)

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


# PATCH: the session-level counterpart to cleanup_incomplete_video_dirs()
# above. That function cleans up ONE video's own folder when its attempt
# produced nothing -- but if EVERY video in a queue-processing/local-MP3
# session fails, each gets individually cleaned and the parent
# runs/<stamp>/ folder is left behind, now genuinely empty, for a human
# to notice and delete by hand. Call this once, right after a session's
# main loop finishes (not after single ad-hoc actions like a phrase
# test, which never write under RUNS_DIR at all -- this is a safe no-op
# for those, since RUNS_DIR/stamp simply won't exist).
def cleanup_empty_session_dir(stamp):
    """
    If runs/<stamp>/ ended up completely empty -- nothing anywhere in
    its tree, recursively -- removes the folder itself. Only fires when
    the session genuinely produced zero output: if even one video
    succeeded (even with zero mutation rows found -- a legitimate
    result, not a choke), its segments/words/pos CSVs are still sitting
    there and this is a no-op, same "don't touch real data" discipline
    as cleanup_incomplete_video_dirs(). Returns True if the folder was
    removed.
    """
    session_dir = RUNS_DIR / stamp
    if not session_dir.exists():
        return False
    if any(session_dir.rglob("*")):
        return False
    shutil.rmtree(session_dir, ignore_errors=True)
    tqdm.write(f"  🧹 Session {stamp} produced no results at all -- "
               f"removed the empty runs/{stamp}/ folder.")
    return True