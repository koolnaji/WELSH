"""
mutation_engine.py
===================
The Welsh linguistic analysis engine: configuration, the Cysill API client,
spaCy integration, the lemma cache, audio-transcript preprocessing
(hallucination filtering, segment dedup, contraction handling), the mutation
trigger tables, confidence scoring, and the core mutation evaluation/
detection logic (process_comprehensive_mutations).

This module is self-contained -- it doesn't import from corpus_ops or
pipeline. Everything here is "the linguistics", independent of how audio
gets fed into it or how a human drives the CLI.
"""

import os
import unicodedata
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter
import re
from tqdm import tqdm

# spaCy loading/parsing now lives in spacy_tagging.py (see that file for
# SPACY_NLP/SPACY_AVAILABLE/load_spacy/parse_spacy_doc). load_spacy is
# re-exported here for welsh_pipeline.py, which imports it from this module.
from spacy_tagging import load_spacy, parse_spacy_doc
import bangor_lexicon


try:
    from simplemma import lemmatize as simplemma_lemmatize
    from simplemma import is_known as simplemma_is_known
except Exception:
    simplemma_lemmatize = None
    simplemma_is_known = None

# ========================= CONFIGURATION =========================
# Keep data outside the source tree so the program can be copied, installed,
# or run from any working directory.  Set WELSH_ANALYSIS_DIR to override this
# location (for example, to use an external drive or a shared project folder).
BASE_DIR      = Path(os.environ.get("WELSH_ANALYSIS_DIR",
                                   str(Path.home() / "welsh_analysis"))).expanduser()
from youtube_access import configure as configure_youtube_access
configure_youtube_access(BASE_DIR)
AUDIO_DIR     = BASE_DIR / "audio"
TRANS_DIR     = BASE_DIR / "transcriptions"
MUT_DIR       = BASE_DIR / "mutations"
# PATCH: dedicated home for run-level summary/rollup CSVs (research_summary,
# erosion_by_trigger_type, erosion_by_rule) so they don't get mixed in with
# the per-video transcription CSVs in TRANS_DIR.
SUMMARY_DIR   = BASE_DIR / "summaries"
VIDEO_QUEUE   = BASE_DIR / "video_queue.json"
PROCESSED_LOG = BASE_DIR / "processed_videos.json"
LOCAL_MP3_DIR = BASE_DIR / "test_audio"
# PATCH: mirrors PROCESSED_LOG but for local MP3 batches (menu option 1),
# which previously had no resume capability at all -- a crash partway
# through a folder of local files meant reprocessing everything from
# scratch, at 25-50+ min/video on this hardware. Keyed by filename, with
# file size recorded so a same-named file that's actually been swapped out
# gets reprocessed rather than incorrectly skipped.
LOCAL_PROCESSED_LOG = BASE_DIR / "processed_local_mp3s.json"
# PATCH: queue videos that fail (download error, transcription crash, etc.)
# used to get silently marked "processed" forever with no record of why --
# a transient network blip meant losing that video permanently. This tracks
# per-video attempt count and last error so failures are retried a bounded
# number of times before being given up on, instead of either infinite-
# retrying a permanently broken video or silently dropping a good one.
FAILED_LOG           = BASE_DIR / "failed_videos.json"
FAILED_MAX_RETRIES    = 3
# PATCH: LOCAL_PROCESSED_LOG/FAILED_LOG above give resume at the WHOLE-VIDEO
# level -- a crash means reprocess this video from scratch, which is fine
# for a video that fails fast, but not for one that got hours into the
# expensive Cysill/spaCy tagging pass in enrich_words() before dying (an
# interrupted Cysill run genuinely lost 8 hours of work with nothing on
# disk to show for it -- see enrich_words()'s own checkpoint comments).
# CHECKPOINT_DIR holds CHUNK-level progress within a single video's
# enrich_words() call, one JSON file per in-progress video, so a kill/crash
# partway through tagging resumes from the last completed chunk instead of
# re-tagging (and re-hitting Cysill's rate limit for) everything again.
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# PATCH: previously only defined locally inside fetch_captions.py and
# corpus_analyzer.py respectively -- redefined here too (same value, same
# BASE_DIR) purely so ensure_dirs() below can create every output folder
# up front, not just the ones menu option 3 happens to write to. This
# doesn't change where fetch_captions.py/corpus_analyzer.py look; it just
# means the folder already exists by the time they get around to needing
# it, instead of being created lazily on first use.
CAPTIONS_DIR = BASE_DIR / "captions"
OUT_DIR      = BASE_DIR / "analysis"
FIG_DIR      = OUT_DIR / "figures"

# PATCH: YouTube 429s on the caption/timedtext endpoint were assumed to be
# a simple request-rate problem (see fetch_captions.py's backoff/circuit-
# breaker PATCH comments) -- turns out the block that triggers is commonly
# reported as lasting on the order of half a day, per real-world yt-dlp
# users hitting this exact error (yt-dlp issues #7123, #13770, #13831),
# not something any in-process backoff or breaker can wait out. The
# consistent community fix is authenticating the request the same way a
# logged-in browser tab would -- YouTube rate-limits anonymous requests
# far harder than cookied ones. Two ways to supply that, either optional:
#   YTDLP_COOKIES_FROM_BROWSER=chrome   (or firefox/edge/brave/...)
#     Reads cookies live from an installed, logged-in browser profile.
#     Simplest for interactive use, but doesn't travel with the repo --
#     the PCbang machine needs its OWN logged-in browser for this to work,
#     it won't reuse a cookie jar exported from the Windows PC.
#   YTDLP_COOKIES_FILE=C:\path\to\cookies.txt
#     A cookies.txt exported once (e.g. via a "Get cookies.txt" browser
#     extension) and copied to wherever the pipeline runs -- portable
#     across the two-machine workflow this project actually uses, unlike
#     the browser option above.
# If both are set, the file wins (unambiguous, doesn't depend on what's
# installed on the current machine). Neither is required -- omit both and
# yt-dlp runs fully anonymous, exactly as before this existed.
YTDLP_COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
YTDLP_COOKIES_FILE         = os.environ.get("YTDLP_COOKIES_FILE", "").strip()


def yt_dlp_cookie_opts():
    """Returns a dict of yt-dlp opts (possibly empty) to merge into any
    ydl_opts/opts dict in this project, so every yt-dlp call site can opt
    into cookie auth the same way rather than each reimplementing this
    env-var check. Empty dict (no-op) if neither env var is set."""
    if YTDLP_COOKIES_FILE:
        return {"cookiefile": YTDLP_COOKIES_FILE}
    if YTDLP_COOKIES_FROM_BROWSER:
        return {"cookiesfrombrowser": (YTDLP_COOKIES_FROM_BROWSER,)}
    return {}

# PATCH: persistent lemma cache path
LEMMA_CACHE_PATH = BASE_DIR / "lemma_cache.json"

# BUGFIX: dedicated home for ad-hoc "test a Welsh phrase" output (menu
# option 4, via run_paths() below). This used to write straight into
# TRANS_DIR/MUT_DIR using flat filenames (mutations_<stamp>.csv, no
# per-video subfolder). Because corpus_analyzer.py and rerun_rules.py both
# aggregate the real corpus via MUT_DIR.rglob("mutations_*.csv") --
# recursive, so nesting alone doesn't help -- a throwaway typed-phrase
# test (explicitly documented as being for "checking a linguistic rule
# against a specific example", not corpus contribution) would silently
# get swept into the real erosion-rate research figures alongside genuine
# video data. interactive_batch_selection() in corpus_analyzer.py does
# let you spot and exclude it by hand at load time, but nothing stops it
# from being missed, especially once there are many real batches to skim.
# Giving phrase-test output its own directory entirely outside MUT_DIR's/
# TRANS_DIR's tree means the glob simply never sees it -- no reliance on
# a human catching it later.
PHRASE_TEST_DIR = BASE_DIR / "phrase_tests"

# PATCH: dedicated home for local-MP3 "preview, don't save" output --
# the Testing menu's save-or-preview toggle (see welsh_pipeline.py).
# Same rationale as PHRASE_TEST_DIR just above: a sibling of MUT_DIR/
# TRANS_DIR, outside either directory's tree, so a preview run (e.g.
# sanity-checking a new/unfamiliar audio source before trusting it) is
# never picked up by option 6's or rerun_rules.py's *.rglob("mutations_*.csv")
# discovery -- something you haven't decided to keep shouldn't silently
# become part of your erosion-rate figures.
PREVIEW_DIR = BASE_DIR / "mp3_previews"

# PATCH: explicit imports instead of `import *` -- a star import makes
# every name "maybe defined" to any linter/IDE, which is what generated
# the wall of false-positive "possibly undefined" warnings after the
# module split. Spelling out exactly what's used here fixes that.
from cysill_client import (
    TECHIAITH_API_KEY,
    reset_cysill_circuit_breaker,   # re-exported for welsh_pipeline.py
    fetch_lemma, fetch_pos_for_chunk, chunk_words_for_pos,
    CYSILL_CALL_FAILED,
)


# PATCH: CURATED_CHANNELS entries now carry a "channel_register" tag
# alongside the URL. It flows through discover_new_videos -> video queue
# entries -> video_meta -> every output row (segments/words/lemmas/pos/
# mutations), so BBC-vs-S4C erosion-rate comparisons become a
# groupby("channel_register") on the mutation CSVs instead of a manual join
# reconstructed later.
#
# NOTE: deliberately named "channel_register", not "register_class" --
# register_class already exists as a per-mutation-instance column produced
# by classify_register_adjustment() below (colloquial_variant / n/a), which
# is a different concept (documented colloquial variant vs. genuine
# erosion for one specific mutation). channel_register is about which
# source/channel a video came from. Do not merge these two columns.
#
# Sgorio removed: nominally Welsh-language sports coverage, but verified
# (Vkq5, 2026-07) to be virtually all-English in practice -- was silently
# contaminating the "informal Welsh" bucket with non-Welsh audio.
#
# "unverified" register: channel is plausibly Welsh-medium but its actual
# language mix hasn't been spot-checked yet. Don't fold these into any
# side of a formal/informal/casual comparison until checked -- see BBC
# Cymru Wales below, which is a general bilingual channel (TV/Radio/
# Online), not confirmed Welsh-medium.
#
# Register scale, most to least formal: formal (BBC Radio Cymru) >
# informal (Hansh / Rownd a Rownd / S4C -- produced but informal) >
# casual (fully spontaneous, unscripted peer conversation -- e.g.
# podcasts like Haclediad). Casual sources currently only enter via the
# local-MP3 path (option 1's register prompt in welsh_pipeline.py), not
# CURATED_CHANNELS, since none of these are YouTube channels yet.
CURATED_CHANNELS = [
    {"url": "https://www.youtube.com/c/HanshS4C/videos",   "channel_register": "informal"},
    {"url": "https://www.youtube.com/@RowndaRownd/videos", "channel_register": "informal"},
    {"url": "https://www.youtube.com/@S4C/videos",         "channel_register": "informal"},
    {"url": "https://www.youtube.com/@BBCRadio_Cymru/videos", "channel_register": "formal"},
    # PATCH: BBC Cymru Wales (UCNH53DFjL1OBl3oWNxMmC_Q) removed -- it
    # sat as "unverified" since it was added (general bilingual channel,
    # never actually spot-checked as Welsh-medium) and was never going to
    # be usable in a formal/informal/casual comparison in that state.
    # Re-add with a real "name" and a decided register if it's ever
    # verified.
    #
    # PATCH: casual-register additions (not YouTube channels -- see
    # corpus_ops.py's _resolve_entry_url()/_discover_ypod_json() for how
    # discover_new_videos() handles non-YouTube sources). Each carries an
    # explicit "name" -- unlike a YouTube channel URL (where the slug in
    # the URL itself, e.g. "HanshS4C", already reads fine), an RSS path
    # or cache filename ("rss", "podcast-cwins.json?v=1") tells a human
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
    {"url": "https://feeds.fireside.fm/haclediad/rss", "name": "Haclediad",
     "channel_register": "casual"},
    # Colli'r Plot (Y Pod) -- four novelists chatting about books and
    # whatever else, unscripted. RSS feed via its Spreaker host.
    {"url": "https://www.spreaker.com/show/5059223/episodes/feed",
     "name": "Colli'r Plot", "channel_register": "casual"},
    # Pryd ar Dafod (Y Pod) -- casual food-and-chat interview podcast.
    # Confirmed working: Anchor/Spotify-for-Podcasters publishes a
    # standard public RSS feed for this show, so it goes through the
    # normal yt-dlp path like Haclediad/Colli'r Plot -- no need for
    # the _discover_ypod_json cache adapter after all. That adapter is
    # left in corpus_ops.py in case a future Y Pod-only show turns out
    # not to have a real feed the way this one did.
    {"url": "https://anchor.fm/s/1064d88dc/podcast/rss",
     "name": "Pryd ar Dafod", "channel_register": "casual"},
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
     "name": "Siarad Siop efo Mari a Meilir",
     "channel_register": "casual", "type": "ypod_json"},
]

# ========================= MUTATION TABLES =========================
# All pure linguistic data now lives in mutation_tables.py -- see that
# file for every trigger list, mutation dict, and lexicon. Explicit
# imports here (not `import *`) so linters/IDEs can actually verify
# every name instead of flagging all of them as "maybe undefined".
from mutation_tables import (
    SOFT_MUTATION, SOFT_MUTATION_LIMITED, NASAL_MUTATION, ASPIRATE_MUTATION,
    RADICAL_TO_MUTATED, MUTATION_MAP, SPACY_MUTATION_MAP,
    SELECTIVE_INVARIANT_MAP, ASPIRATE_INITIALS,
    TRIGGERS, DEFINITE_ARTICLE_FORMS,
    LL_RH_SOFT_EXEMPT_TRIGGERS_ANY_POS, LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY,
    BOD_SUBJECT_EXEMPT_TRIGGERS, NUMERAL_FEM_SOFT_LIMITED_TRIGGERS,
    NUMERAL_GENERAL_SOFT_TRIGGERS, NASAL_NUMERAL_TRIGGERS,
    NASAL_NUMERAL_VALID_TARGETS, MIXED_MUTATION_TRIGGERS,
    PREPOSED_ADJECTIVE_LEXICON, COMPOUND_NOUN_SECOND_ELEMENT,
    OBJ_DEPS, PHANTOM_CONTEXT_TAGS, KNOWN_HOMOGRAPH_COLLISIONS,
    PREP_TAG_PREFIXES, PREDYN_TAGS, REL_INT_TAGS, VOCAT_DEPS,
    DIGRAPHS, WELSH_FILLERS, WELSH_VOWELS,
    ENGLISH_FUNCTION_WORDS, WELSH_ENGLISH_HOMOGRAPHS, BOD_SURFACE_FORMS,
    WELSH_CONTRACTION_SPLITS, SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS,
    _LEGAL_BASE_MARK_PAIRS,
)



# ========================= HELPERS =========================
def ensure_dirs():
    # PATCH: previously only created the folders menu option 3 needs
    # (audio/transcriptions/mutations/summaries) -- LOCAL_MP3_DIR
    # (test_audio), CAPTIONS_DIR, OUT_DIR (analysis), and FIG_DIR were
    # each created lazily by whichever menu option first needed them
    # (option 1, caption download, option 6). That's not wrong, just
    # confusing to look at from the outside -- a user who's only run
    # option 3 sees a handful of folders and no signal that the others
    # are just as real, only not-yet-triggered. Creating everything up
    # front makes the whole output layout visible from the very first
    # run, regardless of which menu options get used afterward.
    for p in [BASE_DIR, AUDIO_DIR, TRANS_DIR, MUT_DIR, SUMMARY_DIR,
              LOCAL_MP3_DIR, CAPTIONS_DIR, OUT_DIR, FIG_DIR, PHRASE_TEST_DIR,
              PREVIEW_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_paths(stamp):
    """
    BUGFIX: this used to point "mutations" at
    MUT_DIR / f"mutations_{stamp}.csv" -- a flat filename living directly
    in MUT_DIR, matching the exact glob (mutations_*.csv) that
    corpus_analyzer.py and rerun_rules.py both use to aggregate the real
    per-video corpus. run_paths() is only ever called for menu option 4's
    ad-hoc "test a Welsh phrase" output (see save_analysis_outputs in
    corpus_ops.py) -- explicitly documented as being for checking a rule
    against one example, not for contributing to the research corpus. Left
    as-is, every phrase you tested via option 4 would silently become part
    of the real erosion-rate figures the next time you ran option 6.
    Redirected to PHRASE_TEST_DIR (a sibling of MUT_DIR/TRANS_DIR, outside
    either directory's tree) so the aggregating globs -- which are
    recursive and would still find it even nested deeper inside MUT_DIR --
    never see it at all, regardless of filename.
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
    video within it, same as before a previous session flattened this.
    Filenames still carry both stamp and slug (not simplified to e.g.
    "words.csv") so corpus_analyzer.py's existing filename-based parsing,
    and every *.rglob("mutations_*.csv")-style discovery call elsewhere,
    keep working unchanged regardless of nesting depth.
    Falls back to the video id, then the run stamp, if neither is available
    for the slug itself.

    # PATCH: a previous fix collapsed both transcripts (which used to
    # nest two levels deep, TRANS_DIR / stamp / folder_name) AND mutations
    # (which had no per-video subfolder at all, flat under MUT_DIR / stamp)
    # down to a single flat <stamp>_<slug> folder for both -- solving the
    # transcript/mutation inconsistency, but as a side effect also
    # removing the ability to browse a run's videos grouped together under
    # one folder, which turned out to be wanted behavior, not a bug.
    # Restored here the other direction: both output types now nest
    # run-then-video symmetrically, rather than collapsing to flat.
    # find_segments_csv() in fetch_captions.py/manual_editing.py and
    # _mutations_dir_for() in rerun_rules.py all had fast-path candidates
    # hardcoded to the old flat layout -- updated alongside this change.
    #
    # PATCH: added "captions_dir" so caption data (.vtt + the parsed
    # captions CSV) nests the same run-then-video way as transcriptions
    # and mutations, instead of landing flat in CAPTIONS_DIR alongside
    # every other video's captions ever fetched. Unlike the other keys
    # above, this deliberately returns a DIRECTORY, not a filename --
    # download_captions() derives the actual .vtt filename itself from
    # the video ID (yt-dlp's own outtmpl convention), so the caller just
    # needs somewhere to point out_dir at. welsh_pipeline.py's caption-
    # fetch block now runs _video_slug() before fetching captions (it
    # used to run after) specifically so this directory exists in time.
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
    choice for local MP3 analysis (see welsh_pipeline.py). Deliberately a
    separate small function rather than parameterizing _video_slug() with
    a base-dir argument: _video_slug() is also called from the queue-
    processing path, where "preview" isn't a concept at all (queue videos
    are always real corpus data) -- keeping this separate avoids threading
    an always-unused parameter through that call site too.

    No captions_dir here -- local MP3s never fetch captions regardless of
    preview/save, same as the real path.
    """
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

def normalize_word(word):
    if not word:
        return ""
    # PATCH: normalise curly/smart apostrophes before any other processing
    word = word.replace("\u2018", "'").replace("\u2019", "'") \
               .replace("\u201c", '"').replace("\u201d", '"')
    w = unicodedata.normalize("NFC", word.lower().strip(".,!?;:'\"()[]"))
    w = w.replace("'r", "").replace("'n", "yn")
    return w

def initial_cluster(word):
    w = normalize_word(word)
    for d in sorted(DIGRAPHS, key=len, reverse=True):
        if w.startswith(d):
            return d
    return w[0] if w else ""

def mutation_type_from_surface(word):
    return MUTATION_MAP.get(initial_cluster(word))

def is_english_code_switch(word, techiaith_lemma):
    w = normalize_word(word)
    if not w:
        return False
    if w in ENGLISH_FUNCTION_WORDS and w not in WELSH_ENGLISH_HOMOGRAPHS:
        return True
    if simplemma_is_known is not None:
        try:
            if simplemma_is_known(w, lang="en") and not simplemma_is_known(w, lang="cy"):
                if not (w.startswith("ts") or w.startswith("j")):
                    return True
        except Exception:
            pass
    return False

# ========================= TUNING CONSTANTS =========================
# Not linguistic data (that's mutation_tables.py) -- these are pipeline
# behaviour knobs: cache state, confidence thresholds, filter sensitivity.
LEMMA_CACHE = {}
COLLOQUIAL_REGISTER           = True
DECLINING_MUTATIONS           = {"nasal", "aspirate"}
EROSION_CONFIDENCE_THRESHOLD  = 0.60
HALLUCINATION_REPEAT_RATIO    = 0.4
GAP_ALIGN_WINDOW              = 5    # lookahead window for gap-tolerant alignment

# ========================= LEMMA CACHE PERSISTENCE =========================
def load_lemma_cache():
    """Load persisted lemma cache from disk to avoid redundant API calls across runs."""
    if LEMMA_CACHE_PATH.exists():
        try:
            LEMMA_CACHE.update(
                json.loads(LEMMA_CACHE_PATH.read_text(encoding="utf-8"))
            )
            print(f"✅ Lemma cache loaded ({len(LEMMA_CACHE)} entries).")
        except Exception as e:
            print(f"⚠️  Could not load lemma cache: {e}")

def save_lemma_cache():
    """Persist lemma cache to disk for reuse in future runs."""
    try:
        # Match the crash-safe persistence used for queue and checkpoint
        # state. A direct write can leave truncated JSON after an interrupt.
        _write_json_atomic(LEMMA_CACHE_PATH, LEMMA_CACHE)
    except Exception as e:
        print(f"⚠️  Could not save lemma cache: {e}")

# SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS now lives in
# mutation_tables.py (imported via `from mutation_tables import *` above).
def _resolve_suppletive_comparative_radical(raw):
    """
    Returns the correct comparative/superlative radical when `raw` is one of
    SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS itself, or a regularly-
    mutated surface form of one of them (e.g. "well" <- soft-mutated
    "gwell"). Returns None for anything else, so the normal lemma lookup
    (Cysill/simplemma) applies as before.
    """
    raw = normalize_word(raw)
    if not raw:
        return None
    if raw in SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS:
        return raw
    for radical in SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS:
        if raw in expected_surface_forms(radical, ["soft", "soft_limited", "nasal", "aspirate"]):
            return radical
    return None

def load_bangor_lexicon():
    """
    Thin wrapper around bangor_lexicon.load() that degrades gracefully
    instead of crashing pipeline startup -- the lexicon is a traffic-
    reduction optimization, not a hard dependency. Every consumer
    (get_welsh_lemma below, enrich_words) already checks
    bangor_lexicon.is_loaded() and falls through to the live
    Cysill/simplemma path when it's False, so a missing/not-yet-
    downloaded lexicon file should mean "run exactly as before", not
    "pipeline won't start". Same print-and-continue convention as
    load_spacy() in spacy_tagging.py.

    Call this once at pipeline startup (see welsh_pipeline.py's main(),
    right next to load_spacy()) -- NOT automatically on import, and NOT
    lazily inside get_welsh_lemma()/enrich_words() themselves, so a
    missing file produces one clear message at startup instead of a
    silent per-call no-op that's easy to not notice for an entire run.
    """
    try:
        bangor_lexicon.load()
        print(f"✅ Bangor lexicon loaded ({bangor_lexicon.wordform_count():,} wordforms).")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Continuing without it -- lemma/POS lookups will go through "
              "Cysill/simplemma only, same as before this existed.")
        return False


def get_welsh_lemma(word):
    w = normalize_word(word)
    if not w:
        return None
    suppletive_radical = _resolve_suppletive_comparative_radical(w)
    if suppletive_radical:
        return suppletive_radical
    # PATCH: check for English code-switching BEFORE hitting Cysill or
    # simplemma, not after. is_english_code_switch() only ever inspects
    # `word` itself (its second parameter, the lemma, is unused inside the
    # function) -- so there was never a real dependency forcing the lemma
    # lookup to happen first. Previously every English word (~8-10% of
    # this corpus, per the code-switching research question itself) was
    # sent to the Welsh lemmatizer regardless, burning an API call for
    # nothing and sometimes getting "Welshified" into a nonsense answer
    # (observed live: "gosh" -> "cosh", the same failure mode already
    # documented for "the" -> "te" and "for" -> "mor" in
    # ENGLISH_FUNCTION_WORDS' own comment). Skipping the network path
    # entirely for recognized English words fixes both the wasted calls
    # and the garbage cache entries in one place, for every caller of
    # get_welsh_lemma (this was previously only checked downstream in
    # layer_2_lemma_analysis, which doesn't cover corpus_ops.py's
    # per-word loop that actually populates LEMMA_CACHE in the first
    # place).
    if is_english_code_switch(w, None):
        return None
    if w in LEMMA_CACHE:
        return LEMMA_CACHE[w]
    # PATCH: Bangor lexicon lookup BEFORE hitting the live Cysill
    # lemmatizer endpoint -- same "cache-first, network as fallback"
    # shape as the LEMMA_CACHE check just above, just pre-seeded with
    # ~497k wordforms instead of only whatever this run has already
    # looked up. Deliberately uses the lemma-unambiguous check (96.3%
    # coverage, measured against the real file), not a full-entry
    # lookup -- see bangor_lexicon.lemma_if_unambiguous()'s own
    # docstring for why picking among genuinely ambiguous readings here
    # would be a guess, not a lookup. is_loaded() is checked rather than
    # calling load() here, so a missing/not-yet-downloaded lexicon file
    # degrades to "skip this step" rather than crashing every lemma
    # lookup in the pipeline -- see bangor_lexicon.load()'s own error
    # message for how to fetch and place the file.
    if bangor_lexicon.is_loaded():
        bangor_lemma = bangor_lexicon.lemma_if_unambiguous(w)
        if bangor_lemma:
            LEMMA_CACHE[w] = bangor_lemma
            return bangor_lemma
    lemma = None
    cysill_call_failed = False
    # PATCH: route through fetch_lemma so retry/reconnect logic applies
    if TECHIAITH_API_KEY:
        result = fetch_lemma(w)
        if result is CYSILL_CALL_FAILED:
            cysill_call_failed = True
        else:
            lemma = result
    if not lemma and simplemma_lemmatize is not None:
        try:
            fallback = simplemma_lemmatize(w, lang="cy")
            lemma = fallback if fallback and fallback != w else None
        except Exception:
            lemma = None
    # PATCH: only skip caching when Cysill's call itself genuinely failed
    # (rate limit, timeout, circuit breaker) AND we still ended up with
    # nothing. simplemma is local/deterministic -- if it already weighed
    # in (found something, or confirmed nothing), that answer will be
    # identical on every future run, so there's no benefit to leaving it
    # uncached. Only a real, unanswered Cysill call is worth retrying.
    # This also fixes the previous version of this patch, which skipped
    # caching on ANY None result -- including simplemma's confirmed
    # "no distinct lemma" answers, which are stable and were being
    # needlessly re-fetched from the API on every future run.
    if lemma is not None or not cysill_call_failed:
        LEMMA_CACHE[w] = lemma
    return lemma

def extract_gender_from_spacy(spacy_token):
    if not spacy_token:
        return None
    g = spacy_token.get("gender")
    if g == "Fem":
        return "feminine"
    elif g == "Masc":
        return "masculine"
    return None

# PATCH: the fem_noun+adjective / definite_article+fem_noun mutation rules
# (Layers 1B/1C/1C-bis) are specifically a FEMININE SINGULAR phenomenon --
# "merch dda" (good girl) mutates, "merched da" (good girls) does not,
# regardless of gender. Neither rule checked number before this, so a
# plural feminine noun (e.g. "ffrindiau", plural of "ffrind") could
# wrongly fire the rule and count a correctly-unmutated plural adjective
# as erosion. spaCy's UD morph features already carry Number=Sing/Plur on
# every token (see the "morph" dict captured at record-build time), so
# this reads directly off data already being collected, no new dependency.
def extract_number_from_spacy(spacy_token):
    if not spacy_token:
        return None
    n = (spacy_token.get("morph") or {}).get("Number")
    if n == "Plur":
        return "plural"
    elif n == "Sing":
        return "singular"
    return None

def filter_hallucinated_segments(segments):
    clean = []
    for seg in segments:
        # PATCH: faster-whisper computes avg_logprob (the model's own
        # confidence in the tokens it produced) and no_speech_prob (how
        # likely this stretch of audio is non-speech at all) for every
        # segment, for free -- neither was being checked before this, even
        # though they're the most direct signal available for exactly the
        # two failure modes seen in practice: total silence (high
        # no_speech_prob) and confident-sounding nonsense over noise (very
        # low avg_logprob, since the model itself wasn't sure what it was
        # producing even though the repeat-ratio and orthographic checks
        # below don't catch it -- e.g. diverse short words like "I'm, I'm,
        # fuck, fuck, I think" don't repeat enough to trip the ratio check
        # and aren't Welsh so the orthographic check doesn't apply either).
        # Thresholds are deliberately conservative (0.6 / -1.0) since a
        # false-positive here silently deletes real speech -- tune down
        # further only if you're still seeing hallucinations get through.
        no_speech_prob = getattr(seg, "no_speech_prob", 0.0)
        avg_logprob    = getattr(seg, "avg_logprob", 0.0)
        if no_speech_prob > 0.6:
            tqdm.write(f" 🚫 Non-speech at {seg.start:.1f}s: "
                  f"no_speech_prob={no_speech_prob:.2f} -- dropped")
            continue
        if avg_logprob < -1.0:
            tqdm.write(f" 🚫 Low-confidence decode at {seg.start:.1f}s: "
                  f"avg_logprob={avg_logprob:.2f} -- dropped")
            continue

        words = [w.word.strip().lower() for w in seg.words if w.word.strip()]
        if not words:
            continue
        # PATCH: single-word or two-word segments are never hallucinations —
        # the repeat-ratio test fires at 100% / 50% on trivially short segs
        # (e.g. "Ie." or "Iawn.") and silently drops real content.
        if len(words) < 3:
            clean.append(seg)
            continue
        most_common_count = Counter(words).most_common(1)[0][1]
        if most_common_count / len(words) > HALLUCINATION_REPEAT_RATIO:
            # PATCH: tqdm.write(), not print() -- a plain print() here lands
            # mid-line whenever a tqdm bar (or yt-dlp's own progress meter)
            # currently owns the terminal line via \r, producing garbled
            # output like "...ETA 06:39 🚫 Hallucination at ...". tqdm.write()
            # clears the active bar's line first, prints cleanly, then
            # redraws the bar underneath.
            tqdm.write(f" 🚫 Hallucination at {seg.start:.1f}s: "
                  f"'{Counter(words).most_common(1)[0][0]}' "
                  f"repeated {most_common_count}/{len(words)} -- dropped")
            continue
        # PATCH: orthographic sweep -- reject segments where more than 30% of
        # tokens contain diacritic combinations impossible in Welsh (umlaut,
        # tilde, caron, cedilla, etc.). These are Whisper hallucinations
        # produced when forced into cy mode on non-Welsh or silent audio.
        implausible = [w for w in words if not _is_plausible_welsh_token(w)]
        if implausible and len(implausible) / len(words) > 0.30:
            tqdm.write(f" 🚫 Orthographic hallucination at {seg.start:.1f}s: "
                  f"implausible tokens {implausible} "
                  f"({len(implausible)}/{len(words)}) -- dropped")
            continue
        clean.append(seg)
    return clean


def deduplicate_overlapping_segments(segments, overlap_threshold=0.80):
    """
    Remove segments whose time span substantially overlaps with a previously
    accepted segment.

    This catches a specific faster-whisper pathology on CPU (and occasionally
    GPU) where the VAD misfires on quiet audio, long pauses, or background
    music and re-transcribes the same audio window multiple times, producing
    several separate segment objects that all cover roughly the same span.
    The repeat-ratio hallucination filter does not catch this because each
    individual segment contains the word only once -- the duplication is
    across segments, not within a single segment.

    Strategy: for each candidate segment, compute what fraction of its
    duration overlaps with the most recent accepted segment.  If either
    direction of overlap exceeds overlap_threshold (default 80%), the
    candidate is a duplicate -- keep whichever has the higher mean word
    confidence, discard the other.

    overlap_threshold=0.80 means: if 80%+ of segment A's duration falls
    inside segment B (or vice versa), they are the same audio window.
    """
    if not segments:
        return segments

    accepted = [segments[0]]
    for cand in segments[1:]:
        prev = accepted[-1]
        # Compute overlap duration
        overlap_start = max(cand.start, prev.start)
        overlap_end   = min(cand.end,   prev.end)
        overlap_dur   = max(0.0, overlap_end - overlap_start)

        cand_dur = max(cand.end - cand.start, 1e-6)
        prev_dur = max(prev.end - prev.start, 1e-6)

        overlap_ratio_cand = overlap_dur / cand_dur
        overlap_ratio_prev = overlap_dur / prev_dur

        if max(overlap_ratio_cand, overlap_ratio_prev) >= overlap_threshold:
            # Duplicate -- keep the one with higher mean word confidence
            cand_conf = (sum(w.probability for w in cand.words) / len(cand.words)
                         if cand.words else 0.0)
            prev_conf = (sum(w.probability for w in prev.words) / len(prev.words)
                         if prev.words else 0.0)
            if cand_conf > prev_conf:
                tqdm.write(f" 🔁 Duplicate segment at {cand.start:.1f}s–{cand.end:.1f}s "
                      f"(overlap {max(overlap_ratio_cand, overlap_ratio_prev):.0%} "
                      f"with prev) -- keeping higher-confidence version")
                accepted[-1] = cand
            else:
                tqdm.write(f" 🔁 Duplicate segment at {cand.start:.1f}s–{cand.end:.1f}s "
                      f"(overlap {max(overlap_ratio_cand, overlap_ratio_prev):.0%} "
                      f"with prev) -- dropped")
        else:
            accepted.append(cand)

    return accepted


# ========================= PRE-PROCESSING =========================
def expand_whisper_tokens(raw_words):
    """
    Expands Welsh contractions in Whisper's word stream so that our
    internal token list matches how Cysill and spaCy tokenize the same
    text. For example: "i'r" -> ["i", "'r"].

    The first sub-token inherits the original word's timestamp and
    confidence. Synthetic sub-tokens are flagged so they can be excluded
    from mutation analysis (they're grammatical clitics, not content words).

    PATCH: contracted "'n" (the colloquial spelling of the predicate/
    aspect particle "yn", e.g. "mae'n", "oedd'n") needs different handling
    from i'r/a'r/o'r. Two failure modes without this:
      1. A bare "'n" token (Whisper split it off on its own, e.g. "mae" +
         "'n" + "meddwl") normalizes to "n" (length 1) and gets silently
         dropped by corpus_ops.py's length>=2 filter before mutation
         analysis ever runs -- "mae" and "meddwl" end up directly
         adjacent, and "mae"'s own trigger definition wrongly claims
         "meddwl" as its target, even though grammatically "yn" is what
         governs "meddwl" (or, per the yn+verb-noun exemption above, that
         nothing should be evaluated there at all).
      2. A fused "X'n" token (e.g. Whisper keeps "mae'n" as ONE word)
         normalizes via the existing "'n"->"yn" replace in normalize_word
         into a broken merge ("maeyn") that matches no TRIGGERS key at
         all -- the row is silently never generated, undercounting
         evaluable "yn" occurrences.
    Both are fixed the same way as i'r/a'r/o'r already are: split into
    real, independently-evaluable sub-tokens, EXCEPT unlike 'r (which is
    just definite-article scaffolding for tagger alignment), "yn" here is
    the whole point -- it must NOT be marked synthetic, or it becomes
    invisible to trigger detection all over again.
    """
    expanded = []
    for w in raw_words:
        surface = w["word"].strip().lower().replace("\u2019", "'")
        if surface in WELSH_CONTRACTION_SPLITS:
            parts = WELSH_CONTRACTION_SPLITS[surface]
            for idx, part in enumerate(parts):
                is_last = idx == len(parts) - 1
                expanded.append({
                    **w,
                    "word":      part,
                    "synthetic": idx > 0,   # only first sub-token has real timestamp
                    # PATCH: only the last sub-token was actually followed
                    # by whatever punctuation the original whole word had;
                    # earlier sub-tokens are immediately followed by the
                    # next sub-token of the SAME original word, never a
                    # real boundary.
                    "_clause_boundary_after": w.get("_clause_boundary_after", False) if is_last else False,
                })
        elif surface == "'n":
            # Whisper already split it off as its own token -- just
            # canonicalize the spelling so it survives downstream and
            # matches TRIGGERS["yn"].
            expanded.append({**w, "word": "yn", "synthetic": False})
        elif surface.endswith("'n") and len(surface) > 2:
            # Fused "X'n" token (e.g. "mae'n") -- split into stem + "yn",
            # both real (non-synthetic) tokens. Same boundary-ownership
            # logic as above: only "yn" (the last sub-token) can have
            # actually been followed by trailing punctuation.
            stem = surface[:-2]
            expanded.append({**w, "word": stem, "synthetic": False,
                              "_clause_boundary_after": False})
            expanded.append({**w, "word": "yn", "synthetic": False})
        else:
            expanded.append({**w, "synthetic": False})
    return expanded


def preprocess_segment(seg, seg_id=None):
    """
    Single normalization pass on a Whisper segment before any tagger
    sees it. Returns a clean, expanded word list.

    Steps:
    1. Strip leading/trailing whitespace and edge punctuation from each word
    2. Expand Welsh contractions to match tagger tokenization
    3. Drop tokens that became empty after stripping

    The original text sent to Cysill and spaCy is NOT modified -- the
    taggers still receive the raw segment text. Only our internal word
    list is expanded, so alignment counts match what the taggers produce.

    PATCH: each word is stamped with "_seg_id" (the index of the Whisper
    segment it came from). This is internal bookkeeping only -- output
    rows are built via explicit key-selection dicts elsewhere, so this
    never leaks into any CSV column. It exists so chunk_words_for_pos()
    can refuse to merge words from two different segments into the same
    tagger chunk, even when they'd otherwise fit under the character
    budget. Without it, a chunk boundary could fall mid-sentence or --
    worse, in multi-speaker audio -- splice the tail of one segment onto
    the head of the next with no signal in the text that a boundary
    (silence gap, possible speaker change) occurred there at all.
    """
    raw_words = []
    for w in seg.words:
        original = w.word.strip()
        surface = original.strip(".,!?;:\"()[]")
        if not surface:
            continue
        # PATCH: record whether this word was immediately followed by
        # clause/list-boundary punctuation in Whisper's own output,
        # BEFORE that punctuation gets stripped off below and lost
        # forever. Without this, "roi pob ffaith, pob myth" collapses to
        # a flat token sequence [..., "ffaith", "pob", "myth", ...] with
        # nothing distinguishing the comma-separated "pob myth" (a new,
        # unrelated list item) from a genuine adjacent word -- confirmed
        # live: Layer 1C (fem_noun+adjective) fired treating "ffaith" as
        # trigger and the NEXT list item's leading "pob" as its target
        # adjective, purely because "pob" happened to sit at
        # words_list[i+1] and got POS-tagged in a way that matched, with
        # no awareness a comma (i.e. a clause/list boundary, not a real
        # adjective relationship) separated them. Internal bookkeeping
        # only, same convention as _seg_id -- never leaks into any CSV
        # column, output rows are built from explicit key-selection
        # dicts elsewhere.
        clause_boundary_after = original.rstrip().endswith((",", ";", ":", "."))
        raw_words.append({
            "word":       surface,
            "start":      round(w.start, 3),
            "end":        round(w.end, 3),
            "confidence": round(w.probability, 4),
            "synthetic":  False,
            "_seg_id":    seg_id,
            "_clause_boundary_after": clause_boundary_after,
        })
    return expand_whisper_tokens(raw_words)


# ========================= GAP-TOLERANT ALIGNMENT =========================
def _tokens_match_fuzzy(norm_whisper, norm_tagger, tagger_token=None):
    """
    Three-stage fuzzy matching between a normalized Whisper word and a
    normalized tagger token.

    Stage 1 -- Exact match: "gi" == "gi"
    Stage 2 -- Substring: handles punctuation residue or partial overlaps
    Stage 3 -- Lemma fallback: "gi" matches lemma "ci" (soft mutation)
               Uses both spaCy's lemma field and the Techiaith lemma cache.

    This is the key function that handles the mutation mismatch problem --
    Whisper writes "gi" (mutated form) but spaCy's lemma is "ci" (radical).
    Without the lemma fallback, these would never match and alignment would
    drift for every mutated word in the segment.
    """
    if not norm_whisper or not norm_tagger:
        return False

    # Stage 1: exact surface match
    if norm_whisper == norm_tagger:
        return True

    # Stage 2: substring containment
    if norm_whisper in norm_tagger or norm_tagger in norm_whisper:
        return True

    # Stage 3: lemma fallback
    if tagger_token:
        # spaCy provides a lemma field directly
        spacy_lemma = normalize_word(tagger_token.get("lemma", "") or "")
        # Cysill doesn't -- but the Techiaith lemma cache may have it
        cached_lemma = normalize_word(LEMMA_CACHE.get(norm_whisper, "") or "")

        for lemma in filter(None, [spacy_lemma, cached_lemma]):
            if norm_whisper == lemma or norm_tagger == lemma:
                return True
            # Radical reconstruction: mutated form shares a suffix with radical
            # e.g. "nghi" vs "ci" -- last 2 chars match
            if len(norm_whisper) >= 2 and len(lemma) >= 2:
                if norm_whisper.endswith(lemma[-2:]) or lemma.endswith(norm_whisper[-2:]):
                    return True

    return False


def align_with_gap_tolerance(tagger_tokens, whisper_words, window=GAP_ALIGN_WINDOW):
    """
    Gap-tolerant alignment between a tagger token stream and a Whisper
    word list.

    Unlike simple positional alignment, this function can absorb
    mismatches without cascading the error to every subsequent token.
    When a match isn't found within `window` tokens, a None gap is
    inserted for that Whisper word and the algorithm re-anchors on the
    next clean match.

    This handles:
    - Contraction splits (Cysill produces 2 tokens, Whisper has 1)
    - Hallucinated words (Whisper token has no tagger correspondent)
    - Residual punctuation tokens that weren't stripped
    - Mutated forms vs radical lemmas (via _tokens_match_fuzzy stage 3)

    Returns: list of (whisper_word_dict, tagger_token_or_None) pairs
    """
    # Strip punctuation from tagger stream before alignment
    content_tokens = [t for t in (tagger_tokens or [])
                      if not t.get("is_punct", False)
                      and t.get("pos") not in ("PUNCT", "SPACE", "punct", "space", "EOS")]

    result = []
    t_idx  = 0

    for w in whisper_words:
        # Skip synthetic sub-tokens -- they were added by expand_whisper_tokens
        # to match the tagger's count but don't correspond to real audio words
        if w.get("synthetic"):
            # Try to find and consume the corresponding tagger token
            if t_idx < len(content_tokens):
                norm_t = normalize_word(
                    content_tokens[t_idx].get("text") or
                    content_tokens[t_idx].get("token", ""))
                norm_w = normalize_word(w["word"])
                if _tokens_match_fuzzy(norm_w, norm_t, content_tokens[t_idx]):
                    t_idx += 1
            result.append((w, None))   # synthetic tokens get no tagger data
            continue

        norm_w = normalize_word(w["word"])
        if not norm_w:
            result.append((w, None))
            continue

        # Search within lookahead window for a matching tagger token
        matched_at = None
        for offset in range(window):
            t_check = t_idx + offset
            if t_check >= len(content_tokens):
                break
            norm_t = normalize_word(
                content_tokens[t_check].get("text") or
                content_tokens[t_check].get("token", ""))
            if _tokens_match_fuzzy(norm_w, norm_t, content_tokens[t_check]):
                matched_at = t_check
                break

        if matched_at is None:
            # No match found -- whisper word has no tagger correspondent
            # (hallucination, or a word the tagger skipped)
            result.append((w, None))
        else:
            # Skip any tagger tokens before the match (they had no whisper correspondent)
            t_idx = matched_at + 1
            result.append((w, content_tokens[matched_at]))

    return result


# ========================= CYSILL API =========================
# Moved to cysill_client.py: _cysill_get, parse_pos_result_extended,
# fetch_pos_for_chunk, fetch_lemma, chunk_words_for_pos. Already
# available here via the `from cysill_client import *` above.


# ==================== ENRICH_WORDS CHECKPOINTING ====================
# Lives here, not its own module (unlike cysill_client.py/spacy_tagging.py,
# which wrap genuinely external systems) -- this is tightly coupled to
# chunk_words_for_pos()'s exact chunking and enrich_words()'s own merge
# logic, not a separable concern. A small atomic-JSON-write helper is
# duplicated from corpus_ops.py's _write_json() rather than imported --
# corpus_ops.py imports FROM mutation_engine.py, so the reverse import
# would be circular. Deliberate, acknowledged duplication of ~4 stable
# lines, not an oversight (see corpus_ops.py's _write_json for the twin).

def _write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, ensure_ascii=False,
                                    default=lambda o: float(o) if hasattr(o, "__float__") else str(o)),
                         encoding="utf-8")
    tmp_path.replace(path)


def _checkpoint_fingerprint(all_preprocessed_words):
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


def _checkpoint_path_for(checkpoint_key):
    """
    Filename is a hash of checkpoint_key (video url/id/audio path -- may
    contain characters that aren't filesystem-safe on every OS this
    project runs on, per the Windows-path issues already hit elsewhere in
    this project) rather than a slugified version of it. The original key
    is still stored INSIDE the checkpoint JSON for anyone grepping
    CHECKPOINT_DIR by hand to figure out which file belongs to which
    video, without needing the filename itself to be readable.
    """
    digest = hashlib.sha256(checkpoint_key.encode("utf-8")).hexdigest()[:24]
    return CHECKPOINT_DIR / f"{digest}.json"


def _load_enrich_checkpoint(checkpoint_path, checkpoint_key, expected_fingerprint,
                            chunk_word_counts):
    """
    Returns {"completed_chunk_count": int, "enriched_words": [...]} if a
    valid, matching checkpoint exists, else None. "Valid" means: the file
    parses, its fingerprint matches the CURRENT input words, its completed
    chunk count is in range, and its saved-word count exactly matches that
    completed chunk prefix. Any mismatch
    is treated as "this checkpoint doesn't apply anymore" and logged, not
    silently discarded -- so a fingerprint mismatch (see
    _checkpoint_fingerprint()'s docstring) is visible instead of just
    quietly re-tagging everything with no explanation of why the resume
    didn't happen.
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


def _save_enrich_checkpoint(checkpoint_path, checkpoint_key, fingerprint,
                             total_chunks, completed_chunk_count, enriched_words):
    _write_json_atomic(checkpoint_path, {
        "source_key":            checkpoint_key,
        "fingerprint":           fingerprint,
        "total_chunks":          total_chunks,
        "completed_chunk_count": completed_chunk_count,
        "enriched_words":        enriched_words,
    })


def _delete_enrich_checkpoint(checkpoint_path):
    checkpoint_path.unlink(missing_ok=True)


def enrich_words(all_preprocessed_words, checkpoint_key=None):
    """
    Master enrichment function. Takes the full flat list of preprocessed
    words across all segments, chunks them, calls both Cysill and spaCy
    on each chunk, and returns a flat list of word dicts with all
    tagger fields merged in.

    Both taggers use the SAME chunk boundaries and SAME gap-tolerant
    alignment, so their fields are directly comparable per word.

    PATCH: alignment length assertions guard against silent mismatches
    where Cysill failure returns [] while spaCy succeeds, causing the
    zip to truncate results without warning.

    checkpoint_key: a stable per-video identifier (analyze() passes
    video url/id, or the audio path as a last resort) -- when given,
    progress is saved to CHECKPOINT_DIR after every completed chunk, so a
    kill/crash/interrupt partway through this video's tagging can resume
    from the last completed chunk on a future run instead of re-tagging
    (and re-hitting Cysill's rate limit for) the whole video again. This
    is exactly the "8 hours, zero output" failure mode: previously this
    entire function had to return before ANYTHING was written to disk for
    the video, so one stuck chunk anywhere meant losing every chunk's
    work, not just that one. None (the default) disables checkpointing
    entirely -- used by analyze_phrase()'s ad hoc test-phrase path, where
    there's nothing worth resuming.
    """
    chunks   = chunk_words_for_pos(all_preprocessed_words)
    enriched = []
    start_chunk_idx = 0

    checkpoint_path = None
    if checkpoint_key:
        fingerprint     = _checkpoint_fingerprint(all_preprocessed_words)
        checkpoint_path = _checkpoint_path_for(checkpoint_key)
        loaded = _load_enrich_checkpoint(
            checkpoint_path, checkpoint_key, fingerprint,
            [len(chunk) for chunk in chunks],
        )
        if loaded is not None and loaded["completed_chunk_count"] > 0:
            enriched        = loaded["enriched_words"]
            start_chunk_idx = loaded["completed_chunk_count"]
            tqdm.write(f" ⏩ Resuming from checkpoint: {start_chunk_idx}/{len(chunks)} "
                       f"chunk(s) already tagged from a previous interrupted run "
                       f"-- skipping straight to chunk {start_chunk_idx + 1}.")

    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx < start_chunk_idx:
            continue
        chunk_text = " ".join(w["word"] for w in chunk)

        # PATCH: whole-chunk skip. If EVERY word in this chunk is either
        # (a) a single unambiguous reading in the offline Bangor lexicon,
        # or (b) recognized as an English code-switch word by
        # is_english_code_switch() -- the SAME function get_welsh_lemma()
        # already uses to avoid Welshifying English lemmas, just not
        # previously consulted anywhere in THIS chunking path -- the
        # entire chunk is resolved locally and fetch_pos_for_chunk() is
        # never called for it.
        #
        # The English-code-switch half of this matters on its own, not
        # just for call-volume: confirmed live (real Cysill traffic) that
        # plain English words like "Hunger Games" were being sent to
        # Cysill's POS endpoint with no gate at all -- burning rate-limit
        # budget on words Cysill can't meaningfully tag as Welsh AND
        # risking the same "Welshified" mis-tag failure mode already
        # observed for get_welsh_lemma() (English "the"/"for" coming back
        # as bogus Welsh lemmas "te"/"mor"). A code-switch word gets
        # every cysill_* field left None here -- deliberately NOT sent to
        # Cysill at all, not even nominally, and NOT run through the
        # Bangor lexicon either (that lexicon is Welsh-only; looking an
        # English word up in it would just be a different flavour of the
        # same mistake).
        #
        # Still all-or-nothing PER CHUNK, not per word: chunks are
        # already segment-bounded (chunk_words_for_pos() breaks at
        # _seg_id changes), so skipping only SOME words out of a chunk's
        # text before sending the rest to Cysill would hand the live
        # tagger a span with words silently missing -- degrading
        # Cysill's own accuracy on the Welsh words that still need it,
        # since its tagging can use surrounding context. Requiring the
        # WHOLE chunk to be resolved (by lexicon or code-switch status)
        # avoids that: a chunk still reaches Cysill with full, untouched
        # context whenever it contains even one genuinely ambiguous or
        # OOV Welsh word.
        #
        # BUGFIX: is_english_code_switch() used to only get consulted
        # inside `if bangor_lexicon.is_loaded():` -- meaning on any run
        # where the lexicon file wasn't present (missing download, wrong
        # BANGOR_LEXICON_PATH, a machine that just never had it placed),
        # the English-code-switch skip didn't fire AT ALL, for any word,
        # lexicon or no lexicon. Confirmed live: chunks that were pure
        # English ("Hunger Games") were still going straight to Cysill's
        # POS endpoint on a run where the lexicon genuinely was loaded,
        # which traced back to this coupling -- the two checks have no
        # real dependency on each other (is_english_code_switch() never
        # touches bangor_lexicon at all), so tying English resolution to
        # the lexicon's load state was accidental, not intentional.
        # Restructured so each word tries is_english_code_switch() FIRST,
        # unconditionally, and only falls through to the lexicon lookup
        # (still gated on is_loaded(), since THAT part genuinely does
        # need the file) for words that aren't English. A chunk that's
        # entirely English code-switching now skips Cysill even with no
        # lexicon loaded at all; a chunk with genuine Welsh words still
        # needs the lexicon (or Cysill) for those.
        candidate_tags = []
        for w in chunk:
            norm = normalize_word(w["word"])
            if is_english_code_switch(w["word"], None):
                candidate_tags.append(
                    {"mutation_type": None, "gender": None, "number": None})
            elif bangor_lexicon.is_loaded():
                candidate_tags.append(
                    bangor_lexicon.resolved_tag_if_unambiguous(norm))
            else:
                candidate_tags.append(None)
        lexicon_tags = candidate_tags if all(t is not None for t in candidate_tags) else None

        if lexicon_tags is not None:
            # Every word resolved locally (lexicon hit or recognized
            # English) -- skip the live call entirely. cysill_pos is
            # intentionally left None (see bangor_lexicon.py's module
            # docstring for why -- different tag scheme, no safe mapping
            # -- and for code-switch words, because Cysill has no
            # meaningful Welsh POS to give an English word anyway), same
            # value it would already have if Cysill had simply failed or
            # been disabled for this word; every downstream consumer
            # already tolerates that by design.
            cysill_aligned = [
                (w, {"pos": None,
                     "tagger_mutation_type": tag["mutation_type"],
                     "gender": tag["gender"],
                     "number": tag["number"]})
                for w, tag in zip(chunk, lexicon_tags)
            ]
        else:
            # ---- Cysill (unchanged path) ----
            # PATCH-NOT-TAKEN: an earlier draft of this also opportunistically
            # filled individual words from the lexicon here, whenever
            # Cysill's OWN per-word alignment came back None for just
            # that word (not the whole chunk). Deliberately cut before
            # shipping: cysill_aligned (below) becomes True for a
            # non-None cysill_tok regardless of WHERE that token came
            # from, and corpus_analyzer.py's tagger-agreement / corroboration
            # bucketing (all_three / cysill+heuristic / disputed) reads
            # cysill_aligned as "Cysill genuinely corroborated this row" --
            # exactly the kind of two-fields-that-look-comparable-but-
            # aren't mismatch this project has already been bitten by
            # (tagger_agreement, DOM/Vnoun conflation). A per-word fallback
            # with no matching provenance flag would silently inflate
            # those corroboration numbers. The whole-chunk skip above
            # avoids this because it has its own locally_resolved flag
            # (see below) -- but that flag still needs corpus_analyzer.py
            # updated to actually branch on it before treating a
            # lexicon-resolved row as "cysill corroborated" for Kappa/
            # bucket purposes. Worth doing as a deliberate follow-up, not
            # smuggled in here as a side effect.
            cysill_tokens  = fetch_pos_for_chunk(chunk_text) or []
            cysill_aligned = align_with_gap_tolerance(cysill_tokens, chunk)

        # ---- spaCy ----
        # PATCH: was `if SPACY_NLP else []` -- SPACY_NLP is now module-level
        # state that lives in spacy_tagging.py and gets set by load_spacy().
        # A plain `from spacy_tagging import SPACY_NLP` copies its value at
        # import time (None, before load_spacy() has run) and never updates
        # again, so that check would always see "not loaded" even after a
        # successful load. parse_spacy_doc() already checks its own
        # module's SPACY_NLP internally and returns None when unset, so
        # just rely on that instead of re-checking it here.
        spacy_raw     = parse_spacy_doc(chunk_text) or []
        spacy_aligned = align_with_gap_tolerance(spacy_raw or [], chunk)

        # PATCH: both aligners must return one pair per input word
        if len(cysill_aligned) != len(chunk) or len(spacy_aligned) != len(chunk):
            tqdm.write(f" ⚠️ Alignment length mismatch in chunk "
                  f"(cysill={len(cysill_aligned)}, spacy={len(spacy_aligned)}, "
                  f"chunk={len(chunk)}) -- padding with Nones")
            # Pad the shorter list rather than crashing; data loss is preferable
            # to losing the whole video
            while len(cysill_aligned) < len(chunk):
                cysill_aligned.append((chunk[len(cysill_aligned)], None))
            while len(spacy_aligned) < len(chunk):
                spacy_aligned.append((chunk[len(spacy_aligned)], None))
            # An aligner should never emit extras, but normalize that case
            # explicitly instead of leaving zip() to hide the mismatch.
            cysill_aligned = cysill_aligned[:len(chunk)]
            spacy_aligned = spacy_aligned[:len(chunk)]

        # ---- Merge ----
        for (w, cysill_tok), (_, spacy_tok) in zip(cysill_aligned, spacy_aligned):
            entry = dict(w)
            # Cysill fields
            entry["cysill_pos"]           = cysill_tok["pos"] if cysill_tok else None
            entry["cysill_mutation_type"] = cysill_tok["tagger_mutation_type"] if cysill_tok else None
            entry["cysill_gender"]        = cysill_tok["gender"] if cysill_tok else None
            # PATCH: best-effort only -- unlike "gender", "number" isn't a
            # field this codebase has confirmed exists in Cysill's response
            # schema, so this uses .get() defensively. If it's not present,
            # this is just None and the spaCy-side check (confirmed via
            # UD morph features) carries the plural gate on its own.
            entry["cysill_number"]        = cysill_tok.get("number") if cysill_tok else None
            # NOTE: True for lexicon-resolved rows too (cysill_tok is the
            # synthetic dict built above, not None) -- check
            # entry["locally_resolved"] alongside this wherever "did the
            # live Cysill API actually corroborate this row" matters
            # (corroboration buckets, tagger-agreement Kappa), not this
            # flag alone.
            entry["cysill_aligned"]       = cysill_tok is not None
            # PATCH: True only when this chunk was skipped wholesale --
            # either via the Bangor lexicon or via recognized English
            # code-switch words (lexicon_tags is not None, set once for
            # the whole chunk above) -- lets any downstream analysis (or
            # a future audit) tell "Cysill answered this" apart from
            # "this was resolved locally instead", since those sources
            # have different failure characteristics even though they
            # share the same output vocabulary.
            entry["locally_resolved"]     = lexicon_tags is not None
            # spaCy fields
            entry["spacy_token"]          = spacy_tok  # full dict or None
            entry["spacy_aligned"]        = spacy_tok is not None
            # Unified gender: spaCy primary, Cysill fallback
            entry["gender"]               = (extract_gender_from_spacy(spacy_tok)
                                              or entry["cysill_gender"])
            # Collision check
            entry["collision_note"]       = KNOWN_HOMOGRAPH_COLLISIONS.get(
                normalize_word(w["word"]))
            enriched.append(entry)

        if checkpoint_path is not None:
            _save_enrich_checkpoint(checkpoint_path, checkpoint_key, fingerprint,
                                     len(chunks), chunk_idx + 1, enriched)

    if checkpoint_path is not None:
        # Every chunk made it through -- the checkpoint has done its job for
        # this video, and leaving it on disk would just mean a *future*,
        # unrelated interrupted run either wastefully overwriting it chunk
        # by chunk from scratch, or (if this exact video is ever queued
        # again for some reason) misleadingly claiming to resume a run
        # that already fully completed.
        _delete_enrich_checkpoint(checkpoint_path)

    return enriched


# ========================= CONFIDENCE SCORING =========================
def compute_confidence(status, is_erosion, spacy_tok, cysill_mutation_type,
                        heuristic_expected, heuristic_surface,
                        cysill_genuinely_aligned=True):
    """
    PATCH: cysill_genuinely_aligned distinguishes "Cysill itself actually
    answered for this word" from "cysill_mutation_type happens to be
    populated". The latter is ALSO true for Bangor-lexicon-resolved words
    (see enrich_words()'s locally_resolved flag) -- a real, useful value,
    just not an independent second tagger opinion. Confirmed live: 6 rows
    in one run were stamped "all_three"/"cysill+heuristic" off a lexicon
    hit with cysill_pos completely empty, before this parameter existed --
    a lexicon-plus-heuristic agreement counted as three-way corroboration.

    IMPORTANT: this gates the two boolean checks below (cysill_c,
    cysill_rad), it does NOT null out cysill_mutation_type itself before
    use. That distinction matters: in the erosion branch, `cysill_rad =
    cysill_mutation_type is None` treats "no mutation reported" as
    Cysill POSITIVELY confirming a radical form -- real corroboration
    evidence. Nulling the raw value for a not-genuinely-aligned row would
    make `is None` trivially true and credit EVERY locally-resolved or
    never-consulted row with false "Cysill confirmed radical" evidence --
    the exact opposite of what this parameter exists to prevent. Gating
    the boolean directly (`cysill_genuinely_aligned and cysill_mutation_type
    is None`) avoids that trap: not-genuinely-aligned means no Cysill
    evidence either way, not evidence for radical.

    Defaults to True so any call site that hasn't been updated keeps its
    prior behavior rather than silently losing all Cysill credit; both
    current call sites (_build_row(), _evaluate_h_mutation()) pass it
    explicitly.
    """
    has_parser      = spacy_tok is not None
    parser_mut      = spacy_tok["mutation"] if has_parser else None
    parser_mut_type = SPACY_MUTATION_MAP.get(parser_mut) if parser_mut else None

    if not is_erosion:
        parser_c   = parser_mut_type and (parser_mut_type == heuristic_surface or
                      parser_mut_type in (heuristic_expected or []))
        cysill_c   = cysill_genuinely_aligned and cysill_mutation_type and (
                      cysill_mutation_type == heuristic_surface or
                      cysill_mutation_type in (heuristic_expected or []))
        heuristic_c = heuristic_surface is not None
        if parser_c and cysill_c and heuristic_c: return 1.00, "all_three"
        if parser_c and cysill_c:                 return 0.95, "parser+cysill"
        if parser_c and heuristic_c:              return 0.80, "parser+heuristic"
        if cysill_c and heuristic_c:              return 0.65, "cysill+heuristic"
        if parser_c:                               return 0.70, "parser_only"
        if cysill_c:                               return 0.50, "cysill_only"
        return 0.30, "heuristic_only"
    else:
        parser_ctx = has_parser and spacy_tok.get("dep") in {
            "obj", "iobj", "nsubj", "obl", "vocative", "amod", "nmod", "xcomp"}
        cysill_rad = cysill_genuinely_aligned and (cysill_mutation_type is None)
        heuristic_f = heuristic_surface is None and heuristic_expected is not None
        if parser_ctx and cysill_rad and heuristic_f: return 0.90, "parser+cysill+heuristic"
        if cysill_rad and heuristic_f:                return 0.75, "cysill+heuristic"
        if heuristic_f:                               return 0.50, "heuristic_only"
        return 0.10, "disputed"


# ========================= TRIGGER DETECTION =========================
def _resolve_mixed_mutation_expected(target_raw):
    cluster = initial_cluster(target_raw)
    return ["aspirate"] if cluster in ASPIRATE_INITIALS else ["soft"]

def _resolve_feminine_ei_expected(target_raw):
    cluster = initial_cluster(target_raw)
    if not cluster:
        return None
    if cluster in WELSH_VOWELS:
        return ["h-mutation"]
    elif cluster in ASPIRATE_INITIALS:
        return ["aspirate"]
    return None

def layer_1_trigger_detection(trigger_word, cysill_pos=None,
                               cysill_gender=None, spacy_token=None):
    trigger       = normalize_word(trigger_word)
    base_expected = TRIGGERS.get(trigger)
    if not base_expected:
        return {"trigger_detected": False, "expected_mutation": None,
                "trigger_word": trigger, "fem_ei": False,
                "mixed": False, "h_mutation": False}

    expected_list = base_expected.split("|") if "|" in base_expected else [base_expected]
    fem_ei = False
    mixed  = False

    # PATCH: "oes" homograph guard. TRIGGERS maps "oes" unconditionally to
    # the copula sense ("is/are there...?", soft-mutation trigger on the
    # following predicate), but "oes" is also a genuine feminine noun
    # meaning "age/era" (e.g. "oes aur" - golden age, "oes Fictoria" -
    # Victorian era). In that noun sense "oes" isn't a mutation trigger at
    # all -- treating every occurrence as copula risked wrongly firing on
    # genitive noun+noun constructions headed by the noun "oes". The token's
    # own POS tag (not the following word's) distinguishes the two senses:
    # a noun tag here means this is "age/era", not the copula. This is
    # separate from the unrelated `oes` COMPLEMENT question of whether
    # "oes"-as-copula's own following word is itself the subject
    # (see BOD_SUBJECT_EXEMPT_TRIGGERS).
    if trigger == "oes":
        pos = cysill_pos or ""
        spacy_pos = spacy_token.get("pos") if spacy_token else None
        is_noun_sense = pos.split("+")[0].startswith("N") or spacy_pos in ("NOUN", "PROPN")
        if is_noun_sense:
            return {"trigger_detected": False, "expected_mutation": None,
                    "trigger_word": trigger, "fem_ei": False,
                    "mixed": False, "h_mutation": False}

    if trigger == "yn":
        pos       = cysill_pos or ""
        spacy_dep = spacy_token.get("dep") if spacy_token else None
        if any(pos.startswith(p) for p in PREP_TAG_PREFIXES) or spacy_dep == "case":
            expected_list = ["nasal"]
        elif pos in PREDYN_TAGS or any(x in pos for x in ("VERBADJ", "VB", "ADV")) \
                or spacy_dep in ("aux", "mark"):
            expected_list = ["soft_limited"]

    if trigger == "a":
        pos       = cysill_pos or ""
        spacy_dep = spacy_token.get("dep") if spacy_token else None
        if any(t in pos for t in REL_INT_TAGS) or "PRONREL" in pos:
            expected_list = ["soft"]
        elif "CONJ" in pos or spacy_dep in ("cc",):
            expected_list = ["aspirate"]
        elif spacy_dep == "nsubj":
            expected_list = ["soft"]

    if trigger == "ei":
        gender = (extract_gender_from_spacy(spacy_token)
                  if spacy_token else None) or cysill_gender
        if gender == "feminine":
            fem_ei = True
            expected_list = ["fem_ei_pending"]
        elif gender == "masculine":
            expected_list = ["soft"]

    if trigger in MIXED_MUTATION_TRIGGERS:
        mixed = True
        expected_list = ["mixed_pending"]

    return {
        "trigger_detected": True, "expected_mutation": expected_list,
        "trigger_word": trigger, "fem_ei": fem_ei, "mixed": mixed,
        "h_mutation": expected_list == ["h-mutation"],
    }


# ========================= WELSH WORD PLAUSIBILITY =========================
# Bigrams that cannot occur in Welsh orthography.  These catch Whisper
# hallucinations that pass the diacritic filter (no illegal marks) but are
# orthographically impossible in Welsh -- e.g. "gerioed" (impossible: Welsh
# has "erioed"), "bennol" (impossible: "penodol"), "xw", "qy" etc.
# The list is conservative: only bigrams with zero attestation in standard
# Welsh orthography are included.  Do NOT add bigrams that appear in
# loanwords or code-switched English.
_IMPOSSIBLE_WELSH_BIGRAMS = frozenset({
    # Consonant clusters impossible in Welsh onset/nucleus position
    "xw", "xr", "xb", "xd", "xf", "xg", "xl", "xm", "xn", "xp", "xs", "xt",
    "qy", "qw", "qa", "qe", "qi", "qo", "qu",
    # PATCH: consonant+h clusters that are not legal Welsh digraphs and do
    # not correspond to any real mutation output. The only legal C+h
    # clusters in Welsh are ch/ph/th/rh (their own digraphs) and mh/nh/ngh
    # (nasal mutation of p/t/c). Aspirate mutation ONLY applies to p/t/c
    # (-> ph/th/ch) -- there is no aspirate mutation of b, d, g, f, or s,
    # so a word-initial "bh" (etc.) has no possible linguistic origin.
    # Previously absent from this list entirely, and also too short (2
    # consonants) to trip the separate consecutive-consonant-length check
    # below (threshold is >3) -- so a hallucinated "bh"-initial token
    # passed both hallucination-plausibility filters undetected. Confirmed
    # live: word-initial "bh" observed in output that should have been
    # rejected at this stage.
    "bh", "dh", "fh", "gh", "sh",
    # Vowel sequences impossible in Welsh (Welsh has ae, ai, au, aw, ei, eu,
    # ew, oe, oi, ou, wy, yw but NOT these):
    "uu", "ii", "aa",
    # Triple-consonant clusters that cannot occur in Welsh
    "nng", "mmh", "llf", "llg", "llb", "lld", "llm", "lln", "llp", "lls",
    "rhr", "rhl", "rhm", "rhn", "rhb", "rhd",
})

# Hoisted to module level to avoid rebuilding on every token call
_WELSH_VOWELS_CHECK    = frozenset("aeiouwyâêîôûŵŷ")
_KNOWN_WELSH_CLUSTERS  = frozenset({
    "mh", "nh", "ng", "ll", "rh", "ch", "ph", "th", "ff", "ngh"
})

def _is_plausible_welsh_token(word: str) -> bool:
    """
    Returns False if the token contains a (base vowel, combining mark)
    pair that cannot occur in genuine Welsh orthography.

    Strategy: NFD-decompose each character into its base letter plus any
    combining mark(s). Each (base, mark) pair is checked against
    _LEGAL_BASE_MARK_PAIRS -- deliberately more specific than "is this
    mark ever legal in Welsh", since some marks (diaeresis in particular)
    are legal on some vowels and not others. Circumflex is legal on any
    of the seven long vowels; acute/grave are tolerated on vowels in
    loanwords/poetry; diaeresis is legal ONLY on i/e, where it marks
    hiatus (a vowel forming its own syllable rather than gliding into a
    neighbour as part of a diphthong).

    Catches hallucinations such as:
        töii   (diaeresis-o, U+0308 on 'o' -- not a legal pair)
        tõii   (tilde-o,     U+0303 -- not in the legal set at all)
        děkuji (caron-e,     U+030C -- not in the legal set at all)
        ţară   (cedilla-t,   U+0326 -- not in the legal set at all)

    Does NOT reject:
        tŷ       (circumflex-y -- legal Welsh)
        llên     (circumflex-e -- legal Welsh)
        café     (acute-e     -- tolerated loanword)
        cwmnïau  (diaeresis-i -- legal Welsh, hiatus-marking)

    # PATCH (base-vowel-aware diacritic check): the previous version of
    # this function checked mark identity alone (any diaeresis anywhere
    # = illegal), which meant it correctly rejected "töii" but ALSO
    # incorrectly rejected genuine Welsh words carrying a native ï/ë,
    # such as "cwmnïau" -- confirmed live: a real, well-formed segment
    # logged as an "Orthographic hallucination" and dropped, purely
    # because it happened to contain a diaeresis at all rather than a
    # diaeresis on a vowel where Welsh doesn't use one. Fixed by
    # checking (base, mark) pairs instead of marks alone -- see
    # _LEGAL_BASE_MARK_PAIRS in mutation_tables.py for the full table
    # and reasoning.

    # PATCH: this function (and its supporting table, now
    # _LEGAL_BASE_MARK_PAIRS) went missing entirely during the module
    # split -- dropped along with the mutation-tables block it happened
    # to sit next to in the original file, leaving three call sites
    # silently calling an undefined name. Caught by pyflakes, not by the
    # earlier import smoke test, since a NameError here only fires the
    # first time a caller actually executes this code path at runtime.
    # Restored here; its table lives in mutation_tables.py since it's
    # pure data.
    """
    for ch in unicodedata.normalize("NFC", word.lower()):
        nfd = unicodedata.normalize("NFD", ch)
        if len(nfd) == 1:
            continue  # plain character, no combining marks
        base = nfd[0]
        for mark in nfd[1:]:
            if (base, mark) not in _LEGAL_BASE_MARK_PAIRS:
                return False
    return True

def _is_plausible_welsh_word(word: str) -> bool:
    """
    Returns False if the word contains a bigram that is orthographically
    impossible in Welsh.  Used as a second-layer hallucination filter after
    _is_plausible_welsh_token (which only checks diacritics).

    This catches forms like:
        gerioed  -- 'ger' + 'ioed' is phonotactically impossible; real word
                    is 'erioed' (ever).  Whisper prepends a ghost consonant.
        bennol   -- 'nn' followed by 'ol' does not occur; real word is
                    'penodol' (specific/particular).

    Does NOT reject:
        llaeth   -- 'll' is a legal Welsh digraph
        ngh      -- legal nasal mutation cluster
        mhob     -- legal nasal mutation cluster
    """
    w = unicodedata.normalize("NFC", word.lower())
    for i in range(len(w) - 1):
        if w[i:i+2] in _IMPOSSIBLE_WELSH_BIGRAMS:
            return False
    # Collapse known digraphs/trigraphs to single markers before counting
    # consecutive consonants.  More than 3 in a row is almost certainly a
    # hallucination -- Welsh allows mh/nh/ngh/ll/rh/ch/ph/th/ff/ng but
    # nothing beyond those clusters.
    collapsed = w
    for cluster in sorted(_KNOWN_WELSH_CLUSTERS, key=len, reverse=True):
        collapsed = collapsed.replace(cluster, "V")
    consecutive = 0
    for ch in collapsed:
        if ch in _WELSH_VOWELS_CHECK or ch == "V":
            consecutive = 0
        else:
            consecutive += 1
            if consecutive > 3:
                return False
    return True


def layer_2_lemma_analysis(following_word, cysill_pos=None, spacy_token=None):
    raw   = normalize_word(following_word)
    lemma = get_welsh_lemma(raw)
    cluster = initial_cluster(raw)

    # Check high-confidence English before Welsh plausibility filters. Otherwise
    # English tokens can be mislabelled as hallucinations or Welshified lemmas.
    if is_english_code_switch(raw, lemma):
        return {"raw_word": raw, "lemma": lemma, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "English Code-Switch", "is_code_switch": True}

    # PATCH: reject orthographically impossible tokens (Whisper hallucinations
    # with foreign diacritics -- umlaut, tilde, caron, etc.)
    if not _is_plausible_welsh_token(raw):
        return {"raw_word": raw, "lemma": None, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "Implausible-Orthography", "is_code_switch": False}

    # PATCH: reject phonotactically impossible Welsh words (Whisper mishears
    # that pass the diacritic filter but contain bigrams/clusters impossible
    # in Welsh orthography -- e.g. "gerioed" for "erioed", "bennol" for
    # "penodol").  These cause lemmatizer failures and false mutation counts.
    if not _is_plausible_welsh_word(raw):
        return {"raw_word": raw, "lemma": None, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "Impossible-Welsh-Phonotactics", "is_code_switch": False}

    # PATCH: suppress all surface forms of 'bod' as mutation targets.
    # These are either suppletive (mae, yw, oedd) or their mutation morphology
    # is not triggered by the preceding trigger word (fydd, fu). Analysing them
    # as targets produces systematic false erosion counts.
    if raw in BOD_SURFACE_FORMS:
        return {"raw_word": raw, "lemma": lemma, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "Bod-Inflection", "is_code_switch": False}

    # PATCH: if BOTH taggers failed to tag this token (cysill_pos unknown/None
    # AND spaCy has no token), the heuristic alone is too noisy to classify.
    # Suppress rather than risk a false erosion count.
    cysill_unknown = (not cysill_pos) or cysill_pos in ("?", "none", "n/a", "")
    spacy_absent   = spacy_token is None
    if cysill_unknown and spacy_absent:
        return {"raw_word": raw, "lemma": lemma, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "Both-Taggers-Failed", "is_code_switch": False}

    base_form = normalize_word(lemma if lemma else raw)
    if base_form and base_form[0] in WELSH_VOWELS:
        return {"raw_word": raw, "lemma": lemma, "cluster": cluster,
                "surface_mutation": None, "colloquial_mutation": None,
                "skip_reason": "Vowel-Initial", "is_code_switch": False}

    surface_mut = mutation_type_from_surface(raw)
    if lemma and lemma.startswith("g") and not raw.startswith("g"):
        if raw == lemma[1:]:
            surface_mut = "soft"

    colloquial_mut = None
    if raw.startswith("j") and len(raw) > 1:
        colloquial_mut = "colloquial_affricate"

    return {"raw_word": raw, "lemma": lemma, "cluster": cluster,
            "surface_mutation": surface_mut, "colloquial_mutation": colloquial_mut,
            "skip_reason": None, "is_code_switch": False}

def expected_surface_forms(radical, expected_mutations):
    radical = normalize_word(radical)
    if not radical or not expected_mutations:
        return set()

    forms = set()
    for mut_type in expected_mutations:
        mapping = RADICAL_TO_MUTATED.get(mut_type, {})
        for source, replacement in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
            if source and radical.startswith(source):
                forms.add(replacement + radical[len(source):])
                break

    return {f for f in forms if f}

def reverse_mutation_candidates(surface, expected_mutations=None):
    surface = normalize_word(surface)
    if not surface:
        return set()

    allowed = set(expected_mutations or RADICAL_TO_MUTATED.keys())
    candidates = set()
    # PATCH: the surface's true atomic initial unit, digraph-aware.
    surface_cluster = initial_cluster(surface)

    for mut_type, mapping in RADICAL_TO_MUTATED.items():
        if mut_type not in allowed:
            continue
        for radical_initial, mutated_initial in sorted(
                mapping.items(), key=lambda x: len(x[1]), reverse=True):
            if mutated_initial and surface.startswith(mutated_initial):
                # PATCH: `surface.startswith(mutated_initial)` alone is a
                # naive string-prefix check that doesn't know Welsh
                # digraphs (ff, dd, ll, rh, mh, ngh...) are atomic, single
                # sounds, not two separate letters. A short mutated value
                # like "f" (from b->f or m->f) will match any word that
                # merely *starts with the character* f -- including one
                # that actually begins with the unrelated, independently-
                # radical digraph "ff" (e.g. "ffenest", "ffrind", or a
                # Welsh-orthography English loanword like "fflamboyant"),
                # which never mutates from anything. The same failure
                # mode produces impossible clusters like "bh"/"gh": a
                # surface genuinely starting with "mh" (nasal mutation of
                # p) or "ngh" (nasal mutation of c) gets wrongly
                # re-matched against the unrelated single-letter values
                # "m" (b->m) / "ng" (g->ng). Requiring the surface's true
                # digraph-aware initial cluster to equal mutated_initial
                # exactly -- not just be string-prefixed by it -- fixes
                # all of these at once.
                if surface_cluster != mutated_initial:
                    continue
                candidates.add(radical_initial + surface[len(mutated_initial):])
            elif mutated_initial == "" and radical_initial == "g" and surface[0] in WELSH_VOWELS:
                candidates.add("g" + surface)

    if "h-mutation" in allowed and surface.startswith("h") and len(surface) > 1:
        if surface[1] in WELSH_VOWELS:
            candidates.add(surface[1:])

    return {c for c in candidates if c and c != surface}

def radical_candidates_for_target(target_node, t2, expected):
    candidates = []
    raw_word = normalize_word(t2.get("raw_word") or "")

    def add(value, verify=False):
        value = normalize_word(value or "")
        if not value or value in candidates:
            return
        if verify and value != raw_word:
            # PATCH: cross-validate a tagger-supplied lemma guess against
            # this pipeline's own forward mutation tables before trusting
            # it as a candidate radical. The previous fix here only
            # gated spaCy's lemma on content-POS tagging -- but POS and
            # lemma are separate model outputs that can fail
            # independently, and it never touched Cysill's own lemma
            # guess (t2.get("lemma")) at all, which had no gate
            # whatsoever. Confirmed live: "digalon" -> "igalon" and
            # "fi" -> "i", both cases of a tagger stripping a real
            # word-initial consonant as if it were a mutation artifact --
            # the same failure mode already documented for
            # "dibynol" -> "ibynol". The one thing we can actually verify
            # without guessing: does forward-mutating this candidate
            # under one of the mutation types genuinely expected in this
            # context reproduce the surface word we transcribed? If
            # forward-mutating "igalon" or "i" under soft/nasal/aspirate
            # never produces "digalon"/"fi" -- and it doesn't, since
            # those tables only take consonant-initial radicals, never
            # vowel-initial ones -- there is no rule in this pipeline
            # that explains the guess, and it's dropped rather than
            # polluting the candidate list.
            if raw_word not in expected_surface_forms(value, expected):
                return
        candidates.append(value)

    add(t2.get("lemma"), verify=True)
    spacy_tok = target_node.get("spacy_token")
    if spacy_tok:
        # Only trust spaCy's own lemma guess when spaCy actually tagged
        # this token with a content POS (not PUNCT/SPACE/X/absent) --
        # kept as a cheap first-pass filter, but no longer the only one;
        # see the verify=True cross-check in add() above.
        spacy_pos = spacy_tok.get("pos")
        spacy_content_tagged = spacy_pos not in (None, "", "PUNCT", "SPACE", "X", "unknown")
        if spacy_content_tagged:
            add(spacy_tok.get("lemma"), verify=True)
    for candidate in reverse_mutation_candidates(t2.get("raw_word"), expected):
        add(candidate)   # derived directly from this pipeline's own mutation tables -- always trustworthy
    add(t2.get("raw_word"))
    return candidates

def raw_has_radical_evidence(target_node, t2):
    """
    Returns True only when there is reliable external evidence that the raw
    surface form IS the radical (i.e. no mutation occurred).

    Evidence hierarchy:
    1. Cysill successfully tagged this token AND its lemma equals the surface
       form -- the tagger has seen the radical form directly.
    2. Cysill's lemma and the surface form share the same initial consonant
       cluster (digraph-aware) -- since Welsh mutation categorically only
       ever changes a word's INITIAL cluster, an unchanged initial cluster
       is proof of no mutation regardless of what the rest of the word
       does. This catches genuinely radical (unmutated) inflected forms
       that evidence tier 1 misses -- e.g. "pethau" (lemma "peth", plural
       -au suffix) or "plant" (lemma "plentyn", irregular plural) -- where
       the full surface form differs from the lemma for a completely
       unrelated reason (normal inflection) but the initial letter, the
       only thing mutation can touch, is identical. A genuinely mutated
       word's lemma (e.g. "gorff" -> lemma "corff") will NOT match on
       initial cluster, so this can't misclassify real erosion as radical.
       PATCH: previously this function only had tier 1, so any radical but
       inflected surface form (plurals, in particular) fell all the way
       through to the ambiguous "mutation_mismatch" bucket instead of
       being correctly recognized -- confirmed live against real pipeline
       output: 5/9 mutation_mismatch rows were exactly this pattern.

    The previous version included `raw == spacy_lemma` as evidence, but spaCy
    frequently echoes the surface form as its own lemma for unknown/rare words,
    making `raw == spacy_lemma` trivially true and causing false radical
    classifications (→ false erosion counts) whenever Cysill is down.
    """
    raw = normalize_word(t2.get("raw_word"))
    if not raw:
        return False

    # Evidence 1: Cysill-confirmed lemma match
    lemma = normalize_word(t2.get("lemma") or "")
    # PATCH: same fix as _build_row()'s cysill_has_signal -- cysill_aligned
    # is True for locally-resolved rows too, but this comment specifically
    # says "Cysill-confirmed", so it should mean Cysill, not a Bangor-
    # lexicon substitute answering in its place.
    if lemma and target_node.get("cysill_aligned") and not target_node.get("locally_resolved"):
        if raw == lemma:
            return True
        # Evidence 2: same initial cluster despite differing elsewhere
        if initial_cluster(raw) == initial_cluster(lemma):
            return True

    # A differing spaCy lemma is evidence for a possible mutated form
    # (e.g. gi -> ci), not evidence that the surface is radical.

    return False

def _surface_is_h_initial_radical(raw_word, expected):
    """
    Returns True if the surface word starts with 'h' AND the expected
    mutation type is soft, soft_limited, nasal, or aspirate.

    Under these triggers, 'h' is NEVER a mutated form of anything -- it
    cannot be produced by soft/nasal/aspirate mutation.  Therefore an
    h-initial surface form is always the radical (or an h-prothesis form,
    but h-prothesis only occurs after ei/ein/eu/u which use 'h-mutation'
    as their expected type, not soft/nasal/aspirate).

    This function exists to bypass raw_has_radical_evidence for h-initial
    loanwords and native words that the lemmatizer cannot confirm (hostel,
    heddlu, haul, etc.) -- the tagger failure is not evidence of mutation,
    it's evidence of an OOV item.

    IMPORTANT: this must NOT fire for expected=['h-mutation'] contexts
    (after ei/ein/eu/u) since there an h-initial form IS the mutated form.
    """
    h_mutation_expected = any(m == "h-mutation" for m in expected)
    if h_mutation_expected:
        return False
    non_h_mutations = {"soft", "soft_limited", "nasal", "aspirate"}
    if not any(m in non_h_mutations for m in expected):
        return False
    raw = normalize_word(raw_word)
    return bool(raw) and raw[0] == "h"


def _find_form_match(raw_word, expected, radical_candidates):
    raw = normalize_word(raw_word)
    for radical in radical_candidates:
        forms = expected_surface_forms(radical, expected)
        if raw in forms:
            return radical, forms
    return None, set()

def cysill_coarse_pos(pos_tag):
    if not pos_tag or pos_tag in ("?", "none", "n/a"):
        return "unknown"
    parts = re.split(r"[+|]", pos_tag)
    coarse = set()
    for part in parts:
        p = part.strip().upper()
        if not p:
            continue
        if p.startswith("VB") or p in {"V", "VERB", "VERBADJ"}:
            coarse.add("VERB")
        elif p.startswith("N") or p in {"PLACE", "PERSON"}:
            coarse.add("NOUN")
        elif p.startswith("ADJ") or p.startswith("ORD"):
            coarse.add("ADJ")
        elif p.startswith("ADV"):
            coarse.add("ADV")
        elif p.startswith("PRON"):
            coarse.add("PRON")
        elif p.startswith("PREP") or p.startswith("CPREP"):
            coarse.add("ADP")
        elif p.startswith("CONJ"):
            coarse.add("CONJ")
        elif p.startswith("CARD"):
            coarse.add("NUM")
        elif p in {"PREDYN", "PART", "EXCL"}:
            coarse.add("PART")
    return "|".join(sorted(coarse)) if coarse else "other"

def spacy_coarse_pos(spacy_tok):
    if not spacy_tok:
        return "unknown"
    pos = (spacy_tok.get("pos") or "").upper()
    if not pos:
        return "unknown"
    if pos in {"PROPN"}:
        return "NOUN"
    if pos in {"AUX"}:
        return "VERB"
    return pos

def pos_compatible(cysill_pos, spacy_tok):
    c_coarse = cysill_coarse_pos(cysill_pos)
    s_coarse = spacy_coarse_pos(spacy_tok)
    if "unknown" in (c_coarse, s_coarse):
        return None
    return bool(set(c_coarse.split("|")) & {s_coarse})


def pos_tagger_agreement_sufficient(cysill_pos, spacy_tok, require_erosion=False):
    """
    Returns True when there is enough tagger-level evidence to trust a
    mutation classification.

    For correct_mutation: one tagger is enough (we don't want to suppress
    genuine mutations just because one tagger is absent).

    For erosion specifically (require_erosion=True): we demand at least one
    tagger to have successfully tagged the token with a *content* POS. A
    token where both taggers returned unknown/absent is too ambiguous to
    call erosion -- it may be a hallucination, a proper noun, or an OOV
    item that happens to start with a mutable initial.

    Rules:
    - Cysill tagged with a real POS (not ?, none, n/a, unknown) → sufficient
    - spaCy tagged with any content POS (not PUNCT, SPACE, X, unknown) → sufficient
    - Both absent → insufficient (for erosion); sufficient (for non-erosion)
    """
    cysill_real = bool(
        cysill_pos and cysill_pos not in ("?", "none", "n/a", "", "unknown")
    )
    spacy_content = bool(
        spacy_tok and spacy_tok.get("pos") not in
        (None, "", "PUNCT", "SPACE", "X", "unknown")
    )
    if not require_erosion:
        return True   # non-erosion classification always allowed
    return cysill_real or spacy_content

def is_selectively_invariant(radical, expected_mutations):
    if not radical or not expected_mutations:
        return False
    radical_char = normalize_word(radical)[0] if normalize_word(radical) else ""
    for mut_type in expected_mutations:
        if mut_type in SELECTIVE_INVARIANT_MAP and \
                radical_char in SELECTIVE_INVARIANT_MAP[mut_type]:
            return True
    return False

def classify_register_adjustment(expected, surface_mut):
    if not COLLOQUIAL_REGISTER:
        return "standard"
    if surface_mut in DECLINING_MUTATIONS:
        return "colloquial_variant"
    if surface_mut is None and isinstance(expected, list) and \
            any(m in DECLINING_MUTATIONS for m in expected):
        return "colloquial_variant"
    return "standard"


# ========================= MUTATION EVALUATION =========================
def _evaluate_mutation_outcome(target_node, expected):
    t2 = layer_2_lemma_analysis(
        target_node["word"],
        cysill_pos=target_node.get("cysill_pos"),
        spacy_token=target_node.get("spacy_token"),
    )
    if t2.get("skip_reason"):
        return {"skip": True, "t2": t2}

    surface_mut    = t2["surface_mutation"]
    colloquial_mut = t2["colloquial_mutation"]
    register_class = classify_register_adjustment(expected, surface_mut)
    radical_candidates = radical_candidates_for_target(target_node, t2, expected)
    raw_is_radical = raw_has_radical_evidence(target_node, t2)

    # PATCH: h-initial surface forms under soft/soft_limited/nasal/aspirate
    # triggers are always the radical -- 'h' is not a mutated initial under
    # any of these mutation types (h-prothesis is 'h-mutation' only).
    # We bypass the tagger-confirmation requirement here because loanwords
    # and OOV items starting with 'h' (hostel, heddlu, etc.) will never
    # get a Cysill lemma match, causing raw_has_radical_evidence to return
    # False and the word to be falsely classified as erosion.
    if not raw_is_radical and _surface_is_h_initial_radical(t2["raw_word"], expected):
        raw_is_radical = True
    if raw_is_radical:
        matched_radical, matched_forms = None, set()
    else:
        matched_radical, matched_forms = _find_form_match(
            t2["raw_word"], expected, radical_candidates)
    invariant_radical = next(
        (r for r in radical_candidates if is_selectively_invariant(r, expected)),
        None)
    if raw_is_radical and not matched_radical:
        surface_mut = None

    if matched_radical:
        status, is_erosion = "correct_mutation", False
        surface_mut = surface_mut or expected[0]
        note = (f"Correct mutation by radical form: {matched_radical} -> "
                f"{t2['raw_word']}")
    elif surface_mut in expected or colloquial_mut in expected:
        status, is_erosion = "correct_mutation", False
        note = f"Correct mutation: {surface_mut or colloquial_mut}"
    elif invariant_radical and raw_is_radical:
        status, is_erosion = "selective_invariancy", False
        note = f"Selective Invariancy under {expected}"
    elif raw_is_radical and surface_mut is None:
        if invariant_radical:
            status, is_erosion = "selective_invariancy", False
            note = f"Selective Invariancy under {expected}"
        elif any(m in expected for m in ["soft", "soft_limited", "nasal", "aspirate"]):
            # PATCH: require at least one tagger to have tagged this token with
            # a real POS before calling it erosion. When both taggers return
            # unknown/absent, the radical appearance may be a hallucination,
            # OOV item, or proper noun -- not a genuine mutation context.
            if not pos_tagger_agreement_sufficient(
                    target_node.get("cysill_pos"), target_node.get("spacy_token"),
                    require_erosion=True):
                status, is_erosion = "erosion_unverified", False
                note = (f"Radical used under {expected} but both taggers absent "
                        f"-- insufficient evidence for erosion classification")
            else:
                status, is_erosion = "erosion", True
                note = f"**EROSION**: expected {expected}, radical used"
                if register_class == "colloquial_variant":
                    note += " | documented colloquial variant"
        else:
            status, is_erosion = "mutation_absent_expected_none", False
            note = "No mutation required."
    elif surface_mut is not None:
        # PATCH: same POS gate for wrong-type erosion
        if not pos_tagger_agreement_sufficient(
                target_node.get("cysill_pos"), target_node.get("spacy_token"),
                require_erosion=True):
            status, is_erosion = "erosion_unverified", False
            note = (f"Wrong mutation type apparent (expected {expected}, got "
                    f"{surface_mut}) but both taggers absent -- unverified")
        else:
            status, is_erosion = "wrong_mutation_type", True
            note = f"**EROSION (wrong type)**: expected {expected}, got {surface_mut}"
    else:
        status, is_erosion = "mutation_mismatch", False
        note = f"Mismatch: expected {expected}"

    return {"skip": False, "t2": t2, "status": status, "is_erosion": is_erosion,
            "note": note, "register_class": register_class,
            "surface_mut": surface_mut, "colloquial_mut": colloquial_mut,
            "radical_candidates": "|".join(radical_candidates),
            "matched_radical": matched_radical,
            "expected_surface_forms": "|".join(sorted(matched_forms))}


def _build_row(current_node, target_node, expected, outcome,
               conf_current, trigger_word, rule_name):
    t2          = outcome["t2"]
    surface_mut = outcome.get("surface_mut")
    cysill_mut  = target_node.get("cysill_mutation_type")
    spacy_tok   = target_node.get("spacy_token")
    is_erosion  = outcome["is_erosion"]
    cysill_pos  = target_node.get("cysill_pos")
    cpos_coarse = cysill_coarse_pos(cysill_pos)
    spos_coarse = spacy_coarse_pos(spacy_tok)
    pos_ok      = pos_compatible(cysill_pos, spacy_tok)

    # PATCH: cysill_has_signal used to be `bool(target_node.get("cysill_aligned"))`
    # alone, computed further down and used only for tagger_agreement.
    # cysill_aligned is True for locally-resolved rows too (Bangor lexicon
    # hits / recognized code-switch words -- see enrich_words()), since
    # those get a synthetic, non-None cysill_tok. That let a lexicon-only
    # answer masquerade as genuine live Cysill corroboration. Confirmed
    # live (real run, 2026-08-02): 6 rows stamped "all_three"/
    # "cysill+heuristic" with cysill_pos completely empty. cysill_pos is
    # deliberately left None for every locally-resolved row (see
    # bangor_lexicon.py's module docstring), so excluding locally_resolved
    # rows here is the direct fix, not a proxy. Computed BEFORE the
    # compute_confidence() call below (moved up from where it used to
    # live, further down this function) so tagger_agreement and
    # detection_source/confidence_score are driven by the exact same
    # signal instead of two independently-computed approximations of it.
    cysill_has_signal = (bool(target_node.get("cysill_aligned"))
                          and not target_node.get("locally_resolved"))

    confidence, detection_source = compute_confidence(
        outcome["status"], is_erosion, spacy_tok,
        cysill_mut, expected, surface_mut,
        cysill_genuinely_aligned=cysill_has_signal)

    parser_mut_type = SPACY_MUTATION_MAP.get(
        spacy_tok["mutation"]) if spacy_tok and spacy_tok.get("mutation") else None
    # PATCH: previously `cysill_agrees`/`parser_agrees` were computed as
    # `(cysill_mut == surface_mut) if (cysill_mut and surface_mut) else None`
    # -- a truthy guard that requires BOTH values to be non-None/non-empty
    # before even attempting the comparison. That's correct when checking
    # "did the tagger confirm the SAME mutation type", but for any row
    # where surface_mut is None (erosion, mutation_mismatch,
    # selective_invariancy -- by definition, nothing was found on the
    # surface), the guard fails immediately regardless of what the tagger
    # actually said, forcing cysill_agrees/parser_agrees to None every
    # time. Since `elif cysill_agrees is None and parser_agrees is None:
    # agreement = "heuristic_only"` then always fires, EVERY erosion/
    # mismatch/selective-invariancy row was unconditionally stamped
    # "heuristic_only" -- even when Cysill's own tagger explicitly and
    # independently reported "no mutation" on that exact word, which is
    # genuine corroboration, not an absence of signal. Confirmed against
    # real output: 68/68 erosion rows had cysill_mutation_type == "none"
    # (Cysill agreeing) yet 68/68 were labelled heuristic_only, hiding
    # 100% real corroboration.
    #
    # Fix: only treat a tagger as having no signal when it genuinely
    # wasn't consulted or didn't align (cysill_aligned / spacy_aligned is
    # False/missing) -- not merely when its answer happens to be None,
    # since None from an aligned tagger legitimately means "confirmed no
    # mutation" and should compare equal to a None surface_mut.
    parser_has_signal = spacy_tok is not None
    cysill_agrees = (cysill_mut == surface_mut) if cysill_has_signal else None
    parser_agrees = (parser_mut_type == surface_mut) if parser_has_signal else None

    if cysill_agrees and parser_agrees:
        agreement = "all_three"
    elif cysill_agrees:
        agreement = "cysill+heuristic"
    elif parser_agrees:
        agreement = "parser+heuristic"
    elif cysill_agrees is None and parser_agrees is None:
        agreement = "heuristic_only"
    else:
        agreement = "disputed"

    return {
        "timestamp":             f"{current_node['start']}s - {target_node['end']}s",
        "trigger_word":          trigger_word,
        "rule":                  rule_name,
        "following_word":        t2["raw_word"],
        "lemma":                 t2["lemma"],
        "expected_mutation":     expected[0] if len(expected) == 1 else "|".join(expected),
        "mutation_found":        surface_mut or outcome.get("colloquial_mut") or "none",
        "cysill_mutation_type":  cysill_mut or "none",
        "cysill_pos":            cysill_pos,
        "cysill_coarse_pos":     cpos_coarse,
        "spacy_dep":             spacy_tok["dep"] if spacy_tok else "none",
        "spacy_pos":             spacy_tok["pos"] if spacy_tok else "none",
        "spacy_coarse_pos":      spos_coarse,
        "pos_compatible":        pos_ok,
        "spacy_mutation":        spacy_tok["mutation"] if spacy_tok else "none",
        "spacy_gender":          spacy_tok["gender"] if spacy_tok else "none",
        "tagger_agreement":      agreement,
        "confidence_score":      confidence,
        "detection_source":      detection_source,
        "status":                outcome["status"],
        "note":                  outcome["note"],
        "is_erosion":            is_erosion,
        "is_high_confidence_erosion": is_erosion and confidence >= EROSION_CONFIDENCE_THRESHOLD,
        "is_code_switch":        False,
        "collision_flag":        target_node.get("collision_note"),
        "trigger_confidence":    conf_current,
        "following_confidence":  target_node.get("confidence"),
        "register_class":        outcome["register_class"],
        "radical_candidates":    outcome.get("radical_candidates"),
        "matched_radical":       outcome.get("matched_radical"),
        "expected_surface_forms":outcome.get("expected_surface_forms"),
        # PATCH: exposes the same signal cysill_has_signal above is built
        # from, directly on the output row -- lets corpus_analyzer.py (or
        # a manual spot-check) filter/audit locally-resolved rows without
        # having to infer them from cysill_pos being empty.
        "locally_resolved":      bool(target_node.get("locally_resolved")),
    }


def _build_cs_row(current_node, target_node, t1, norm_current, conf_current,
                  rule_name="word_trigger", expected_override=None):
    expected = expected_override if expected_override is not None else t1["expected_mutation"]
    spacy_tok = target_node.get("spacy_token")
    cysill_pos = target_node.get("cysill_pos")
    return {
        "timestamp":            f"{current_node['start']}s - {target_node['end']}s",
        "trigger_word":         norm_current,
        "rule":                 rule_name,
        "following_word":       normalize_word(target_node["word"]),
        "lemma":                None,
        "expected_mutation":    "|".join(expected) if len(expected) > 1 else expected[0],
        "mutation_found":       "code_switch",
        "cysill_mutation_type": "n/a",
        "cysill_pos":           cysill_pos,
        "cysill_coarse_pos":    cysill_coarse_pos(cysill_pos),
        "spacy_dep":            spacy_tok["dep"] if spacy_tok else "none",
        "spacy_pos":            spacy_tok["pos"] if spacy_tok else "n/a",
        "spacy_coarse_pos":     spacy_coarse_pos(spacy_tok),
        "pos_compatible":       pos_compatible(cysill_pos, spacy_tok),
        "spacy_mutation":       "n/a",
        "spacy_gender":         "n/a",
        "tagger_agreement":     "n/a",
        "confidence_score":     1.0,
        "detection_source":     "code_switch",
        "status":               "code_switch",
        "note":                 "Mutation trigger followed by non-Welsh token.",
        "is_erosion":           False,
        "is_high_confidence_erosion": False,
        "is_code_switch":       True,
        "collision_flag":       None,
        "trigger_confidence":   conf_current,
        "following_confidence": target_node.get("confidence"),
        "register_class":       "n/a",
        "radical_candidates":   None,
        "matched_radical":      None,
        "expected_surface_forms": None,
        "locally_resolved":     bool(target_node.get("locally_resolved")),
    }


def _evaluate_feminine_ei(current_node, target_node, trigger_word):
    if target_node.get("confidence", 0) < 0.65:
        return None
    t2 = layer_2_lemma_analysis(
        target_node["word"],
        cysill_pos=target_node.get("cysill_pos"),
        spacy_token=target_node.get("spacy_token"),
    )
    if t2.get("skip_reason"):
        return None
    expected = _resolve_feminine_ei_expected(t2["raw_word"])
    if expected is None:
        return None
    outcome = _evaluate_mutation_outcome(target_node, expected)
    if outcome["skip"]:
        return None
    return _build_row(current_node, target_node, expected, outcome,
                      current_node.get("confidence", 0.0),
                      trigger_word, "feminine_ei")


def _evaluate_mixed_mutation(current_node, target_node, trigger_word):
    if target_node.get("confidence", 0) < 0.65:
        return None
    t2 = layer_2_lemma_analysis(
        target_node["word"],
        cysill_pos=target_node.get("cysill_pos"),
        spacy_token=target_node.get("spacy_token"),
    )
    if t2.get("skip_reason"):
        return None
    expected = _resolve_mixed_mutation_expected(t2["raw_word"])
    outcome  = _evaluate_mutation_outcome(target_node, expected)
    if outcome["skip"]:
        return None
    return _build_row(current_node, target_node, expected, outcome,
                      current_node.get("confidence", 0.0),
                      trigger_word, "mixed_mutation")


def _evaluate_h_mutation(current_node, target_node, trigger_word):
    if target_node.get("confidence", 0) < 0.65:
        return None
    raw   = normalize_word(target_node["word"])
    lemma = get_welsh_lemma(raw)
    if is_english_code_switch(raw, lemma):
        return None
    base_form = normalize_word(lemma) if lemma else (raw[1:] if raw.startswith("h") else raw)
    if not base_form or base_form[0] not in WELSH_VOWELS:
        return None

    h_applied  = raw.startswith("h") and not (lemma or "").startswith("h")
    spacy_tok  = target_node.get("spacy_token")
    cysill_mut = target_node.get("cysill_mutation_type")
    cysill_pos = target_node.get("cysill_pos")
    # PATCH: same fix as _build_row() -- see its comment for the full
    # story (confirmed live: locally-resolved rows getting credited as
    # genuine Cysill corroboration). This rule builds its row differently
    # from _build_row() and has its own compute_confidence() call, so it
    # needed the same fix applied independently rather than inheriting it.
    cysill_has_signal = (bool(target_node.get("cysill_aligned"))
                          and not target_node.get("locally_resolved"))

    status, is_erosion = ("correct_mutation", False) if h_applied else ("erosion", True)
    note = "h-mutation correctly applied." if h_applied else \
        f"**EROSION**: expected h-mutation after {trigger_word}, radical used."
    surface_mut = "h-mutation" if h_applied else None

    confidence, detection_source = compute_confidence(
        status, is_erosion, spacy_tok, cysill_mut, ["h-mutation"], surface_mut,
        cysill_genuinely_aligned=cysill_has_signal)

    return {
        "timestamp":            f"{current_node['start']}s - {target_node['end']}s",
        "trigger_word":         trigger_word,
        "rule":                 "h_mutation",
        "following_word":       raw,
        "lemma":                lemma,
        "expected_mutation":    "h-mutation",
        "mutation_found":       "h-mutation" if h_applied else "none",
        "cysill_mutation_type": cysill_mut or "none",
        "cysill_pos":           cysill_pos,
        "cysill_coarse_pos":    cysill_coarse_pos(cysill_pos),
        "spacy_dep":            spacy_tok["dep"] if spacy_tok else "none",
        "spacy_pos":            spacy_tok["pos"] if spacy_tok else "none",
        "spacy_coarse_pos":     spacy_coarse_pos(spacy_tok),
        "pos_compatible":       pos_compatible(cysill_pos, spacy_tok),
        "spacy_mutation":       spacy_tok["mutation"] if spacy_tok else "none",
        "spacy_gender":         spacy_tok["gender"] if spacy_tok else "none",
        "tagger_agreement":     "n/a",
        "confidence_score":     confidence,
        "detection_source":     detection_source,
        "status":               status,
        "note":                 note,
        "is_erosion":           is_erosion,
        "is_high_confidence_erosion": is_erosion and confidence >= EROSION_CONFIDENCE_THRESHOLD,
        "is_code_switch":       False,
        "collision_flag":       target_node.get("collision_note"),
        "trigger_confidence":   current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "register_class":       "n/a",
        "radical_candidates":   None,
        "matched_radical":      None,
        "expected_surface_forms": None,
        "locally_resolved":     bool(target_node.get("locally_resolved")),
    }


def _find_lookahead_target(i, words_list, norm_current):
    """
    PATCH: the original skip condition `norm_b == norm_current` also skipped
    legitimate doubled trigger words (e.g. "i i mewn").  Now we only skip
    the repetition when the current word is itself a mutation trigger, which
    is the actual stutter/emphasis case we want to absorb.
    """
    lookahead, target_found = 1, None
    current_is_trigger = norm_current in TRIGGERS
    while lookahead <= 3 and (i + lookahead) < len(words_list):
        possible_b = words_list[i + lookahead]
        norm_b     = normalize_word(possible_b["word"])
        skip_repeat = current_is_trigger and norm_b == norm_current
        if skip_repeat or norm_b in WELSH_FILLERS or possible_b.get("synthetic"):
            lookahead += 1
        else:
            target_found = possible_b
            break
    return target_found, lookahead


def _process_word_trigger(i, words_list, current_node, norm_current, t1, conf_current):
    target_found, lookahead = _find_lookahead_target(i, words_list, norm_current)
    if not target_found or target_found.get("confidence", 0) < 0.65:
        return None, lookahead

    expected = t1["expected_mutation"]
    target_norm = normalize_word(target_found["word"])
    target_lemma = get_welsh_lemma(target_norm)
    if is_english_code_switch(target_norm, target_lemma):
        return _build_cs_row(current_node, target_found, t1,
                             norm_current, conf_current), lookahead

    # PATCH: mae/ydy/oes false-trigger fix. Confirmed live in the corpus:
    # "mae gan praed popeth..." was scoring 'following: gan, expected:
    # soft, found: none' as erosion -- but "gan" is the preposition that
    # STARTS the interpolated PP "gan [possessor]", not a mutation target
    # at all. More generally, the word right after a bare bod-form is
    # normally the SUBJECT, which never mutates ("Mae ci yn cysgu"). The
    # real (currently unimplemented) mutation site in "mae gan X Y"
    # constructions is Y -- the word AFTER the whole PP -- per
    # Wiktionary's "word after an interpolated prepositional phrase" rule
    # (mae yn yr ardd gi -> ci mutates to gi). That's a positive rule that
    # needs PP-span detection via the dependency tree, not simple
    # lookahead, and is tracked separately (see potential_fixes.md --
    # needs its own impact scan before it's added). Until then, this just
    # stops the false positive: skip evaluation when what's next is (a)
    # itself a known trigger/function word (we've landed on the START of
    # a PP, not a target) or (b) the subject (spaCy dep == "nsubj").
    if norm_current in BOD_SUBJECT_EXEMPT_TRIGGERS and expected:
        target_dep = (target_found.get("spacy_token", {}) or {}).get("dep")
        if target_norm in TRIGGERS or target_dep == "nsubj":
            return None, lookahead

    # PATCH: yn + verb-noun exemption. Per Wiktionary's Welsh mutations
    # appendix, "yn" only mutates in two environments: the predicate
    # particle (yn + noun/adjective -> soft mutation) and the preposition
    # "in" (yn + noun -> nasal mutation). Neither covers the aspectual
    # "yn" + verb-noun construction (e.g. "mae'n canu", "maen nhw'n
    # cystadlu") -- that's a third, distinct grammatical function of "yn"
    # that simply isn't a mutation environment at all. layer_1 resolves
    # PREDYN/VERBADJ/VB/ADV tags on the *trigger* word itself to
    # "soft_limited", which conflates true predicate "yn" with aspectual
    # "yn" when Cysill/spaCy tag the trigger ambiguously or when POS info
    # on "yn" is missing (heuristic-only rows especially -- see the
    # techiaith-verbatim "ddwy ddim yn cystadlu" false positive). The
    # target word's own POS is the reliable signal: if what follows "yn"
    # is itself a verb, this is the aspect construction, not a mutation
    # environment -- skip evaluation entirely rather than counting the
    # correct radical form as erosion.
    if norm_current == "yn" and expected:
        target_spacy_pos  = (target_found.get("spacy_token", {}) or {}).get("pos", "")
        target_cysill_pos = (target_found.get("cysill_pos") or "").split("+")[0].strip().upper()
        target_is_verb = (target_spacy_pos == "VERB") or target_cysill_pos.startswith("VB")
        if target_is_verb:
            return None, lookahead

    # PATCH: ll/rh soft-mutation exemption (see LL_RH_SOFT_EXEMPT_TRIGGERS_*
    # above). If the expected mutation is soft/soft_limited and the target's
    # radical starts with ll or rh, this trigger doesn't actually mutate it
    # -- skip evaluation rather than wrongly counting the correct radical
    # form as erosion.
    if expected and any(e in ("soft", "soft_limited") for e in expected):
        radical_cluster = initial_cluster(target_lemma or target_norm)
        if radical_cluster in ("ll", "rh"):
            target_pos = (target_found.get("spacy_token", {}) or {}).get("pos", "")
            exempt = (
                norm_current in LL_RH_SOFT_EXEMPT_TRIGGERS_ANY_POS
                or (norm_current in LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY
                    and target_pos != "ADJ")
            )
            if exempt:
                return None, lookahead

    if t1.get("fem_ei"):
        return _evaluate_feminine_ei(current_node, target_found, norm_current), lookahead
    if t1.get("mixed"):
        return _evaluate_mixed_mutation(current_node, target_found, norm_current), lookahead
    if t1.get("h_mutation") or expected == ["h-mutation"]:
        return _evaluate_h_mutation(current_node, target_found, norm_current), lookahead

    # PATCH: unmutable-initial exemption. Some initial clusters simply
    # have no defined output under ANY of a word's expected mutation
    # types -- most commonly loanwords/proper nouns with non-native
    # Welsh initials (e.g. "Victoria" -- Welsh has no native word-initial
    # "v", so SOFT_MUTATION has no "v" entry at all and there is no such
    # thing as a soft-mutated "Victoria"), or a nasal-mutation trigger
    # followed by a vowel-initial word (NASAL_MUTATION only maps
    # p/t/c/b/d/g -- vowels have no nasal-mutated form). Without this,
    # any trigger landing in front of such a word gets scored as erosion
    # for a mutation that was never phonologically possible to begin
    # with, not one that actually eroded. Placed after the fem_ei/mixed/
    # h_mutation dispatches above (those have their own separate rules,
    # e.g. h-mutation specifically targets vowel-initial words) so this
    # only gates the generic soft/soft_limited/nasal/aspirate path below.
    if expected:
        radical_cluster = initial_cluster(target_lemma or target_norm)
        mutation_tables = {
            "soft": SOFT_MUTATION, "soft_limited": SOFT_MUTATION_LIMITED,
            "nasal": NASAL_MUTATION, "aspirate": ASPIRATE_MUTATION,
        }
        checkable = [e for e in expected if e in mutation_tables]
        if checkable and not any(radical_cluster in mutation_tables[e] for e in checkable):
            return None, lookahead

    outcome = _evaluate_mutation_outcome(target_found, expected)
    if outcome["skip"]:
        if outcome["t2"].get("is_code_switch"):
            return _build_cs_row(current_node, target_found, t1,
                                 norm_current, conf_current), lookahead
        return None, lookahead

    return _build_row(current_node, target_found, expected, outcome,
                      conf_current, norm_current, "word_trigger"), lookahead


def _process_gender_trigger(current_node, target_node, expected, trigger_label,
                             rule_name, require_target_gender, require_target_is_noun,
                             require_target_not_plural=False):
    """
    PATCH: guard against synthetic target tokens so gender-trigger layers
    (1B–1G) don't silently advance past synthetic words without analysis.
    """
    if target_node.get("synthetic"):
        return None
    if target_node.get("confidence", 0) < 0.65:
        return None
    target_norm = normalize_word(target_node["word"])
    target_lemma = get_welsh_lemma(target_norm)
    if is_english_code_switch(target_norm, target_lemma):
        return _build_cs_row(
            current_node, target_node, None, trigger_label,
            current_node.get("confidence", 0.0),
            rule_name=rule_name, expected_override=expected)
    spacy_tok     = target_node.get("spacy_token")
    target_gender = (extract_gender_from_spacy(spacy_tok) or target_node.get("cysill_gender"))
    if require_target_gender is not None and target_gender != require_target_gender:
        return None
    # PATCH: feminine-singular-noun mutation rules (definite_article+fem_noun
    # etc.) don't apply to plurals -- "y ferch" mutates, "y merched" doesn't,
    # regardless of gender. Only rejects on a CONFIRMED plural; unknown
    # number (None) passes through rather than being treated as a reason
    # to skip, since most rows won't have reliable number data either way
    # and we don't want to silently stop evaluating rows we're actually
    # uncertain about.
    if require_target_not_plural:
        target_number = (extract_number_from_spacy(spacy_tok) or target_node.get("cysill_number"))
        if target_number == "plural":
            return None
    # PATCH: ll/rh soft-mutation exemption -- this check previously only
    # existed in _process_word_trigger (Layer 1A). It was never ported here,
    # even though LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY contains "y"/"yr"/"r"
    # specifically for Layer 1B (definite_article+fem_noun) and "un" for
    # Layer 1D -- both of which call THIS function, not _process_word_trigger.
    # Wiktionary's own examples are exactly this rule: "y llysywen" (the
    # eel), "i'r rhyd" (to the ford) -- ll/rh nouns stay unmutated after the
    # article, and "un llaw" (one hand) stays unmutated after "un". Without
    # this, every genuine instance of either was being counted as erosion.
    target_pos_for_llrh = spacy_tok["pos"] if spacy_tok else ""
    if expected and any(e in ("soft", "soft_limited") for e in expected):
        radical_cluster = initial_cluster(target_lemma or target_norm)
        if radical_cluster in ("ll", "rh"):
            exempt = (
                trigger_label in LL_RH_SOFT_EXEMPT_TRIGGERS_ANY_POS
                or (trigger_label in LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY
                    and target_pos_for_llrh != "ADJ")
            )
            if exempt:
                return None
    if require_target_is_noun:
        spacy_pos   = spacy_tok["pos"] if spacy_tok else ""
        cysill_pos  = target_node.get("cysill_pos") or ""
        is_noun = spacy_pos in ("NOUN", "PROPN") or cysill_pos.split("+")[0].startswith("N")
        if not is_noun:
            return None
    # PATCH: phonological exception -- d does not mutate to dd after s
    # (Wiktionary: "nos da", not "nos dda"). Only documented for the
    # fem_noun+adjective environment; without it, correctly-unmutated
    # "da" after "nos" gets wrongly counted as erosion.
    if rule_name == "fem_noun+adjective":
        trigger_raw = normalize_word(current_node.get("word", ""))
        target_radical = target_lemma or target_norm
        if trigger_raw.endswith("s") and initial_cluster(target_radical) == "d":
            return None

    outcome = _evaluate_mutation_outcome(target_node, expected)
    if outcome["skip"]:
        return None
    return _build_row(current_node, target_node, expected, outcome,
                      current_node.get("confidence", 0.0), trigger_label, rule_name)


def _process_phantom_check(current_node, cysill_pos, conf_current):
    """
    Phantom detection requires BOTH heuristic surface mutation AND
    Cysill tagger confirmation -- heuristic-only phantoms are too noisy.
    """
    if conf_current <= 0.85:
        return None
    if current_node.get("synthetic"):
        return None
    t2 = layer_2_lemma_analysis(
        current_node["word"],
        cysill_pos=cysill_pos,
        spacy_token=current_node.get("spacy_token"),
    )
    if t2.get("skip_reason"):
        return None
    surface_mut = t2["surface_mutation"]
    if not surface_mut:
        return None
    cysill_mut = current_node.get("cysill_mutation_type")
    if not cysill_mut:
        return None
    expected_phantom_mut = None
    for tag_key, mut in PHANTOM_CONTEXT_TAGS.items():
        if cysill_pos == tag_key or (cysill_pos or "").startswith(tag_key):
            expected_phantom_mut = mut
            break
    if not expected_phantom_mut or surface_mut != expected_phantom_mut:
        return None
    spacy_tok  = current_node.get("spacy_token")
    confidence = 0.70 if spacy_tok else 0.50
    cpos_coarse = cysill_coarse_pos(cysill_pos)
    spos_coarse = spacy_coarse_pos(spacy_tok)
    pos_ok = pos_compatible(cysill_pos, spacy_tok)
    return {
        "timestamp":             f"{current_node['start']}s - {current_node['end']}s",
        "trigger_word":          "[OMITTED]",
        "rule":                  "phantom_check",
        "following_word":        t2["raw_word"],
        "lemma":                 t2["lemma"],
        "expected_mutation":     expected_phantom_mut,
        "mutation_found":        surface_mut,
        "cysill_mutation_type":  cysill_mut or "none",
        "cysill_pos":            cysill_pos,
        "cysill_coarse_pos":     cpos_coarse,
        "spacy_dep":             spacy_tok["dep"] if spacy_tok else "none",
        "spacy_pos":             spacy_tok["pos"] if spacy_tok else "none",
        "spacy_coarse_pos":      spos_coarse,
        "pos_compatible":        pos_ok,
        "spacy_mutation":        spacy_tok["mutation"] if spacy_tok else "none",
        "spacy_gender":          spacy_tok["gender"] if spacy_tok else "none",
        "tagger_agreement":      "cysill+heuristic",
        "confidence_score":      confidence,
        "detection_source":      "cysill+heuristic",
        "status":                "phantom_mutation",
        "note":                  f"Phantom (heuristic+Cysill confirmed) on {cysill_pos}.",
        "is_erosion":            False,
        "is_high_confidence_erosion": False,
        "is_code_switch":        False,
        "collision_flag":        current_node.get("collision_note"),
        "trigger_confidence":    conf_current,
        "following_confidence":  conf_current,
        "register_class":        "colloquial_variant",
        "radical_candidates":    None,
        "matched_radical":       None,
        "expected_surface_forms": None,
    }


# ========================= MAIN MUTATION PROCESSOR =========================
def process_comprehensive_mutations(words_list):
    mutation_rows = []
    i = 0
    while i < len(words_list):
        current_node  = words_list[i]
        if current_node.get("synthetic"):
            i += 1
            continue

        norm_current  = normalize_word(current_node["word"])
        cysill_pos    = current_node.get("cysill_pos") or ""
        conf_current  = current_node.get("confidence", 0.0)
        spacy_tok     = current_node.get("spacy_token")
        gender        = current_node.get("gender")
        # PATCH: True when current_node was immediately followed by a
        # comma/semicolon/colon/period in the original transcript --
        # i.e. words_list[i+1] is NOT actually grammatically adjacent to
        # current_node, it just happens to sit next to it in the flat
        # token list because the punctuation marking a clause/list
        # boundary between them got stripped out during preprocessing
        # with nothing recording that it was ever there. See
        # preprocess_segment's _clause_boundary_after for the full
        # reasoning and the confirmed live false-positive this fixes
        # ("roi pob ffaith, pob myth" wrongly treating the next list
        # item's "pob" as an adjective following "ffaith").
        no_clause_boundary = not current_node.get("_clause_boundary_after")

        if conf_current < 0.65 or len(norm_current) <= 1:
            i += 1
            continue

        # PATCH: skip tokens with impossible Welsh diacritics (Whisper
        # hallucinations). These cannot be trigger words or valid targets.
        if not _is_plausible_welsh_token(norm_current):
            i += 1
            continue

        # PATCH: bod inflections are never valid trigger words in this context
        # (they appear in TRIGGERS as e.g. mae/ydy/oes which is fine, but
        # suppletive forms like roedd/buodd/fydd should not drive mutation
        # analysis as triggers since their environments are not predictable
        # from the TRIGGERS table).
        # Note: mae/ydy/oes/dyma/dyna are intentionally kept in TRIGGERS and
        # are NOT in BOD_SURFACE_FORMS, so they still function as triggers.
        if norm_current in BOD_SURFACE_FORMS and norm_current not in TRIGGERS:
            i += 1
            continue

        # ---- Layer 1A: word-level triggers ----
        t1 = layer_1_trigger_detection(
            norm_current, cysill_pos=cysill_pos,
            cysill_gender=current_node.get("cysill_gender"),
            spacy_token=spacy_tok)
        if t1["trigger_detected"]:
            row, consumed = _process_word_trigger(
                i, words_list, current_node, norm_current, t1, conf_current)
            if row:
                mutation_rows.append(row)
            i += consumed
            continue

        # ---- Layer 1B: definite article + feminine noun → soft_limited ----
        if norm_current in DEFINITE_ARTICLE_FORMS and no_clause_boundary and i + 1 < len(words_list):
            row = _process_gender_trigger(
                current_node, words_list[i + 1], ["soft_limited"],
                norm_current, "definite_article+fem_noun", "feminine", True,
                require_target_not_plural=True)
            if row:
                mutation_rows.append(row)
                i += 2
                continue

        # ---- Layer 1C: feminine noun + adjective → soft ----
        # PATCH: this is specifically a feminine SINGULAR phenomenon
        # ("ffrindiau da", plural, correctly does not mutate to "ffrindiau
        # dda" -- only "ffrind dda", singular, does). trigger_number gates
        # on the TRIGGER's own number here (not the target's, unlike Layer
        # 1B), since it's current_node -- "ffrindiau" -- whose gender is
        # being checked in this condition in the first place.
        spacy_pos = spacy_tok["pos"] if spacy_tok else ""
        trigger_number = extract_number_from_spacy(spacy_tok) or current_node.get("cysill_number")
        if gender == "feminine" and trigger_number != "plural" and \
                (spacy_pos in ("NOUN", "PROPN") or
                cysill_pos.split("+")[0].startswith("N")) and \
                no_clause_boundary and i + 1 < len(words_list):
            next_node = words_list[i + 1]
            nsp = next_node.get("spacy_token", {}).get("pos", "") if next_node.get("spacy_token") else ""
            ncp = next_node.get("cysill_pos") or ""
            if nsp == "ADJ" or ncp.startswith("ADJ"):
                row = _process_gender_trigger(
                    current_node, next_node, ["soft"],
                    norm_current, "fem_noun+adjective", None, False)
                if row:
                    mutation_rows.append(row)
                    i += 2
                    continue

            # ---- Layer 1C-bis: genitive noun+noun after fem singular noun
            # → soft ----
            # PATCH (Bucket 3a): "côt law", "ffon gerdded", "Gŵyl Ddewi" --
            # the second noun in a genitive/compound-like noun+noun pairing
            # mutates after a feminine singular noun. Gated on the spaCy
            # dependency "nmod" (nominal modifier) relation, not just
            # adjacency, to avoid firing on unrelated NOUN-NOUN sequences
            # (e.g. across a clause boundary or coordinated nouns).
            nsd = next_node.get("spacy_token", {}).get("dep", "") if next_node.get("spacy_token") else ""
            if (nsp in ("NOUN", "PROPN") or ncp.split("+")[0].startswith("N")) and nsd == "nmod":
                row = _process_gender_trigger(
                    current_node, next_node, ["soft"],
                    norm_current, "fem_noun+genitive_noun", None, True)
                if row:
                    mutation_rows.append(row)
                    i += 2
                    continue

        # ---- Layer 1D: un + feminine noun → soft_limited ----
        if norm_current in NUMERAL_FEM_SOFT_LIMITED_TRIGGERS and no_clause_boundary and i + 1 < len(words_list):
            row = _process_gender_trigger(
                current_node, words_list[i + 1], ["soft_limited"],
                norm_current, "numeral+fem_noun", "feminine", True)
            if row:
                mutation_rows.append(row)
                i += 2
                continue

        # ---- Layer 1E: dau/dwy + any noun → soft ----
        if norm_current in NUMERAL_GENERAL_SOFT_TRIGGERS and no_clause_boundary and i + 1 < len(words_list):
            next_node = words_list[i + 1]
            nsp = next_node.get("spacy_token", {}).get("pos", "") if next_node.get("spacy_token") else ""
            ncp = next_node.get("cysill_pos") or ""
            if nsp in ("NOUN", "PROPN") or ncp.split("+")[0].startswith("N"):
                row = _process_gender_trigger(
                    current_node, next_node, ["soft"],
                    norm_current, "numeral_general+noun", None, True)
                if row:
                    mutation_rows.append(row)
                    i += 2
                    continue

        # ---- Layer 1F: restricted nasal numerals ----
        if norm_current in NASAL_NUMERAL_TRIGGERS and no_clause_boundary and i + 1 < len(words_list):
            next_node  = words_list[i + 1]
            next_norm = normalize_word(next_node["word"])
            next_lemma_for_cs = get_welsh_lemma(next_norm)
            if next_node.get("confidence", 0) >= 0.65 and \
                    is_english_code_switch(next_norm, next_lemma_for_cs):
                row = _build_cs_row(
                    current_node, next_node, None, norm_current, conf_current,
                    rule_name="restricted_nasal_numeral",
                    expected_override=["nasal"])
                mutation_rows.append(row)
                i += 2
                continue
            next_lemma = get_welsh_lemma(next_node["word"]) or normalize_word(next_node["word"])
            if next_lemma in NASAL_NUMERAL_VALID_TARGETS:
                outcome = _evaluate_mutation_outcome(next_node, ["nasal"])
                if not outcome["skip"]:
                    row = _build_row(current_node, next_node, ["nasal"], outcome,
                                     conf_current, norm_current, "restricted_nasal_numeral")
                    mutation_rows.append(row)
                    i += 2
                    continue

        # ---- Layer 1G: feminine ordinal + noun → soft ----
        is_fem_ordinal = (cysill_pos.startswith("ORDF") or
                          (spacy_tok and spacy_tok.get("pos") == "ADJ" and gender == "feminine"))
        if is_fem_ordinal and no_clause_boundary and i + 1 < len(words_list):
            row = _process_gender_trigger(
                current_node, words_list[i + 1], ["soft"],
                norm_current, "feminine_ordinal+noun", None, True)
            if row:
                mutation_rows.append(row)
                i += 2
                continue

        # ---- Layer 1G-bis: preposed adjective + noun → soft (dependency-checked) ----
        # PATCH (Bucket 2): unlike a flat trigger-table entry, this requires
        # the spaCy dependency parse to confirm norm_current is actually an
        # ADJ modifying (amod) the immediately-following noun, before
        # treating it as a preposed-adjective mutation context. This rules
        # out e.g. "rhyw" used as the noun "sex/gender" rather than the
        # preposed determiner-adjective.
        if (norm_current in PREPOSED_ADJECTIVE_LEXICON and spacy_tok
                and spacy_tok.get("pos") == "ADJ"
                and spacy_tok.get("dep") == "amod"
                and no_clause_boundary and i + 1 < len(words_list)):
            next_node  = words_list[i + 1]
            next_raw   = normalize_word(next_node.get("word", ""))
            head_raw   = normalize_word(spacy_tok.get("head", ""))
            if head_raw and head_raw == next_raw:
                row = _process_gender_trigger(
                    current_node, next_node, ["soft"],
                    norm_current, "preposed_adjective+noun", None, True)
                if row:
                    mutation_rows.append(row)
                    i += 2
                    continue

        # ---- Layer 1G-ter: decompounded compound noun → soft ----
        # PATCH (Bucket 3b): catches a known compound's components spoken
        # as two separate radical-form words instead of one fused,
        # correctly-mutated word (see COMPOUND_NOUN_SECOND_ELEMENT above).
        if norm_current in COMPOUND_NOUN_SECOND_ELEMENT and no_clause_boundary and i + 1 < len(words_list):
            next_node = words_list[i + 1]
            next_norm = normalize_word(next_node.get("word", ""))
            next_lemma = get_welsh_lemma(next_norm)
            expected_second = COMPOUND_NOUN_SECOND_ELEMENT[norm_current]
            # only fires when the second word's radical matches the known
            # compound's second component -- not just any noun pair
            if (next_norm == expected_second or next_lemma == expected_second) \
                    and next_node.get("confidence", 0) >= 0.65:
                row = _process_gender_trigger(
                    current_node, next_node, ["soft"],
                    norm_current, "decompounded_noun", None, False)
                if row:
                    mutation_rows.append(row)
                    i += 2
                    continue

        # ---- Layer 1H: spaCy obj → soft (object of short-form verb) ----
        # PATCH: Direct Object Mutation (DOM) is a property of FINITE verbs
        # only -- per Wiktionary's mutations appendix, a verb-noun's object
        # only mutates if fronted (not the case here), and multiple sources
        # agree DOM applies specifically to objects of finite/short-form
        # verbs, not verb-nouns. POS alone (head_pos == "VERB") is NOT a
        # reliable way to check this: per UD_Welsh-CCG's own feature docs,
        # verb-nouns are tagged VerbForm=Vnoun but can surface under NOUN,
        # VERB, or AUX POS tags interchangeably (e.g. "teimlo" as a
        # verb-noun is tagged POS=NOUN, XPOS=verbnoun in the treebank).
        # Without this check, any verb-noun sitting where a finite verb
        # could sit (e.g. after "am", "i", "wedi" + VN) wrongly triggers
        # DOM on its object -- e.g. "am dreulio diwrnod" was wrongly
        # expecting soft mutation on "diwrnod" even though "dreulio" is a
        # verb-noun, not a short-form/finite verb. Require the head to be
        # explicitly finite (Fin/FinRel); unknown VerbForm (None) is
        # allowed through since most tokens don't carry this feature at
        # all and we don't want to silently stop evaluating rows we're
        # actually uncertain about (same principle as require_target_not_plural).
        if spacy_tok and spacy_tok.get("dep") in OBJ_DEPS and \
                spacy_tok.get("head_dep") in ("ROOT", "ccomp", "xcomp") and \
                spacy_tok.get("head_verbform") != "Vnoun":
            t2 = layer_2_lemma_analysis(
                current_node["word"],
                cysill_pos=cysill_pos,
                spacy_token=spacy_tok,
            )
            if not t2.get("skip_reason"):
                outcome = _evaluate_mutation_outcome(current_node, ["soft"])
                if not outcome["skip"]:
                    fake = {"start": current_node["start"], "end": current_node["start"],
                            "word": "[SHORT_FORM_VERB]", "confidence": 1.0}
                    row = _build_row(fake, current_node, ["soft"], outcome,
                                     conf_current, "[SHORT_FORM_VERB]", "spacy_obj_soft")
                    mutation_rows.append(row)
                    i += 1
                    continue

        # ---- Layer 1I: spaCy vocative -> soft ----
        # BUGFIX: was `spacy_tok.get("dep") == "vocative"` -- a hardcoded
        # string duplicating VOCAT_DEPS from mutation_tables.py instead of
        # importing and using it (VOCAT_DEPS previously wasn't imported
        # into this file at all, so it sat there unused/orphaned -- easy to
        # mistake for dead code, per this file's own stated philosophy
        # that mutation_tables.py should be the single source of truth for
        # exactly this kind of dependency-role set). Switched to `in
        # VOCAT_DEPS` to match how OBJ_DEPS is used just above, and so a
        # future second vocative-marking dep label only needs adding in
        # one place.
        if spacy_tok and spacy_tok.get("dep") in VOCAT_DEPS:
            t2 = layer_2_lemma_analysis(
                current_node["word"],
                cysill_pos=cysill_pos,
                spacy_token=spacy_tok,
            )
            if not t2.get("skip_reason"):
                outcome = _evaluate_mutation_outcome(current_node, ["soft"])
                if not outcome["skip"]:
                    fake = {"start": current_node["start"], "end": current_node["start"],
                            "word": "[VOCATIVE]", "confidence": 1.0}
                    row = _build_row(fake, current_node, ["soft"], outcome,
                                     conf_current, "[VOCATIVE]", "spacy_vocative_soft")
                    mutation_rows.append(row)
                    i += 1
                    continue

        # ---- Layer 2: phantom (heuristic + Cysill required) ----
        row = _process_phantom_check(current_node, cysill_pos, conf_current)
        if row:
            mutation_rows.append(row)
        i += 1

    return mutation_rows
