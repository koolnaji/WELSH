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

import sys
import io
import os
import unicodedata
import random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter
import re
import pandas as pd
import yt_dlp
import requests
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not installed -- dependency parser disabled.")

try:
    from simplemma import lemmatize as simplemma_lemmatize
    from simplemma import is_known as simplemma_is_known
except Exception:
    simplemma_lemmatize = None
    simplemma_is_known = None

# ========================= CONFIGURATION =========================
BASE_DIR      = Path.home() / "Documents" / "welsh_analysis"
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

# PATCH: persistent lemma cache path
LEMMA_CACHE_PATH = BASE_DIR / "lemma_cache.json"

TECHIAITH_API_KEY         = os.getenv("WELSH_LEMMATIZER", "").strip()
WELSH_LEMMATIZER_ENDPOINT = "https://api.techiaith.cymru/cysill/v1/lemmatizer/v1"
WELSH_POS_ENDPOINT        = "https://api.techiaith.cymru/cysill/v1/pos/v1"

http_session = requests.Session()

# PATCH: circuit breaker state for Cysill API
_cysill_consecutive_failures = 0
_CYSILL_FAILURE_THRESHOLD    = 5
_cysill_disabled_for_run     = False


def reset_cysill_circuit_breaker():
    """
    Reset the Cysill circuit breaker for a fresh run.

    The three circuit-breaker globals are module-level, so they persist for
    the lifetime of the Python process.  If the API failed during a previous
    interactive choice (e.g. choice 1 → local MP3s) and the user then picks
    choice 3 (queue processing) in the same session, the breaker would remain
    open and silently disable Cysill for the second run.  Call this at the
    start of each top-level processing branch in main() to give each run a
    clean slate.
    """
    global _cysill_consecutive_failures, _cysill_disabled_for_run
    _cysill_consecutive_failures = 0
    _cysill_disabled_for_run     = False

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
# language mix hasn't been spot-checked yet. Don't fold these into either
# side of a formal/informal comparison until checked -- see BBC Cymru Wales
# below, which is a general bilingual channel (TV/Radio/Online), not
# confirmed Welsh-medium.
CURATED_CHANNELS = [
    {"url": "https://www.youtube.com/c/HanshS4C/videos",   "channel_register": "informal"},
    {"url": "https://www.youtube.com/@RowndaRownd/videos", "channel_register": "informal"},
    {"url": "https://www.youtube.com/@S4C/videos",         "channel_register": "informal"},
    {"url": "https://www.youtube.com/@BBCRadio_Cymru/videos", "channel_register": "formal"},
    {"url": "https://www.youtube.com/channel/UCNH53DFjL1OBl3oWNxMmC_Q/videos",
     "channel_register": "unverified"},  # BBC Cymru Wales -- bilingual, spot-check before use
]

# ========================= SPACY =========================
SPACY_NLP = None

def load_spacy():
    global SPACY_NLP
    if not SPACY_AVAILABLE:
        return False
    if SPACY_NLP is not None:
        return True
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SPACY_NLP = spacy.load("cy_ud_cy_ccg")
        print("✅ Welsh dependency parser loaded.")
        return True
    except Exception as e:
        print(f"⚠️  Could not load Welsh dependency parser: {e}")
        return False

def parse_spacy_doc(text):
    if SPACY_NLP is None:
        return None
    try:
        doc = SPACY_NLP(text)
        tokens = []
        for t in doc:
            morph = dict(t.morph.to_dict()) if t.morph else {}
            head_morph = dict(t.head.morph.to_dict()) if t.head.morph else {}
            tokens.append({
                "text":     t.text,
                "lemma":    t.lemma_,
                "dep":      t.dep_,
                "head":     t.head.text,
                "head_dep": t.head.dep_,
                "head_pos": t.head.pos_,
                # PATCH: needed to distinguish a finite verb from a verb-noun
                # governing this token -- POS alone (VERB/NOUN) is NOT
                # reliable for this in cy_ud_cy_ccg: verb-nouns can surface
                # as either POS tag, distinguished only by this feature.
                # Per UD_Welsh-CCG docs, VerbForm takes Fin/FinRel/Vnoun and
                # occurs on NOUN, VERB, and AUX tokens alike.
                "head_verbform": head_morph.get("VerbForm"),
                "pos":      t.pos_,
                "morph":    morph,
                "mutation": morph.get("Mutation"),
                "gender":   morph.get("Gender"),
                "is_punct": t.is_punct or t.is_space or t.pos_ in ("PUNCT", "SPACE"),
            })
        return tokens
    except Exception as e:
        tqdm.write(f" ⚠️ spaCy parse error: {e}")
        return None

# ========================= MUTATION TABLES =========================
SOFT_MUTATION = {
    "p": "b", "t": "d", "c": "g", "b": "f", "d": "dd", "g": "",
    "ll": "l", "rh": "r", "m": "f"
}
SOFT_MUTATION_LIMITED = {k: v for k, v in SOFT_MUTATION.items()
                         if k not in ("ll", "rh")}
NASAL_MUTATION    = {"p": "mh", "t": "nh", "c": "ngh", "b": "m", "d": "n", "g": "ng"}
ASPIRATE_MUTATION = {"p": "ph", "t": "th", "c": "ch"}
COLLOQUIAL_AFFRICATE_MUTATION = {"ts": "j"}

RADICAL_TO_MUTATED = {
    "soft": SOFT_MUTATION, "soft_limited": SOFT_MUTATION_LIMITED,
    "nasal": NASAL_MUTATION, "aspirate": ASPIRATE_MUTATION,
    "colloquial_affricate": COLLOQUIAL_AFFRICATE_MUTATION,
}

MUTATION_MAP = {}
for mtype, mapping in RADICAL_TO_MUTATED.items():
    for radical, mutated in mapping.items():
        if mutated and mutated not in MUTATION_MAP:
            MUTATION_MAP[mutated] = mtype
MUTATION_MAP[""] = "soft"

MUTATION_TAG_MAP  = {"TM": "soft", "TT": "nasal", "TL": "aspirate", "TH": "h-mutation"}
SPACY_MUTATION_MAP= {"SM": "soft", "NM": "nasal", "AM": "aspirate"}

SELECTIVE_INVARIANT_MAP = {
    "nasal":        {"m", "n", "ll", "rh", "f", "s", "ch", "h", "j"},
    "aspirate":     {"m", "n", "ll", "rh", "b", "d", "g", "f", "s", "h", "j"},
    "soft":         {"n", "f", "s", "ch", "h", "j"},
    "soft_limited": {"n", "f", "s", "ch", "h", "j"},
}
ASPIRATE_INITIALS = set(ASPIRATE_MUTATION.keys())

# ========================= TRIGGER TABLE =========================
TRIGGERS = {
    "am": "soft", "ar": "soft", "at": "soft", "dan": "soft",
    "dros": "soft", "drwy": "soft", "heb": "soft", "wrth": "soft",
    "gan": "soft", "i": "soft", "o": "soft", "hyd": "soft",
    "tua": "aspirate", "thua": "aspirate",
    "neu": "soft", "pan": "soft", "ail": "soft",
    "ni": "soft|aspirate", "nid": "soft|aspirate",
    "na": "soft|aspirate", "oni": "soft|aspirate",
    "beth": "soft", "pa": "soft", "pwy": "soft", "sut": "soft",
    "rhy": "soft", "lled": "soft", "pur": "soft_limited",
    "reit": "soft", "hollol": "soft", "gweddol": "soft",
    "go": "soft", "llwyr": "soft",
    # PATCH: pre-posed adjectives -- when these precede their noun (instead
    # of the usual noun+adjective order), the noun takes soft mutation
    # regardless of gender/number, e.g. "hen ddyn", "rhyw brynhawn",
    # "ychydig fisoedd", "amryw bethau", "prif fachgen".
    "hen": "soft", "rhyw": "soft", "ychydig": "soft",
    "amryw": "soft", "prif": "soft",
    "mor": "soft_limited", "cyn": "soft_limited",
    "mae": "soft", "ydy": "soft", "oes": "soft",
    "sy": "soft", "sydd": "soft",
    "dyma": "soft", "dyna": "soft", "yna": "soft",
    "mai": "soft", "taw": "soft", "pe": "soft",
    "fy": "nasal", "dy": "soft",
    "ei": "soft|aspirate",
    "ein": "h-mutation", "eu": "h-mutation", "u": "h-mutation",
    "tri": "aspirate", "chwe": "aspirate",
    "a": "soft|aspirate", "â": "aspirate",
    "gyda": "aspirate", "tra": "aspirate",
    "yn": "nasal|soft_limited", "ym": "nasal", "yng": "nasal",
    "yr": "soft_limited",
    "fe": "soft", "mi": "soft",
    "dau": "soft", "dwy": "soft",
}

DEFINITE_ARTICLE_FORMS             = {"y", "yr", "r"}
# PATCH: per Wiktionary's Welsh mutations appendix, ll and rh do NOT undergo
# soft mutation after these triggers, even though other soft-mutable
# consonants do. A pipeline that doesn't know this will see the correct,
# un-mutated "llyfr"/"rhyd" after one of these triggers and wrongly flag it
# as erosion, inflating the erosion rate. yn/cyn/mor/pur are exempt for any
# POS; yr/y/r/un are exempt specifically for nouns (adjectives still mutate,
# e.g. "y lonnaf", "un ryfedd" -- Wiktionary's own examples show ll/rh DO
# mutate after these when the target is an adjective, only nouns are
# exempt). "un llaw" (one hand) is Wiktionary's explicit un+noun example --
# "un" was missing from this set entirely, so a genuine "un llaw"/"un rhyd"
# would have wrongly expected mutation and flagged the correct radical as
# erosion.
LL_RH_SOFT_EXEMPT_TRIGGERS_ANY_POS = {"yn", "cyn", "mor", "pur"}
LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY = {"y", "yr", "r", "un"}
NUMERAL_FEM_SOFT_LIMITED_TRIGGERS  = {"un"}
NUMERAL_GENERAL_SOFT_TRIGGERS      = {"dau", "dwy"}
NASAL_NUMERAL_TRIGGERS             = {
    "pump", "saith", "wyth", "naw", "deng",
    "12", "15", "18", "20", "100",
    "deuddeg", "pymtheg", "deunaw", "ugain", "cant",
}
NASAL_NUMERAL_VALID_TARGETS        = {"blynedd", "blwydd", "diwrnod"}
MIXED_MUTATION_TRIGGERS            = {"ni", "nid", "na", "oni"}
# PATCH (Bucket 2): preposed adjectives -- when one of these precedes its
# noun (the inverse of normal Welsh noun-adjective order), the noun takes
# soft mutation. Kept as a closed lexicon (not just any ADJ) because the
# dependency check alone can't rule out other ADJ positions reliably, and
# because some of these words are polysemous (e.g. "rhyw" can be a noun
# meaning "sex/gender"); detection still requires the spaCy dependency
# check below to confirm the adjective genuinely modifies the following
# noun, not just lexical co-occurrence. "cyntaf" and other preposed
# superlatives are deliberately excluded -- Wiktionary notes they do not
# (reliably) trigger mutation. "prif" is excluded too: it has its own
# extra mutation-of-the-adjective-itself rule after a feminine article
# noun (e.g. "y brif fynedfa") that this simple noun-only layer can't
# represent correctly.
PREPOSED_ADJECTIVE_LEXICON         = {"hen", "rhyw", "ychydig", "amryw", 
                                      "unig", "annwyl", "gau", "unrhyw", "holl", "ambell", "aml",
                                      }
# PATCH (Bucket 3b): known Welsh compound nouns whose second element takes
# soft mutation when fused (e.g. "creigardd" = craig + (g)ardd). When
# spoken/transcribed correctly the word surfaces as ONE fused token, so
# there's nothing to flag -- correct usage is invisible to a token-pair
# detector by design, which is fine (no false erosion). The only erosion
# signature this layer can catch is the compound "coming apart" back into
# two separate radical-form words (e.g. "craig gardd" instead of
# "creigardd"). This is a small seed list, not exhaustive -- expand as
# more compounds turn up in the corpus.
COMPOUND_NOUN_SECOND_ELEMENT       = {
    "craig": "gardd",     # creigardd "rock garden"
    "haf":   "dydd",      # hafddydd "summer's day"
    "bar":   "morwyn",    # barforwyn "barmaid"
    "hwyl":  "pren",      # hwylbren "mast"
    "rhwyd": "gwaith",    # rhwydwaith "network"
    "modur": "tŷ",        # modurdy "garage"
}
OBJ_DEPS                           = {"obj", "iobj"}
VOCAT_DEPS                         = {"vocative"}

PHANTOM_CONTEXT_TAGS = {
    "NF": "soft", "NM": "soft", "VBF": "soft", "VB": "soft",
    "VB+NM": "soft", "NM+VBF": "soft", "PLACE": "soft", "PERSON": "soft",
}
GENDER_BEARING_PREFIXES  = ("N", "PRON", "CARD", "ORD")
KNOWN_HOMOGRAPH_COLLISIONS = {
    "chi": "Pronoun 'chi' (you) collides with aspirate-mutated 'ci' (dog -> chi).",
}
PREP_TAG_PREFIXES = ("PREP", "CPREP")
PREDYN_TAGS       = {"PREDYN", "VERBADJ", "PREDYN+VERBADJ"}
REL_INT_TAGS      = {"PRONREL", "PART", "EXCL"}

DIGRAPHS      = ["ngh", "mh", "nh", "dd", "ff", "ll", "ph", "rh", "th",
                 "ch", "ng", "ts", "j"]
WELSH_FILLERS = {"ym", "er", "ah", "iawn", "gwybod", "chdi", "te", "ffeil"}
WELSH_VOWELS  = {"a", "e", "i", "o", "u", "w", "y", "â", "ê", "î", "ô", "û", "ŵ", "ŷ"}

# High-confidence English function words and common spoken insertions.
# These are checked before Welsh lemmatisation/tagger evidence because Cysill
# can occasionally Welshify English tokens (e.g. "the" -> "te", "for" -> "mor").
ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "an", "and", "or", "but", "of", "to", "for", "with", "that",
    "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "as", "at", "from", "by", "if", "then", "than",
    "not", "yes", "you", "your", "they", "he", "she", "them", "his", "her",
    "their", "my", "got", "get", "just", "what", "who", "when", "where",
    "why", "how", "can", "could", "would", "should", "will", "do", "does",
    "did", "have", "has", "had", "because", "about", "into", "over",
    "after", "before", "there", "here", "thing", "things", "people",
    "that's", "it's", "we're", "you're", "they're", "i'm", "don't",
    "doesn't", "didn't", "can't", "couldn't", "wouldn't", "shouldn't",
    "won't",
})
WELSH_ENGLISH_HOMOGRAPHS = frozenset({
    "a", "am", "i", "in", "mi", "no", "un",
})

# ========================= BOD SUPPRESSION =========================
# All surface realisations of 'bod' (to be) suppressed as *mutation targets*.
# These are either suppletive (mae, yw, oedd -- no morphophonological link to
# a predictable radical) or genuinely mutated but in a verbal context that
# is not triggered by the preceding trigger word's mutation slot (fydd→fod,
# fu→bu). Analysing them as targets produces systematic false erosion counts.
#
# They are NOT suppressed as triggers -- mae/ydy/oes/dyma/dyna/sy/sydd
# remain in TRIGGERS and correctly predict soft mutation on the following
# noun/adjective. (Per Wiktionary's Welsh mutations appendix: "The verb
# form sydd, sy triggers soft mutation of a predicate noun or adjective" --
# sy/sydd were previously missing from TRIGGERS despite sitting right here
# in this set, silently undercounting every genuine sydd/sy-triggered
# context in the corpus.)
BOD_SURFACE_FORMS = frozenset({
    # Present tense
    "mae", "maen", "maent",
    "yw", "ydy", "ydyw",
    "oes",
    "sy", "sydd",
    # 1st/2nd/3rd person present
    "wyf", "wyt", "ydych", "ydym", "ydyn",
    # Imperfect / past
    "oedd", "oedden", "oeddech", "oeddet", "oeddwn", "oeddem",
    "roedd", "roedden", "roeddech", "roeddet", "roeddwn", "roeddem",
    # Preterite
    "bu", "buodd", "buon", "buoch", "buost", "bues", "buom",
    "fu", "fuodd", "fuon", "fuoch", "fuost", "fues", "fuom",
    # Future / conditional
    "bydd", "byddan", "byddwch", "byddi", "byddwn", "byddem", "byddent",
    "fydd", "fyddan", "fyddwch", "fyddi", "fyddwn", "fyddem", "fyddent",
    "byddai", "fyddai",
    # Subjunctive / literary conditional / pluperfect
    "byddaf", "byddo", "byddech",
    "bo", "boed", "foed",
    "buasai", "buasem", "buasent", "buaset", "buasech", "buaswn",
    "fuasai", "fuasem", "fuasent", "fuaset", "fuasech", "fuaswn",
    # Verbal noun (radical + soft-mutated form)
    "bod", "fod",
    # Colloquial / contracted forms
    "dw", "dwi",          # dw i (present 1sg)
    "on", "oni",          # o'n i (imperfect) -- 'oni' also a trigger but rare as target
    "dan", "dyn",         # dan ni / dyn ni
    "sa", "san", "set",   # sa i / conditional colloquial
    "se", "sen",          # southern conditional
})

# ========================= WELSH ORTHOGRAPHIC VALIDATION =========================
# Welsh uses exactly one productive diacritic: the circumflex (to bach) on the
# seven vowels â ê î ô û ŵ ŷ. Acute accents appear in a small number of
# loanwords (café, acíwt) and grave accents in a handful of poetry conventions.
# All other combining marks -- umlaut/diaeresis (ö ü), tilde (õ ã), breve,
# caron, cedilla, ring, ogonek -- are completely absent from native Welsh.
#
# When Whisper is forced into cy mode on low-energy or non-Welsh audio it
# hallucinates tokens containing these foreign diacritics (e.g. töii, tõii).
# This table lets us reject those tokens orthographically rather than relying
# on confidence scores, which are unreliable for hallucinated tokens.

WELSH_LEGAL_DIACRITIC_CHARS = frozenset(
    "âêîôûŵŷ"   # circumflex (to bach) -- canonical Welsh diacritic
    "áéíóú"     # acute -- attested in loanwords, tolerated
    "àèìòù"     # grave -- rare but attested in poetry / some loanword spellings
)

# The combining marks that correspond to the legal diacritics above.
# Any other combining mark signals a non-Welsh character.
_COMBINING_LEGAL = frozenset({
    "\u0302",   # combining circumflex  (â ê î ô û ŵ ŷ)
    "\u0301",   # combining acute       (á é í ó ú)
    "\u0300",   # combining grave       (à è ì ò ù)
})


def _is_plausible_welsh_token(word: str) -> bool:
    """
    Returns False if the token contains any diacritic combining mark that
    cannot appear in genuine Welsh orthography.

    Strategy: NFD-decompose each character. If it carries a combining mark
    that is not circumflex (U+0302), acute (U+0301), or grave (U+0300), the
    token is orthographically impossible in Welsh -- umlaut, tilde, breve,
    caron, cedilla, ring, ogonek, etc. all fail.

    Catches hallucinations such as:
        töii   (umlaut-o,   U+0308 combining diaeresis)
        tõii   (tilde-o,    U+0303 combining tilde)
        děkuji (caron-e,    U+030C combining caron)
        ţară   (cedilla-t,  U+0326 combining cedilla below)

    Does NOT reject:
        tŷ     (circumflex-y -- legal Welsh)
        llên   (circumflex-e -- legal Welsh)
        café   (acute-e     -- tolerated loanword)
    """
    for ch in unicodedata.normalize("NFC", word.lower()):
        nfd = unicodedata.normalize("NFD", ch)
        if len(nfd) == 1:
            continue  # plain character, no combining marks
        for mark in nfd[1:]:
            if mark not in _COMBINING_LEGAL:
                return False
    return True

LEMMA_CACHE = {}
COLLOQUIAL_REGISTER           = True
DECLINING_MUTATIONS           = {"nasal", "aspirate"}
POS_CHUNK_MAX_CHARS           = 2700
EROSION_CONFIDENCE_THRESHOLD  = 0.60
HALLUCINATION_REPEAT_RATIO    = 0.4
GAP_ALIGN_WINDOW              = 5    # lookahead window for gap-tolerant alignment

# Welsh contractions that Whisper may keep joined but Cysill/spaCy will split.
# Format: surface_form -> [sub_token_1, sub_token_2, ...]
# Only the first sub-token inherits the original word's timestamp/confidence.
WELSH_CONTRACTION_SPLITS = {
    "i'r":  ["i", "'r"],
    "a'r":  ["a", "'r"],
    "o'r":  ["o", "'r"],
    "yn y": ["yn", "y"],
    "i'w":  ["i", "'w"],
    "a'i":  ["a", "'i"],
    "o'i":  ["o", "'i"],
}


# ========================= HELPERS =========================
def ensure_dirs():
    for p in [BASE_DIR, AUDIO_DIR, TRANS_DIR, MUT_DIR, SUMMARY_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def run_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_paths(stamp):
    return {
        "segments": TRANS_DIR / f"segments_{stamp}.csv",
        "words":    TRANS_DIR / f"words_{stamp}.csv",
        "lemmas":   TRANS_DIR / f"lemmas_{stamp}.csv",
        "pos":      TRANS_DIR / f"pos_{stamp}.csv",
        "mutations":MUT_DIR   / f"mutations_{stamp}.csv",
    }


def _video_slug(meta, stamp):
    """
    Build a filesystem-safe filename slug for a single video, and organise
    that video's output CSVs into their own per-video subfolder rather than
    dumping everything flat into TRANS_DIR / MUT_DIR alongside every other
    video ever processed.

    Layout:
        transcriptions/<stamp>_<slug>/segments_<stamp>_<slug>.csv
        transcriptions/<stamp>_<slug>/words_<stamp>_<slug>.csv
        transcriptions/<stamp>_<slug>/lemmas_<stamp>_<slug>.csv
        transcriptions/<stamp>_<slug>/pos_<stamp>_<slug>.csv
        mutations/<stamp>_<slug>/mutations_<stamp>_<slug>.csv

    Filenames keep the stamp/slug (not just simplified to e.g. "words.csv")
    so corpus_analyzer.py's existing filename-based batch parsing keeps
    working unchanged even though the files now live one level deeper.
    Falls back to the video id, then the run stamp if neither is available.
    """
    import re
    title = meta.get("title") or meta.get("id") or stamp
    # keep only alphanumeric, spaces, hyphens; collapse whitespace; truncate
    slug = re.sub(r"[^\w\s-]", "", str(title), flags=re.UNICODE)
    slug = re.sub(r"[\s]+", "_", slug.strip())[:60]
    slug = slug or "untitled"

    folder_name = f"{stamp}_{slug}"
    video_trans_dir = TRANS_DIR / stamp / folder_name
    video_trans_dir.mkdir(parents=True, exist_ok=True)
    (MUT_DIR / stamp).mkdir(parents=True, exist_ok=True)

    return {
        "segments": video_trans_dir / f"segments_{stamp}_{slug}.csv",
        "words":    video_trans_dir / f"words_{stamp}_{slug}.csv",
        "lemmas":   video_trans_dir / f"lemmas_{stamp}_{slug}.csv",
        "pos":      video_trans_dir / f"pos_{stamp}_{slug}.csv",
        "mutations":MUT_DIR / stamp / f"mutations_{stamp}_{slug}.csv",
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

# ========================= LEMMA CACHE PERSISTENCE =========================
def load_lemma_cache():
    """Load persisted lemma cache from disk to avoid redundant API calls across runs."""
    global LEMMA_CACHE
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
        LEMMA_CACHE_PATH.write_text(
            json.dumps(LEMMA_CACHE, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"⚠️  Could not save lemma cache: {e}")

def get_welsh_lemma(word):
    w = normalize_word(word)
    if not w:
        return None
    if w in LEMMA_CACHE:
        return LEMMA_CACHE[w]
    lemma = None
    # PATCH: route through fetch_lemma so retry/reconnect logic applies
    if TECHIAITH_API_KEY:
        lemma = fetch_lemma(w)
    if not lemma and simplemma_lemmatize is not None:
        try:
            fallback = simplemma_lemmatize(w, lang="cy")
            lemma = fallback if fallback and fallback != w else None
        except Exception:
            lemma = None
    LEMMA_CACHE[w] = lemma
    return lemma

def extract_gender_from_pos(pos_tag):
    if not pos_tag:
        return None
    first = pos_tag.split("+")[0]
    for prefix in GENDER_BEARING_PREFIXES:
        if first.startswith(prefix):
            suffix = first[len(prefix):]
            if suffix == "F":
                return "feminine"
            elif suffix == "M":
                return "masculine"
    return None

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
                expanded.append({
                    **w,
                    "word":      part,
                    "synthetic": idx > 0,   # only first sub-token has real timestamp
                })
        elif surface == "'n":
            # Whisper already split it off as its own token -- just
            # canonicalize the spelling so it survives downstream and
            # matches TRIGGERS["yn"].
            expanded.append({**w, "word": "yn", "synthetic": False})
        elif surface.endswith("'n") and len(surface) > 2:
            # Fused "X'n" token (e.g. "mae'n") -- split into stem + "yn",
            # both real (non-synthetic) tokens.
            stem = surface[:-2]
            expanded.append({**w, "word": stem, "synthetic": False})
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
        surface = w.word.strip().strip(".,!?;:\"()[]")
        if not surface:
            continue
        raw_words.append({
            "word":       surface,
            "start":      round(w.start, 3),
            "end":        round(w.end, 3),
            "confidence": round(w.probability, 4),
            "synthetic":  False,
            "_seg_id":    seg_id,
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


# ========================= CYSILL API (with session resilience) =========================
def _cysill_get(endpoint, params, max_retries=3, base_delay=1.5):
    """
    HTTP GET with retry + session reconnection + circuit breaker.

    PATCH additions over original:
    - HTTP 429 (rate-limit) handling: respects Retry-After header
    - raise_for_status() surfaces 5xx errors as exceptions so the retry
      loop catches them instead of silently returning bad data
    - Jitter on backoff delay to avoid thundering-herd on long queues
    - Circuit breaker: if _CYSILL_FAILURE_THRESHOLD consecutive chunks
      fail the API is disabled for the rest of the run
    """
    global http_session, _cysill_consecutive_failures, _cysill_disabled_for_run

    if _cysill_disabled_for_run:
        return None

    for attempt in range(1, max_retries + 1):
        try:
            resp = http_session.get(endpoint, params=params, timeout=15)

            # PATCH: explicit 429 handling
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After",
                                                   base_delay * attempt * 2))
                tqdm.write(f" ⚠️ Cysill rate-limited (429) -- waiting {retry_after}s "
                      f"(attempt {attempt}/{max_retries})")
                time.sleep(retry_after)
                continue

            # PATCH: surface 5xx errors so the except block catches them
            resp.raise_for_status()

            # Success -- reset failure counter
            _cysill_consecutive_failures = 0
            return resp

        except requests.exceptions.ConnectionError:
            tqdm.write(f" ⚠️ Connection lost -- resetting session (attempt {attempt}/{max_retries})")
            http_session = requests.Session()
            # PATCH: jitter prevents all retries firing at the same moment
            time.sleep(base_delay * attempt + random.uniform(0, 0.5))
        except Exception as e:
            tqdm.write(f" ⚠️ Request error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(base_delay * attempt + random.uniform(0, 0.5))

    # All retries exhausted
    _cysill_consecutive_failures += 1
    if _cysill_consecutive_failures >= _CYSILL_FAILURE_THRESHOLD:
        _cysill_disabled_for_run = True
        tqdm.write(f" 🔴 Cysill API disabled for this run after "
              f"{_cysill_consecutive_failures} consecutive failures. "
              f"Pipeline will continue with spaCy + heuristics only.")
    return None

def parse_pos_result_extended(result):
    """
    Parse raw Cysill POS result string.
    Strips punct/EOS/space tokens BEFORE returning so the token list
    is content-only and matches Whisper's content-only word stream.
    """
    if not result:
        return []
    SKIP_POS = {"punct", "PUNCT", "EOS", "space"}
    out = []
    for item in str(result).split():
        parts = item.split("/")
        if len(parts) < 2:
            continue
        token, pos_tag = parts[0], parts[1]
        if pos_tag in SKIP_POS:
            continue
        mutation_raw = parts[2] if len(parts) >= 3 else None
        mutation_tag = None if mutation_raw in (None, "-") else mutation_raw
        out.append({
            "token":                token,
            "text":                 token,   # unified key for _tokens_match_fuzzy
            "pos":                  pos_tag,
            "mutation_tag":         mutation_tag,
            "tagger_mutation_type": MUTATION_TAG_MAP.get(mutation_tag),
            "gender":               extract_gender_from_pos(pos_tag),
            "is_punct":             False,   # already stripped above
        })
    return out

def fetch_pos_for_chunk(chunk_text, max_retries=3, base_delay=1.5):
    if not TECHIAITH_API_KEY:
        return None
    resp = _cysill_get(WELSH_POS_ENDPOINT,
                       {"text": chunk_text, "api_key": TECHIAITH_API_KEY},
                       max_retries=max_retries, base_delay=base_delay)
    if resp is None:
        tqdm.write(" 💥 POS API permanently failed for this chunk.")
        return None
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is True:
            return parse_pos_result_extended(data.get("result"))
        tqdm.write(f" ⚠️ POS API success=False: {data}")
    except Exception as e:
        tqdm.write(f" ⚠️ POS API parse error: {e}")
    return None

def fetch_lemma(word, max_retries=2, base_delay=1.0):
    if not TECHIAITH_API_KEY:
        return None
    resp = _cysill_get(WELSH_LEMMATIZER_ENDPOINT,
                       {"text": word, "api_key": TECHIAITH_API_KEY},
                       max_retries=max_retries, base_delay=base_delay)
    if resp is None:
        return None
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is True:
            return data.get("result")
    except Exception:
        pass
    return None

def chunk_words_for_pos(all_words, max_chars=POS_CHUNK_MAX_CHARS):
    """
    Group a flat list of preprocessed word dicts into chunks that fit
    within the Cysill API character limit. Used for both Cysill and spaCy
    so both taggers see the same chunk boundaries.

    Unlike the segment-based chunker, this operates on preprocessed words
    (already expanded) so chunk text is built from the same token list
    used for alignment. This guarantees the text sent to taggers matches
    the token stream we align against.

    PATCH: a change in "_seg_id" between consecutive words now forces a
    new chunk, even if the character budget has room left. Previously
    chunking was purely character-budget-driven with no awareness of
    Whisper segment boundaries (silence gaps, and in multi-speaker audio,
    likely speaker changes) -- a chunk could span across one of those
    boundaries and hand the tagger a block of text with a sentence, or
    two different speakers' turns, silently spliced together mid-parse.
    Words with no "_seg_id" (e.g. from analyze_phrase's synthetic word
    list, which isn't segment-derived) all share the value None and so
    never trigger this break -- unchanged behavior for that path.
    """
    chunks    = []
    cur_words = []
    cur_len   = 0
    cur_seg   = None

    for w in all_words:
        word_len = len(w["word"]) + 1  # +1 for space
        seg_id   = w.get("_seg_id")
        seg_changed = cur_words and seg_id != cur_seg
        over_budget = cur_words and cur_len + word_len > max_chars
        if seg_changed or over_budget:
            chunks.append(cur_words)
            cur_words = []
            cur_len   = 0
        cur_words.append(w)
        cur_len += word_len
        cur_seg  = seg_id

    if cur_words:
        chunks.append(cur_words)
    return chunks


def enrich_words(all_preprocessed_words):
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
    """
    chunks   = chunk_words_for_pos(all_preprocessed_words)
    enriched = []

    for chunk in chunks:
        chunk_text = " ".join(w["word"] for w in chunk)

        # ---- Cysill ----
        cysill_tokens  = fetch_pos_for_chunk(chunk_text) or []
        cysill_aligned = align_with_gap_tolerance(cysill_tokens, chunk)

        # ---- spaCy ----
        spacy_raw     = parse_spacy_doc(chunk_text) if SPACY_NLP else []
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
            entry["cysill_aligned"]       = cysill_tok is not None
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

    return enriched


# ========================= CONFIDENCE SCORING =========================
def compute_confidence(status, is_erosion, spacy_tok, cysill_mutation_type,
                        heuristic_expected, heuristic_surface):
    has_parser      = spacy_tok is not None
    parser_mut      = spacy_tok["mutation"] if has_parser else None
    parser_mut_type = SPACY_MUTATION_MAP.get(parser_mut) if parser_mut else None

    if not is_erosion:
        parser_c   = parser_mut_type and (parser_mut_type == heuristic_surface or
                      parser_mut_type in (heuristic_expected or []))
        cysill_c   = cysill_mutation_type and (cysill_mutation_type == heuristic_surface or
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
        cysill_rad = cysill_mutation_type is None
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

    for mut_type, mapping in RADICAL_TO_MUTATED.items():
        if mut_type not in allowed:
            continue
        for radical_initial, mutated_initial in sorted(
                mapping.items(), key=lambda x: len(x[1]), reverse=True):
            if mutated_initial and surface.startswith(mutated_initial):
                candidates.add(radical_initial + surface[len(mutated_initial):])
            elif mutated_initial == "" and radical_initial == "g" and surface[0] in WELSH_VOWELS:
                candidates.add("g" + surface)

    if "h-mutation" in allowed and surface.startswith("h") and len(surface) > 1:
        if surface[1] in WELSH_VOWELS:
            candidates.add(surface[1:])

    return {c for c in candidates if c and c != surface}

def radical_candidates_for_target(target_node, t2, expected):
    candidates = []

    def add(value):
        value = normalize_word(value or "")
        if value and value not in candidates:
            candidates.append(value)

    add(t2.get("lemma"))
    spacy_tok = target_node.get("spacy_token")
    if spacy_tok:
        add(spacy_tok.get("lemma"))
    for candidate in reverse_mutation_candidates(t2.get("raw_word"), expected):
        add(candidate)
    add(t2.get("raw_word"))
    return candidates

def raw_has_radical_evidence(target_node, t2):
    """
    Returns True only when there is reliable external evidence that the raw
    surface form IS the radical (i.e. no mutation occurred).

    Evidence hierarchy:
    1. Cysill successfully tagged this token AND its lemma equals the surface
       form -- the tagger has seen the radical form directly.

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
    if lemma and raw == lemma and target_node.get("cysill_aligned"):
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

    confidence, detection_source = compute_confidence(
        outcome["status"], is_erosion, spacy_tok,
        cysill_mut, expected, surface_mut)

    parser_mut_type = SPACY_MUTATION_MAP.get(
        spacy_tok["mutation"]) if spacy_tok and spacy_tok.get("mutation") else None
    cysill_agrees   = (cysill_mut == surface_mut) if (cysill_mut and surface_mut) else None
    parser_agrees   = (parser_mut_type == surface_mut) if (parser_mut_type and surface_mut) else None

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

    status, is_erosion = ("correct_mutation", False) if h_applied else ("erosion", True)
    note = "h-mutation correctly applied." if h_applied else \
        f"**EROSION**: expected h-mutation after {trigger_word}, radical used."
    surface_mut = "h-mutation" if h_applied else None

    confidence, detection_source = compute_confidence(
        status, is_erosion, spacy_tok, cysill_mut, ["h-mutation"], surface_mut)

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
        if norm_current in DEFINITE_ARTICLE_FORMS and i + 1 < len(words_list):
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
                i + 1 < len(words_list):
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
        if norm_current in NUMERAL_FEM_SOFT_LIMITED_TRIGGERS and i + 1 < len(words_list):
            row = _process_gender_trigger(
                current_node, words_list[i + 1], ["soft_limited"],
                norm_current, "numeral+fem_noun", "feminine", True)
            if row:
                mutation_rows.append(row)
                i += 2
                continue

        # ---- Layer 1E: dau/dwy + any noun → soft ----
        if norm_current in NUMERAL_GENERAL_SOFT_TRIGGERS and i + 1 < len(words_list):
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
        if norm_current in NASAL_NUMERAL_TRIGGERS and i + 1 < len(words_list):
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
        if is_fem_ordinal and i + 1 < len(words_list):
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
                and i + 1 < len(words_list)):
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
        if norm_current in COMPOUND_NOUN_SECOND_ELEMENT and i + 1 < len(words_list):
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

        # ---- Layer 1I: spaCy vocative → soft ----
        if spacy_tok and spacy_tok.get("dep") == "vocative":
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