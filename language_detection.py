"""
language_detection.py
======================
Language identification for the news scraper, merged from what used to
be two separate files (language_judges.py and language_voices.py) --
kept apart longer than necessary; the panel (judges) and its tiebreaker
(the Gemini voice) are one decision pipeline, always used together, and
splitting them was adding a file to track without real isolation
benefit -- unlike gemini_retry.py and boilerplate.py, which genuinely
serve independent callers and would be WORSE merged (see boilerplate.py's
own docstring for that reasoning).

PART 1 -- THE LOCAL JUDGE PANEL (was language_judges.py):

language_judges.py
===================
Replaces the old two-voice (lingua local / Gemini API) language ID system
with a weighted panel of LOCAL, offline judges -- no API calls, no RPM
ceiling, no key rotation to think about. Gemini is kept, but demoted to
a rare tiebreaker: it's only ever called for articles the local panel
itself can't resolve, which is the actual fix for the RPM-pressure
problem raised earlier, not just a workaround for it.

Why three judges instead of one
--------------------------------
GlotLID (cis-lmu/glotlid on HuggingFace) is the anchor: a fastText model
covering 2000+ ISO 639-3 languages, including everything on this
project's radar (Icelandic, Dutch, Danish, Swedish, Norwegian, Welsh,
Irish, AND Kalaallisut/Greenlandic -- none of which lingua's closed
candidate list could represent, and one of which, Greenlandic, lingua's
underlying library doesn't support at ANY size list). It's the reason
this module exists at all.

But one judge, however broad its coverage, is still one opinion. The
other two each cover a real, different failure mode GlotLID alone
doesn't:

  - OpenLID-v3 (HPLT/OpenLID-v3) -- GlotLID's own paper reports it wins
    on recall but OpenLID wins on PRECISION and false-positive rate, and
    recommends ensembling the two. Where GlotLID casts the widest net,
    OpenLID is the one likelier to catch a GlotLID false positive.
  - lingua -- already in this codebase, already proven (see
    icelandic_text_extractor.py's own comment on the langdetect
    Icelandic/Norwegian mix-up) to be solid WITHIN its closed candidate
    list. Kept as a cheap corroborating opinion; extended here to include
    Dutch, Irish, Italian, and Portuguese (all in lingua's underlying
    75-language set).

(CLD3 / gcld3 was previously a fourth judge. Removed: no reliable Windows
wheel, and with only three strong local signals the panel is quieter
without sacrificing coverage from GlotLID + OpenLID.)

None of the three is trusted alone. See judge_language()'s docstring for
how the panel's votes turn into a decision vs. a dispute.

PART 2 -- THE GEMINI TIEBREAKER VOICE (was language_voices.py):

language_voices.py
===================
Gemini-backed language identification -- ONE call this module exposes,
detect_language_llm(), used by language_judges.py's judge_language()
ensemble strictly as a TIEBREAKER, not a coequal second voice the way it
used to be.

Old design (superseded): every article got a lingua call AND, if the
flag was on, a Gemini call, and the two were reconciled directly in the
main script. That meant Gemini's call volume scaled with total articles
scraped, which is exactly what ran into the Gemini API's per-minute
request ceiling (RPM binds far sooner than TPM does for a call this
small -- a couple hundred tokens per sample against a 5 RPM free-tier
cap, well before the 250K TPM cap is anywhere close).

Current design: language_judges.py runs a panel of three LOCAL, offline
judges (GlotLID, OpenLID-v3, lingua) first. Gemini, via this module, is
only ever invoked for the subset of articles that panel couldn't already
resolve on its own -- see judge_language() and reconcile_gemini_tiebreak()
in language_judges.py. That's what actually fixes the RPM pressure, not
just spreads the same per-article call rate out with retries/backoff.

Opt-in only (--detect-language-llm / the interactive-menu prompt), off
by default -- same reasoning as boilerplate_detector.py: worth paying
for specifically when the local panel is genuinely split on what you're
scraping, not something to burn API budget on by default.

Reuses the SAME cached Gemini client as boilerplate_detector.py
(get_gemini_client() there is a module-level singleton, so this doesn't
spin up a second client) and the same pinned model/retry constants, so
there's one place to bump the model string when Google ships a new Flash
generation, not two. Keeps its OWN quota-exhausted flag, though -- a
language-ID call and a boilerplate call are different requests, and
tripping one voice's rate limit shouldn't be assumed to silently disable
the other (if the underlying daily quota really is exhausted, both will
independently trip anyway within a call or two of each other).
"""

import os
import re
import json
import asyncio
from collections import namedtuple
from datetime import datetime, timezone
from urllib.parse import urlparse

from gemini_retry import GeminiRetryCaller
from boilerplate import (
    get_gemini_client,
    DETECTION_MODEL,
    MAX_RETRIES,
    INITIAL_BACKOFF_SECONDS,
    rotate_gemini_key,
    mark_gemini_key_exhausted,
    gemini_key_count,
    reset_gemini_key_pool,
)

# --------------------------------------------------------------------------
# Site-level language overrides (compact table -- field docs + per-domain
# rationale live in LANGUAGE_OVERRIDES_NOTES at the bottom of this file)
# --------------------------------------------------------------------------
# Keyed by domain (www.-stripped). Fields: expected_language, language_lock,
# listing_urls. Mode 3 only reads listing_urls from language_lock'd entries.
LANGUAGE_OVERRIDES = {
    "ruv.is": {
        "expected_language": "is",
        "language_lock": True,
        "listing_urls": {
            "is": ["https://www.ruv.is/frettir"],
        },
    },
    "bbc.com": {
        "expected_language": {"default": "en", "/cymrufyw/": "cy"},
        "language_lock": True,
        "listing_urls": {
            "en": ["https://www.bbc.com/news"],
            "cy": ["https://www.bbc.com/cymrufyw"],
        },
    },
    "apnews.com": {
        "expected_language": "en",
        "language_lock": True,
        "listing_urls": {
            "en": ["https://apnews.com/"],
        },
    },
    "theguardian.com": {
        # English-only per the SITE_OVERRIDES entry's own sample -- no
        # non-live-blog counterexample seen. /live/ pages are excluded
        # from SITE_OVERRIDES entirely (different template, not scraped
        # via this path), so nothing routes through this lock for them.
        # Listing URL is /international (Guardian's edition-switcher
        # default, as opposed to /uk, /us, /au) rather than /world --
        # neither has been run through actual auto-discovery yet, so
        # this is still an assumption, not a confirmed-working listing
        # page the way ruv.is/frettir or bbc.com/news are.
        "expected_language": "en",
        "language_lock": True,
        "listing_urls": {
            "en": ["https://www.theguardian.com/international"],
        },
    },
    "tagesschau.de": {
        "expected_language": "de",
        "language_lock": True,
        "listing_urls": {
            "de": ["https://www.tagesschau.de/"],
        },
    },
    "dw.com": {
        "expected_language": "de",
        "language_lock": True,
        "listing_urls": {
            "de": ["https://www.dw.com/de/"],
        },
    },
    "nrk.no": {
        "expected_language": "nb",
        "language_lock": True,
        "listing_urls": {
            "nb": ["https://www.nrk.no/", "https://www.nrk.no/nyheter/"],
        },
    },
    "nos.nl": {
        "expected_language": "nl",
        "language_lock": True,
        "listing_urls": {
            "nl": ["https://nos.nl/", "https://nos.nl/nieuws"],
        },
    },
    "ansa.it": {
        "expected_language": "it",
        "language_lock": True,
        "listing_urls": {
            "it": ["https://www.ansa.it/"],
        },
    },
    "rtve.es": {
        "expected_language": "es",
        "language_lock": True,
        "listing_urls": {
            "es": ["https://www.rtve.es/noticias/"],
        },
    },
    "rtp.pt": {
        "expected_language": "pt",
        "language_lock": True,
        "listing_urls": {
            "pt": ["https://www.rtp.pt/noticias/"],
        },
    },
    "svt.se": {
        "expected_language": "sv",
        "language_lock": True,
        "listing_urls": {
            "sv": ["https://www.svt.se/", "https://www.svt.se/nyheter/"],
        },
    },
    "lefigaro.fr": {
        "expected_language": "fr",
        "language_lock": True,
        "listing_urls": {
            "fr": ["https://www.lefigaro.fr/"],
        },
    },
    "france24.com": {
        "expected_language": "fr",
        "language_lock": True,
        "listing_urls": {
            "fr": ["https://www.france24.com/fr/"],
        },
    },
    "newyddion.s4c.cymru": {
        "expected_language": "cy",
        "language_lock": True,
        "listing_urls": {
            # ORDER MATTERS -- discovered_via credits the first listing that
            # links an article. Topic pages (10 newest each, labels from the
            # homepage nav, /topics/3 loaded and verified) come first so
            # those articles carry S4C's own topic label; the sitemap (verified
            # leaf, ~500 items over a rolling month) comes last and covers the
            # rest, which get discovered_via = the sitemap. Topic links use
            # /article/cy/<slug> and the sitemap uses /article/<id>, so each
            # topic-page article is fetched twice; the second fetch is dropped
            # by content-hash dedup (~130 extra page loads per run -- the
            # price of the topic labels). Verified 2026-09-27 that the two
            # forms extract byte-identical ArticleBody text (a YesCymru piece:
            # /article/cy/<slug> vs /article/36862, 38 paras each), so the
            # dedup reliably fires rather than saving the article twice. Learners' section first, so learner
            # Welsh gets TAGGED rather than blended into another topic.
            "cy": [
                "https://newyddion.s4c.cymru/topics/22",  # Dysgu Cymraeg (learners)
                "https://newyddion.s4c.cymru/topics/3",   # Gwleidyddiaeth (politics)
                "https://newyddion.s4c.cymru/topics/25",  # Trosedd (crime)
                "https://newyddion.s4c.cymru/topics/6",   # Chwaraeon (sport)
                "https://newyddion.s4c.cymru/topics/20",  # Celfyddydau (arts)
                "https://newyddion.s4c.cymru/topics/19",  # Addysg (education)
                "https://newyddion.s4c.cymru/topics/4",   # Amaeth ac amgylchedd (farming/environment)
                "https://newyddion.s4c.cymru/topics/17",  # Iechyd (health)
                "https://newyddion.s4c.cymru/topics/18",  # Rhyngwladol (international)
                "https://newyddion.s4c.cymru/topics/5",   # Lleol (local)
                "https://newyddion.s4c.cymru/topics/30",  # Teyrngedau (tributes)
                "https://newyddion.s4c.cymru/topics/0",   # Newyddion (general news)
                "https://newyddion.s4c.cymru/topics/1",   # NEWYDDION (general news, 2nd nav entry)
                "https://newyddion.s4c.cymru/server-sitemap.xml",
            ],
        },
    },
    "ycymro.cymru": {
        "expected_language": "cy",
        "language_lock": True,
        "listing_urls": {
            # Sections before the homepage so discovered_via credits the
            # section (the homepage mixes news and opinion).
            "cy": [
                "https://ycymro.cymru/barn/",             # opinion section (homepage nav link)
                "https://ycymro.cymru/newyddion/",        # verified: 10 article links
                "https://ycymro.cymru/newyddion/page/2/",  # verified: linked from /newyddion/
                # 3-5 follow WordPress's /page/N/ convention, not individually
                # verified -- a page past the end just discovers nothing new.
                "https://ycymro.cymru/newyddion/page/3/",
                "https://ycymro.cymru/newyddion/page/4/",
                "https://ycymro.cymru/newyddion/page/5/",
                "https://ycymro.cymru/",
            ],
        },
    },
}


def _domain_of(url):
    return urlparse(url).netloc.replace("www.", "").lower()


def get_expected_language(url):
    """Best-effort site-level language PRIOR, sourced from this URL's
    LANGUAGE_OVERRIDES entry (if any) -- the idea being that a Dutch
    news site is never going to randomly publish a Cantonese article, so
    knowing the site is real corroborating signal for language ID, not
    just content analysis.

    Returns None if the domain has no override, or its override doesn't
    set "expected_language" -- both mean "no prior available, judge the
    text on its own merits", which is the existing behavior for every
    domain not listed here.

    IMPORTANT: this is a PRIOR, not a verdict, UNLESS the domain is also
    language_lock'd -- see is_language_locked() and LANGUAGE_OVERRIDES's
    own field comments above for the two different ways this value gets
    used downstream."""
    override = LANGUAGE_OVERRIDES.get(_domain_of(url))
    if not override:
        return None
    hint = override.get("expected_language")
    if hint is None:
        return None
    if isinstance(hint, str):
        return hint
    # Dict form: longest matching path prefix wins (most specific override
    # takes priority), "default" is the fallback for anything else on the
    # domain.
    path = urlparse(url).path
    best_prefix, best_code = None, hint.get("default")
    for prefix, code in hint.items():
        if prefix != "default" and path.startswith(prefix):
            if best_prefix is None or len(prefix) > len(best_prefix):
                best_prefix, best_code = prefix, code
    return best_code


def is_language_locked(url):
    """True if this URL's domain has language_lock: True in its
    LANGUAGE_OVERRIDES entry -- see that field's own comment above for
    what locking means and why it's a materially bigger trust step than
    the plain expected_language prior.

    Deliberately does NOT check whether get_expected_language(url)
    actually returns a code -- that's the caller's job (detect_language()
    in the main script), so it can tell "locked but this specific path
    isn't mapped" (falls through to the full panel) apart from "not
    locked at all" (same fallback, but for a different reason) without
    this function blurring the two."""
    override = LANGUAGE_OVERRIDES.get(_domain_of(url))
    return bool(override and override.get("language_lock"))


def known_locked_languages():
    """Every language code reachable via the "scrape by language" wizard
    mode -- the union of listing_urls' keys across every language_lock'd
    domain in LANGUAGE_OVERRIDES. Used purely for the error message when
    a user types a code nothing is configured for; not used to VALIDATE
    codes (an unrecognized code is still a legitimate thing to type, it
    just currently resolves to zero URLs -- see
    resolve_listing_urls_for_languages())."""
    codes = set()
    for override in LANGUAGE_OVERRIDES.values():
        if override.get("language_lock"):
            codes.update(override.get("listing_urls", {}).keys())
    return sorted(codes)


def resolve_listing_urls_for_languages(codes):
    """codes: iterable of lowercase 2-letter language codes (as typed by
    the user in the "scrape by language" wizard mode). Returns a flat,
    de-duplicated list of listing/section URLs to auto-discover from,
    pooled across every language_lock'd LANGUAGE_OVERRIDES domain that
    serves any of the requested codes.

    Deliberately domain-agnostic from the caller's point of view -- if
    two locked domains both serve, say, "is", both domains' listing URLs
    come back together in one list; run_auto() already handles multiple
    listing pages in a single call, discovering articles from each in
    turn, so this doesn't need its own per-domain looping.

    Only ever draws from language_lock'd domains ON PURPOSE: this mode
    exists to skip manual URL entry, which only makes sense to offer for
    domains this project is already willing to trust without per-
    article inspection. A domain with only a plain (non-locked)
    expected_language prior isn't included here -- picking it by
    language code alone would be promising more certainty than that
    prior actually claims to have. Extend a domain's listing_urls (and
    set language_lock: True) if you want it selectable this way."""
    codes = set(codes)
    urls = []
    seen = set()
    for override in LANGUAGE_OVERRIDES.values():
        if not override.get("language_lock"):
            continue
        for code, code_urls in override.get("listing_urls", {}).items():
            if code not in codes:
                continue
            for u in code_urls:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
    return urls

# --------------------------------------------------------------------------
# Judge weights and abstain thresholds
# --------------------------------------------------------------------------
# Base weight reflects how much this judge's vote should count RELATIVE
# TO THE OTHERS, not an absolute confidence -- the actual per-article
# vote strength is base_weight * this_judge's_own_confidence_on_THIS_text,
# so a normally-strong judge that's unsure on a given article still
# contributes less than a normally-weaker judge that's very sure.
JUDGE_WEIGHTS = {
    "glotlid": 1.0,   # broadest coverage, best recall per its own paper
    "openlid": 1.0,   # best precision/FPR -- meant to pair with glotlid
    "lingua":  0.6,   # cheapest/oldest signal here, closed candidate
                       # list, documented history of confusing close
                       # Nordic-language pairs -- kept for corroboration,
                       # not as a primary voice
    "gemini":  0.9,   # tiebreak-only (see reconcile_gemini_tiebreak
                       # below) -- a strong general model but not a
                       # calibrated LID specialist; sits between lingua
                       # and the two dedicated fastText judges
    "site_hint": 0.5, # a domain-level language PRIOR (see
                       # get_expected_language() -- e.g. ruv.is is never
                       # anything but Icelandic), NOT a judge that
                       # actually looked at this specific text. Weighted
                       # below every real judge on purpose: strong
                       # enough to corroborate a lone local judge (which
                       # is the whole point -- see judge_language()'s
                       # site_hint_code parameter), never strong enough
                       # to outvote two real judges that agree with each
                       # other against it.
}

# Confidence value used for the synthetic site_hint vote above -- not a
# per-text confidence (there's nothing to measure confidence ABOUT, it's
# metadata, not a model output), just a fixed "how much do we trust site-
# level priors in general" constant. Combined with JUDGE_WEIGHTS
# ["site_hint"] above, this is deliberately calibrated so that ONE real
# local judge (weight >= 0.6) agreeing with the hint clears
# CONSENSUS_SHARE_THRESHOLD together, but the hint alone (with every real
# judge abstaining) never reaches MIN_CORROBORATING_JUDGES on its own --
# see judge_language()'s `if votes:` guard, which only ever injects this
# vote alongside at least one real judge's vote, never as a substitute
# for one.
SITE_HINT_CONFIDENCE = 0.65

# fastText's own `threshold` predict() argument -- below this, predict()
# returns nothing at all for that call, which this module treats as an
# abstain rather than forcing a low-confidence guess into the vote.
# OpenLID's own model card usage example already threshold=0.5s its
# calls; GlotLID gets a slightly lower floor since its 2000+-language
# coverage means peak confidence is naturally more spread out even on
# genuinely correct calls (fastText fully wrong-language guesses do
# still tend to sit well below either floor in the fastText paper's own
# reporting, so this isn't just "less thorough", it's calibrated to how
# each model's confidence distribution actually looks).
GLOTLID_THRESHOLD = 0.3
OPENLID_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Ensemble resolution thresholds
# --------------------------------------------------------------------------
# A winner needs a clear lean, not just a technical plurality -- 0.55
# means the winning code has to hold more than half the total weight
# among judges who actually voted, not just more than any single
# runner-up. This is deliberately independent of HOW the weight got
# there: a landslide among 2 judges and a narrow win among 3 both have
# to clear the same bar. Weighting decides WHO's ahead; this threshold
# decides whether "ahead" is convincing enough to act on. A weighted
# vote with a weak margin is still a real disagreement underneath --
# see the module docstring's opening paragraph on why this project
# wanted that distinction preserved rather than papered over.
CONSENSUS_SHARE_THRESHOLD = 0.55

# However confident a single judge is, one opinion is not corroboration.
# This is the paranoia knob: raising it demands agreement from more
# independent judges before anything gets auto-filed; MIN_CORROBORATING
# _JUDGES = 1 would let GlotLID (or any one judge) decide alone whenever
# everyone else abstains, which defeats the point of building a panel.
MIN_CORROBORATING_JUDGES = 2

DISPUTES_FILENAME = "language_disputes.json"

# The three local judges that ALWAYS get a chance to vote in judge_language()
# -- used at logging time to work out who abstained, since verdict.votes
# only ever contains judges that DID vote (abstaining judges are already
# excluded there by design -- see judge_language()'s own docstring). This
# list is what makes "nobody voted" and "everybody voted but disagreed"
# distinguishable in the dispute log itself, not just in the live verdict
# object.
LOCAL_JUDGE_NAMES = ("glotlid", "openlid", "lingua")

JudgeVote = namedtuple("JudgeVote", ["code", "confidence"])

LanguageVerdict = namedtuple("LanguageVerdict", [
    "code",           # final ISO code to file under, or None if disputed/unknown
    "disputed",       # True if the panel couldn't reach consensus
    "votes",          # dict: judge name -> JudgeVote, only judges that didn't abstain
    "winner_code",    # the panel's best guess even when disputed (for logging)
    "winner_share",   # winner's weight / total participating weight (for logging)
])

# --------------------------------------------------------------------------
# ISO 639-3 (+ script) -> ISO 639-1 normalization
# --------------------------------------------------------------------------
# GlotLID and OpenLID both label with three-letter ISO codes plus a
# script tag (e.g. "nld_Latn", "kal_Latn") -- this project's folder
# layout and Stanza lemmatizer calls expect the shorter two-letter codes
# lingua already speaks. This table is a closed list on purpose,
# same philosophy as _LINGUA_LANGUAGES in the main script: covers what's
# actually been scraped so far, extend it here first if a new language
# starts showing up. Anything NOT in this table falls back to keeping
# the raw three-letter code as the folder name instead of forcing it
# into UNKNOWN_LANG_FOLDER -- GlotLID's whole value is covering languages
# far outside this project's current list, and a real (if unmapped) ISO
# code is still far more useful as a folder name than "unknown" would be.
ISO_639_3_TO_1 = {
    "isl": "is", "eng": "en", "cym": "cy", "nob": "nb", "nno": "nn",
    "swe": "sv", "dan": "da", "deu": "de", "fra": "fr", "spa": "es",
    "nld": "nl", "gle": "ga", "kal": "kl",
    "ita": "it", "por": "pt",
}


def _normalize_iso639_3_label(raw_label):
    """'__label__nld_Latn' -> 'nl' (or 'nld' if not in ISO_639_3_TO_1)."""
    code3 = raw_label.replace("__label__", "").split("_")[0].lower()
    return ISO_639_3_TO_1.get(code3, code3)


# --------------------------------------------------------------------------
# Individual judges -- each returns a JudgeVote or None (abstain: package/
# model unavailable, or this judge's own confidence didn't clear its
# floor). Every loader is lazy + cached + wrapped so one judge failing to
# load (no internet for the model download, missing package, etc.) never
# takes down the others -- same "losing one voice doesn't take down the
# other" principle language_voices.py already established for the old
# two-voice system, just extended to three judges instead of two.
# --------------------------------------------------------------------------

_glotlid_model = None
_glotlid_load_failed = False


def _load_glotlid():
    global _glotlid_model, _glotlid_load_failed
    if _glotlid_model is not None or _glotlid_load_failed:
        return _glotlid_model
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        _glotlid_model = fasttext.load_model(model_path)
    except Exception as e:
        _glotlid_load_failed = True
        print(f"⚠️  GlotLID unavailable ({e}) -- this judge will abstain for "
              "the rest of the run. (Needs internet access to download "
              "cis-lmu/glotlid's ~1.7GB model.bin the first time, and the "
              "fasttext + huggingface_hub packages installed.)")
    return _glotlid_model


def judge_glotlid(text):
    model = _load_glotlid()
    if model is None:
        return None
    sample = text[:2000].replace("\n", " ")
    labels, probs = model.predict(sample, k=1, threshold=GLOTLID_THRESHOLD)
    if not labels:
        return None  # below GLOTLID_THRESHOLD -- treated as abstain, not a guess
    return JudgeVote(_normalize_iso639_3_label(labels[0]), float(probs[0]))


_openlid_model = None
_openlid_load_failed = False


def _load_openlid():
    global _openlid_model, _openlid_load_failed
    if _openlid_model is not None or _openlid_load_failed:
        return _openlid_model
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="HPLT/OpenLID-v3", filename="openlid-v3.bin")
        _openlid_model = fasttext.load_model(model_path)
    except Exception as e:
        _openlid_load_failed = True
        print(f"⚠️  OpenLID-v3 unavailable ({e}) -- this judge will abstain "
              "for the rest of the run. (Needs internet access to download "
              "HPLT/OpenLID-v3's ~1.2GB model the first time.)")
    return _openlid_model


_OPENLID_NONWORD_RE = re.compile(r"[^\w\s]|\d", re.UNICODE)
_OPENLID_SPACE_RE = re.compile(r"\s\s+")


def _openlid_preprocess(text):
    """Mirrors OpenLID-v3's own documented preprocessing (lowercase,
    flatten newlines, strip digits/punctuation, squeeze whitespace) --
    the model card's own usage example applies this before predicting,
    and skipping it measurably lowers confidence (see the OpenLID-v2
    card's own before/after example: 0.81 with cleaning vs. lower
    without, on the exact same sentence)."""
    text = text.strip().replace("\n", " ").lower()
    text = _OPENLID_NONWORD_RE.sub("", text)
    return _OPENLID_SPACE_RE.sub(" ", text)


def judge_openlid(text):
    model = _load_openlid()
    if model is None:
        return None
    sample = _openlid_preprocess(text[:2000])
    labels, probs = model.predict(sample, k=1, threshold=OPENLID_THRESHOLD)
    if not labels:
        return None
    return JudgeVote(_normalize_iso639_3_label(labels[0]), float(probs[0]))


def judge_lingua(text, lingua_detector):
    """lingua_detector is passed in rather than built here -- it's already
    built once at import time in icelandic_text_extractor.py (adding
    Dutch and Irish to that build list is a separate, one-line fix; see
    the conversation this module came out of), and there's no reason to
    duplicate that detector or its startup cost."""
    if lingua_detector is None:
        return None
    sample = text[:2000]
    result = lingua_detector.detect_language_of(sample)
    if result is None:
        return None  # lingua's own internal ambiguity gate -- verified by
                      # hand that this returns None on genuinely unusable
                      # input rather than forcing a low-confidence guess
    confidence_values = lingua_detector.compute_language_confidence_values(sample)
    top_confidence = confidence_values[0].value if confidence_values else 0.0
    return JudgeVote(result.iso_code_639_1.name.lower(), top_confidence)


# --------------------------------------------------------------------------
# Ensemble resolution
# --------------------------------------------------------------------------

def _resolve_votes(votes):
    """votes: dict of judge name -> JudgeVote (abstaining judges already
    excluded by the caller). Returns (winner_code, winner_share,
    disputed) -- pure function, no I/O, so it's cheap to re-run after a
    Gemini tiebreak vote gets added to the same dict.

    BUGFIX: corroboration used to be judged by `len(votes) <
    MIN_CORROBORATING_JUDGES` -- i.e. "did at least 2 judges vote at
    all", not "did at least 2 judges agree with the winner". Since
    site_hint is injected into this same votes dict, that let a real
    judge + a site-level language PRIOR (a metadata assumption, not an
    actual read of this article's text) jointly outvote a SECOND real
    judge that was actively voting for a different language -- e.g.
    glotlid="en" + site_hint="en" clearing CONSENSUS_SHARE_THRESHOLD
    against lingua="cy" got filed as confident consensus, even though
    two real judges genuinely disagreed with each other. site_hint's
    only documented job (see its own comment above) is to turn a single
    real judge's LONE vote into the two corroborating votes
    MIN_CORROBORATING_JUDGES asks for when nothing else voted at all --
    it was never meant to be able to overrule a second real, disagreeing
    judge, or to win on its own when the one real judge present actually
    disagrees with it. Fixed by counting corroboration among REAL
    (non-site_hint) judges specifically; site_hint only ever completes
    the count when it's the sole other participant AND it agrees with
    that one real judge -- matching the "doesn't file off site metadata
    alone" principle already stated in judge_language()'s docstring."""
    if not votes:
        return None, 0.0, True

    weighted = {}
    for judge_name, vote in votes.items():
        weighted[vote.code] = weighted.get(vote.code, 0.0) + JUDGE_WEIGHTS[judge_name] * vote.confidence

    total_weight = sum(weighted.values())
    winner_code, winner_weight = max(weighted.items(), key=lambda kv: kv[1])
    winner_share = (winner_weight / total_weight) if total_weight > 0 else 0.0

    real_votes = {name: v for name, v in votes.items() if name != "site_hint"}
    real_voters_for_winner = sum(1 for v in real_votes.values() if v.code == winner_code)
    site_hint_backs_winner = votes.get("site_hint", JudgeVote(None, 0.0)).code == winner_code

    corroborated = (
        real_voters_for_winner >= MIN_CORROBORATING_JUDGES
        or (len(real_votes) == 1 and real_voters_for_winner == 1 and site_hint_backs_winner)
    )

    disputed = (not corroborated) or (winner_share < CONSENSUS_SHARE_THRESHOLD)
    return winner_code, winner_share, disputed


def judge_language(text, lingua_detector, site_hint_code=None):
    """Runs all three local judges (Gemini is NOT called here -- see
    reconcile_gemini_tiebreak below) and resolves their votes.

    site_hint_code: optional ISO 639-1 code from the site's own
    SITE_OVERRIDES["expected_language"] entry (see
    icelandic_text_extractor.py's get_expected_language()). A Dutch news
    site is never going to randomly publish a Cantonese article, so
    knowing the site is real corroborating signal -- injected as one
    extra synthetic vote (JUDGE_WEIGHTS["site_hint"], SITE_HINT_
    CONFIDENCE), but ONLY when at least one real local judge already
    voted (see the `if votes:` guard below). That guard is doing real
    work: it means this can turn ONE confident local judge's lone vote
    into the two needed for MIN_CORROBORATING_JUDGES (the actual payoff
    -- resolving locally instead of falling to a Gemini tiebreak or
    staying disputed), but it can never single-handedly resolve a text
    that every real judge abstained on. This project doesn't file an
    article's language off site metadata alone with zero actual content
    inspection, on principle.

    Returns a LanguageVerdict. Three outcomes:

      1. No judge had anything to say at all (every model failed to
         load, or the text was too short/garbled for all three) ->
         code=None, disputed=False. This is the old UNKNOWN_LANG_FOLDER
         case, distinct from a genuine dispute: nobody voted, so there's
         nothing to disagree ABOUT.
      2. The panel reached consensus (a code holds >= CONSENSUS_SHARE_
         THRESHOLD of the participating weight, from >= MIN_CORROBORATING
         _JUDGES judges) -> code=<that code>, disputed=False.
      3. Real disagreement, OR only one judge managed to vote at all ->
         code=None, disputed=True. winner_code/winner_share are still
         filled in (the panel's best guess and how weak its lean was)
         so the caller can log a useful dispute record even though it's
         not being trusted to file the article under that language.
    """
    votes = {}
    for judge_name, vote in (
        ("glotlid", judge_glotlid(text)),
        ("openlid", judge_openlid(text)),
        ("lingua", judge_lingua(text, lingua_detector)),
    ):
        if vote is not None:
            votes[judge_name] = vote

    if not votes:
        return LanguageVerdict(code=None, disputed=False, votes={}, winner_code=None, winner_share=0.0)

    if site_hint_code:
        votes["site_hint"] = JudgeVote(site_hint_code, SITE_HINT_CONFIDENCE)

    winner_code, winner_share, disputed = _resolve_votes(votes)
    return LanguageVerdict(
        code=None if disputed else winner_code,
        disputed=disputed,
        votes=votes,
        winner_code=winner_code,
        winner_share=winner_share,
    )


def reconcile_gemini_tiebreak(verdict, gemini_result):
    """Only called by the caller when verdict.disputed is True AND the
    --detect-language-llm flag is on -- this is the whole point of
    demoting Gemini to a tiebreaker: it's invoked ONLY for the subset of
    articles the local panel couldn't already resolve on its own, not
    per-article, which is the actual fix for the RPM-pressure problem
    this module was built to address, not just a smaller version of it.

    gemini_result: whatever language_voices.py's detect_language_llm()
    returned -- a (code, confidence) tuple (code already lowercase
    2-letter; confidence self-reported by the model, falling back to
    GEMINI_CONFIDENCE_FALLBACK if missing/invalid -- see that module),
    or None if the API voice had nothing to say (quota exhausted,
    transient failure, etc).

    Returns a new LanguageVerdict with Gemini's vote folded in and the
    dispute re-resolved. If Gemini's vote is enough to push a code over
    CONSENSUS_SHARE_THRESHOLD, the dispute resolves; if not (Gemini
    agrees with nobody, or the panel was split three-plus ways), it
    stays disputed -- adding one more opinion doesn't manufacture
    consensus that isn't there, on purpose."""
    if gemini_result is None:
        return verdict  # API voice had nothing to add -- unchanged

    gemini_code, gemini_confidence = gemini_result
    votes = dict(verdict.votes)
    votes["gemini"] = JudgeVote(gemini_code, gemini_confidence)  # weighted by
                                                     # the model's OWN
                                                     # self-reported
                                                     # confidence, same as
                                                     # every local judge --
                                                     # previously this
                                                     # hardcoded 1.0, so
                                                     # Gemini always voted
                                                     # at its full 0.9
                                                     # weight regardless of
                                                     # how sure it actually
                                                     # was, quietly
                                                     # outweighing local
                                                     # judges reporting
                                                     # their genuine,
                                                     # usually-sub-1.0
                                                     # confidence
    winner_code, winner_share, disputed = _resolve_votes(votes)
    return LanguageVerdict(
        code=None if disputed else winner_code,
        disputed=disputed,
        votes=votes,
        winner_code=winner_code,
        winner_share=winner_share,
    )


# --------------------------------------------------------------------------
# Dispute logging
# --------------------------------------------------------------------------

def log_language_dispute(output_dir, url, verdict):
    """Appends a full per-judge vote breakdown to language_disputes.json,
    the direct successor to the old language_disagreements.json --
    renamed since this is no longer a two-voice disagreement log, it's a
    full panel breakdown. Deduped by URL, same as before.

    Now also records WHICH local judges abstained (voted for nothing),
    not just which ones voted -- "abstained" and "voted but lost" are
    different problems (a judge that never loads vs. a judge that loads
    fine but keeps disagreeing), and previously only the second was
    visible in this log. gemini isn't included in LOCAL_JUDGE_NAMES since
    it's only ever invoked conditionally (the --detect-language-llm flag,
    and only as a tiebreak) -- its absence from votes doesn't mean
    "abstained" the same way a local judge's absence does, so it's left
    out of the abstained list rather than logged misleadingly.

    Returns True if a new entry was added, False if this URL was already
    logged."""
    path = os.path.join(output_dir, DISPUTES_FILENAME)

    existing = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    if any(isinstance(e, dict) and e.get("url") == url for e in existing):
        return False

    abstained = [name for name in LOCAL_JUDGE_NAMES if name not in verdict.votes]

    existing.append({
        "url": url,
        "domain": urlparse(url).netloc.replace("www.", ""),
        "votes": {name: {"code": v.code, "confidence": v.confidence} for name, v in verdict.votes.items()},
        "abstained": abstained,
        "best_guess": verdict.winner_code,
        "best_guess_share": round(verdict.winner_share, 3),
        "logged_at": datetime.now(timezone.utc).isoformat(),
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    return True


# ============================================================
# PART 2 -- THE GEMINI TIEBREAKER VOICE (was language_voices.py)
# ============================================================

# Same slice detect_language() (the local lingua voice, in the main
# script) uses for its own sample. Keeping both voices looking at
# exactly the same stretch of text is what makes a disagreement between
# them meaningful -- if the API voice saw more of the article than the
# local one, a "disagreement" could just be that, not an actual split
# opinion on the same evidence.
MAX_SAMPLE_CHARS = 2000

_LANGUAGE_SYSTEM_PROMPT = """You identify the language of a piece of news-article text.

Respond with ONLY a JSON object with exactly two keys:
  "language_code": the two-letter ISO 639-1 code for the DOMINANT language of the text (lowercase, e.g. "is", "en", "cy")
  "confidence": a number from 0.0 to 1.0 for how confident you are that this code is correct. Use a LOW number (below 0.5) if the text is short, mixed-language, boilerplate-heavy, or otherwise ambiguous -- do not default to a high number out of habit.

If the text is genuinely mixed-language with no single dominant language, pick whichever language makes up more of the text and lower your confidence accordingly. If you cannot identify the language at all, use "unknown" as the language_code and 0.0 as the confidence.

Example response: {"language_code": "is", "confidence": 0.9}"""

# Gemini isn't a calibrated LID specialist the way the local judges are
# (see language_judges.py's own comment on why its base weight sits
# between lingua and the two fastText models), and its self-reported
# confidence field above is a best-effort ask, not a guaranteed-accurate
# probability. This floor is what a genuinely unremarkable, "yeah
# probably" call should land at -- used whenever the model's own
# confidence value is missing, malformed, or (as LLM self-reported
# confidence tends to run) implausibly pinned at the extremes. Previously
# this path just hardcoded 1.0 for every tiebreak vote regardless of how
# sure the model actually sounded, which meant Gemini's real contribution
# to a tiebreak (weight x confidence) was always its full 0.9 weight --
# quietly outweighing local judges who report their genuine, usually-
# sub-1.0 confidence on the same vote.
GEMINI_CONFIDENCE_FALLBACK = 0.6

_language_caller = GeminiRetryCaller(
    label="the LLM language voice",
    get_client=get_gemini_client,
    rotate_key=rotate_gemini_key,
    mark_exhausted=mark_gemini_key_exhausted,
    key_count=gemini_key_count,
    max_retries=MAX_RETRIES,
    initial_backoff_seconds=INITIAL_BACKOFF_SECONDS,
    fallback_note="Language detection keeps going on the local lingua "
                   "voice alone, just without a second opinion.",
)  # separate instance (own quota_exhausted flag) from
   # boilerplate_detector.py's caller by design -- see module docstring.
   # Both still route through the SAME underlying key pool via the
   # get_client/rotate_key/etc. callables above.


def reset_language_quota_flag():
    """Call at the start of each top-level run (run_auto/run_url_mode)
    before any scraping begins, mirroring boilerplate_detector.py's
    reset_quota_flag() -- so a previous run's rate-limit trip doesn't
    silently carry over into a fresh one in the same interactive-menu
    session. Also resets the SHARED key-exhaustion pool (via
    reset_gemini_key_pool(), not just this caller's own flag, since this
    flag and boilerplate's are intentionally independent -- see module
    docstring) so a previous run's rotated-through keys get another
    chance too."""
    _language_caller.reset()
    reset_gemini_key_pool()


def _parse_language_response(raw_response_text):
    """Strips ```json fences if the model added any despite JSON mode
    being requested, then parses and validates. Returns None (not a
    raise) on anything malformed or on the model's own "unknown" --
    either way the API voice has nothing usable to contribute, and the
    caller falls back to the local voice alone.

    Returns (code, confidence) on success. confidence falls back to
    GEMINI_CONFIDENCE_FALLBACK if the field is missing, non-numeric, or
    outside [0.0, 1.0] -- a malformed confidence field shouldn't sink an
    otherwise-valid language code, but it also shouldn't be silently
    trusted as 1.0 just because it round-tripped through JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response_text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    code = str(data.get("language_code", "")).strip().lower()
    if not re.fullmatch(r"[a-z]{2}", code):
        return None  # covers "unknown" (3+ letters) and anything malformed

    raw_confidence = data.get("confidence")
    try:
        confidence = float(raw_confidence)
        if not (0.0 <= confidence <= 1.0):
            confidence = GEMINI_CONFIDENCE_FALLBACK
    except (TypeError, ValueError):
        confidence = GEMINI_CONFIDENCE_FALLBACK

    return code, confidence


def _call_gemini_with_retry(sample):
    return _language_caller.call(
        DETECTION_MODEL, sample, _LANGUAGE_SYSTEM_PROMPT, max_output_tokens=50,
    )


def detect_language_llm_sync(text):
    """Blocking call -- run via asyncio.to_thread() from the async scrape
    path so it doesn't stall the Playwright event loop. Returns a
    (lowercase 2-letter ISO 639-1 code, confidence) tuple, or None if the
    API voice couldn't produce one (quota exhausted, transient failure,
    or the model itself said "unknown") -- None means "no second opinion
    available", not "the article has no language", so callers should
    fall back to the local voice rather than treating None as a real
    answer."""
    get_gemini_client()  # fail fast if package/key(s) missing
    sample = text[:MAX_SAMPLE_CHARS]
    response = _call_gemini_with_retry(sample)
    if response is None:
        return None
    raw = getattr(response, "text", None)
    if not raw:
        return None
    return _parse_language_response(raw)


async def detect_language_llm(text):
    """Async wrapper. Returns a (code, confidence) tuple or None -- see
    detect_language_llm_sync(). Swallows get_gemini_client()'s RuntimeError
    (missing google-genai package or GEMINI_API_KEY) instead of letting it
    propagate -- this is only ever called as a tiebreaker for the subset
    of articles language_judges.py's local panel couldn't already
    resolve, so a missing/misconfigured Gemini client should just mean
    "the dispute stays a dispute, filed under DISPUTED_LANG_FOLDER" here,
    not fail the article scrape entirely."""
    try:
        return await asyncio.to_thread(detect_language_llm_sync, text)
    except Exception:
        return None


# ==========================================================================
# LANGUAGE_OVERRIDES_NOTES
# ==========================================================================
# Why this table lives here (not in SITE_OVERRIDES): extraction/discovery
# and language policy are separate features that both key off domain.
# Keeping language data in this module means language-only edits don't
# live in a file named after extraction.
#
# Fields
# ------
# expected_language  Prior only (site_hint vote, weight 0.5) unless locked.
#                    str  = whole domain is that code
#                    dict = path-prefix -> code; longest prefix wins;
#                           "default" is the fallback bucket
# language_lock      True = certainty: skip the panel, return mapped code
#                    from expected_language. Only for domains confirmed not
#                    to publish outside the map. Unmapped path on a locked
#                    domain still falls through to the panel.
# listing_urls       code -> [section/listing page URLs]. Mode 3 only reads
#                    these on language_lock'd entries (resolve_listing_
#                    urls_for_languages). Expand this to grow mode 3.
#
# Per-domain rationale
# --------------------
# ruv.is   National Icelandic broadcaster; never observed non-Icelandic.
#          Single-code lock. listing: top frettir index.
# bbc.com  Path-deterministic multi-language: /cymrufyw/ = cy, else en.
#          Locked because every path maps; add new prefixes before a third
#          language section lands in "default" as en by mistake. listing:
#          /news and /cymrufyw only (not every BBC section front).
# newyddion.s4c.cymru
#          S4C's Welsh-language news service. <html lang="cy">, no hreflang
#          alternates or English toggle found (checked 2026-09-27). Locked.
#          Includes learners' content (/topics/22 "DYSGU CYMRAEG") in the
#          sitemap -- simplified learner Welsh, which will skew register/
#          complexity proxies; can't be excluded by URL (numeric IDs).
#          Scraping /topics/22 first tags the most recent ~10 via
#          discovered_via; older learner articles reached only through
#          the sitemap stay untagged, so filter by title/content if needed.
# ycymro.cymru
#          Y Cymro, monthly national Welsh-language paper (WordPress).
#          Locked: homepage lang="cy"; a /barn/ column with an English
#          title ("CYMRU NO MORE?") turned out to be Welsh prose with an
#          embedded English quote -- code-switching, not English content.
#          /newyddion/ = news, /barn/ = opinion: url_topic_hint picks this
#          up as a free news-vs-opinion genre column.
#
# Checked and deliberately NOT added (2026-09-27):
# golwg.360.cymru, bro.360.cymru (whole 360.cymru network)
#          robots.txt blocks AI/dataset crawlers by name (CCBot, img2dataset,
#          Scrapy, GPTBot, ClaudeBot, ChatGPT-User...) site-wide; license.xml
#          (RSL) prohibits "train-ai" use. Generic User-agent: * rules only
#          block /safle/ and /fi/, so a non-training research scrape isn't
#          excluded by the letter of either -- but that's a judgement call
#          for the researcher (and their ethics process), not something to
#          add by default. Inspect with inspect_selectors.py yourself if you
#          decide to include it.
#
# LOCK TRADE-OFF (applies to every cy source above, bbc.com/cymrufyw too):
# a locked domain skips the language panel entirely, so its rows in
# corpus_metadata.jsonl get language_votes/language_winner_share = null.
# For Welsh specifically, winner_share on unlocked text is a plausible
# code-switching signal (see Y Cymro note above). To collect it, set
# language_lock: False -- expected_language still acts as a prior, most
# articles still file under cy/, but mode 3 will no longer offer the domain
# (it only uses locked entries), so scrape it via mode 1 with the listing
# URLs above instead.