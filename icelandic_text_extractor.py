"""
Generic News Article Scraper
=============================
Works against (almost) any news site, not just RUV.

Two modes:

  Mode 1 - "auto":  give it one or more LISTING/SECTION pages
                     (e.g. a homepage or a "/world" section page).
                     It crawls the page, discovers likely article
                     links automatically, and scrapes all new ones.

      python icelandic_text_extractor.py auto https://example.com/news https://example.com/sport

  Mode 2 - "url":   give it one or more specific ARTICLE URLs directly.
                     It scrapes exactly those, no discovery step.

      python icelandic_text_extractor.py url https://example.com/news/some-headline-slug

Common options:
  --output-dir DIR       where .txt files + manifest go (default ./news_corpus).
                          The raw extracted text always goes in
                          raw/<lang>/ -- and if --lemmatize is given, a
                          SEPARATE lemmatized copy is additionally saved in
                          lemmatized/<lang>/, filename suffixed '__lemma', so
                          both variants of the same article coexist rather
                          than the lemmatized one replacing the raw one (see
                          corpus_layout.py). <lang> is detected from the
                          article's own text -- not from which site/URL it
                          came from, so a mixed-language listing page sorts
                          itself correctly. Undetectable text lands in
                          raw/unknown/, panel disagreements in raw/disputed/.
                          The scraped_urls.txt manifest itself stays at the
                          top level, shared across all languages, since dedup
                          is corpus-wide.
  --lemmatize LANG_CODE  optional Stanza lemmatization, e.g. --lemmatize en --
                          ADDS a lemmatized copy alongside the raw one (see
                          above), it doesn't replace it. Leave off to only
                          save raw text.
                          Language detection for folder routing always runs
                          on the RAW text first, regardless of this flag.
  --delay SECONDS         pause between requests, default 1.5s (+jitter)

Requires:
  pip install playwright trafilatura lingua-language-detector tqdm
  playwright install chromium
  # only if you use --lemmatize:
  pip install stanza
"""

import os
import re
import sys
import hashlib
import random
import asyncio
import argparse
from html import unescape
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from tqdm import tqdm
import trafilatura

# Language detection for per-language output folders. Uses `lingua`
# rather than the more commonly-reached-for `langdetect`: tested against
# this project's actual RUV samples, langdetect consistently misread
# Icelandic as Norwegian/Swedish (confirmed on multiple real articles,
# not a one-off) -- it leans on character n-grams that overlap heavily
# with Icelandic's Nordic neighbors and evidently doesn't have enough
# signal for a smaller language like Icelandic to separate them
# reliably. lingua correctly identified all of them, including short
# ones, in the same test. Worth keeping in mind if this list ever needs
# to grow -- re-test any newly-added closely-related-language pair
# before trusting it, same caution either detector deserves.
try:
    from lingua import Language, LanguageDetectorBuilder
    _LINGUA_LANGUAGES = [
        Language.ICELANDIC, Language.ENGLISH, Language.WELSH,
        Language.BOKMAL, Language.NYNORSK, Language.SWEDISH, Language.DANISH,
        Language.GERMAN, Language.FRENCH, Language.SPANISH,
        # Added when lingua got folded into the multi-judge panel
        # (language_detection.py) -- both are in lingua's own 75-language
        # set, they just weren't in THIS project's build list before,
        # meaning NOS.nl (Dutch) articles were being locally misread as
        # whichever of the above lingua considered closest. Spot-checked
        # against real samples before trusting them, same caution this
        # comment already asks for on any newly-added language.
        Language.DUTCH, Language.IRISH,
        # Italian + Portuguese added 2026-07-31 when ANSA / RTP overrides
        # were introduced -- without these, lingua had no candidate for
        # either language and routinely fell back to English.
        Language.ITALIAN, Language.PORTUGUESE,
    ]
    _lingua_detector = LanguageDetectorBuilder.from_languages(*_LINGUA_LANGUAGES).build()
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    _lingua_detector = None
    print("⚠️  lingua not installed -- per-language output folders disabled, "
          "everything will land in 'unknown/' (pip install lingua-language-detector).")

DEFAULT_OUTPUT = os.path.join(os.getcwd(), "news_corpus")
MANIFEST_NAME = "scraped_urls.txt"
UNKNOWN_LANG_FOLDER = "unknown"
# Distinct from UNKNOWN_LANG_FOLDER: "unknown" means no judge in the
# language_detection.py panel had anything to say at all (all three failed
# to load, or the text was too short/garbled for every one of them).
# "disputed" means the OPPOSITE problem -- judges DID vote, they just
# didn't agree enough to trust. Keeping these separate matters for
# review: an "unknown" article is a coverage/extraction problem, a
# "disputed" one is a genuine language-identification ambiguity (or a
# language outside every judge's real competence, like Greenlandic
# sometimes will be -- see language_detection.py's module docstring).
DISPUTED_LANG_FOLDER = "disputed"

# Path fragments that almost never indicate an actual article, across
# most news CMSs (WordPress, custom, Drupal, etc.)
NON_ARTICLE_HINTS = [
    "/tag/", "/tags/", "/topic/", "/topics/", "/category/", "/categories/",
    "/author/", "/authors/", "/video/", "/videos/", "/photo/", "/photos/",
    "/gallery/", "/live/", "/subscribe", "/newsletter", "/about", "/contact",
    "/privacy", "/terms", "/search", "/rss", "/feed", "/login", "/signup",
    "/account", "/cart", "/advert",
]

STATIC_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf",
    ".mp4", ".mp3", ".css", ".js", ".zip", ".ico",
)

# --------------------------------------------------------------------------
# Per-site overrides
# --------------------------------------------------------------------------
# Keyed by domain (www.-stripped). Each entry can supply:
#   article_selector      - CSS selector for the real article body, tried
#                            BEFORE trafilatura/generic-container guessing
#                            on this domain. Falls through to the generic
#                            path if this selector matches nothing.
#   exclude_selectors      - selectors for junk to strip OUT of whatever
#                            article_selector matched, before extracting
#                            paragraph text (e.g. share widgets, related-
#                            content boxes living inside the container).
#   listing_path_prefixes  - if set, discover_article_links() treats any
#                            same-domain link whose PATH starts with one
#                            of these as an article, instead of running
#                            the generic hyphen-count/digit-ID heuristic.
#                            Needed for sites using opaque IDs in the URL
#                            (BBC's /news/articles/<id> has neither
#                            hyphens nor a long digit run, so the generic
#                            heuristic would silently discover zero BBC
#                            articles from a listing page without this).
#
# This table is extraction/discovery data ONLY (boilerplate detection's
# concern) -- language-related overrides (expected_language,
# language_lock, listing_urls) live in language_detection.py's own
# LANGUAGE_OVERRIDES table instead, keyed by the same domains but kept
# separate on purpose: boilerplate detection and language detection are
# two independent features that happen to both key off domain, and
# folding both features' data into one dict here would mean a language-
# only change lives in, and is reasoned about via, a file named after a
# different feature. See LANGUAGE_OVERRIDES's own header comment in
# language_detection.py for the language-side field list.
#
# Every entry below is evidence-backed from a real inspect_selectors.py
# run against a live article on that domain, not guessed -- see each
# comment for what was actually observed. If a site starts silently
# under- or mis-extracting, re-run the inspector against a fresh article
# on it before assuming the override itself has gone stale (templates do
# change; scoping to <article> plateaus at "identical to <main article>"
# rather than "guaranteed forever").
SITE_OVERRIDES = {
    "ruv.is": {
        # UPDATED after a second inspect_selectors.py run, this time
        # against an ordinary bylined news article (the first run was a
        # sports live-blog, which turned out to be a different template
        # -- see below):
        #
        #   'article'        -> 9 paragraphs, 1806 chars, paragraph #1 is
        #                        the reporter's byline ("Anna Lilja
        #                        Þórisdóttir") sitting as its own <p>.
        #   '.article-body'  -> 8 paragraphs, 1784 chars -- exactly the
        #                        same content MINUS that byline paragraph
        #                        (1806-1784=22 chars, and
        #                        len("Anna Lilja Þórisdóttir")==22 exactly,
        #                        so this is confirmed to be precisely the
        #                        byline dropped, nothing else lost).
        #
        # A byline name is a proper noun, not article prose -- worth
        # excluding structurally for a linguistic corpus rather than
        # leaving it in and filtering after the fact. '.article-body'
        # did NOT appear at all as a candidate on the earlier live-blog
        # URL (that template doesn't use the class), so this only
        # activates on the bylined-article template it was confirmed
        # against; live-blog pages fall through to the generic
        # trafilatura/'article'/'main' path exactly as before, since
        # extract_via_override() returns None when the selector matches
        # nothing rather than forcing it.
        "article_selector": ".article-body",
        "exclude_selectors": [],
        "listing_path_prefixes": None,
    },
    "bbc.com": {
        # inspect_selectors.py on a BBC News article: 'article' and
        # 'main article' both returned identical text (39 paragraphs);
        # bare 'main' added 5 extra paragraphs 'article' lacked (same
        # recirc-widget-inside-main pattern as RUV above). Nothing
        # suspicious flagged inside 'article' itself.
        #
        # inspect_selectors.py on a BBC Cymru Fyw article (also served
        # under bbc.com, distinguished only by path): 'article', 'main
        # article', and bare 'main' were all IDENTICAL (553 chars) --
        # no widget contamination on this one at all. 'article' still
        # covers it correctly, so one selector serves both sections.
        # Caveat: that Cymru Fyw sample was a short quiz-format page,
        # not a typical long-form story -- worth re-checking against an
        # ordinary Cymru Fyw article before fully trusting this on that
        # section specifically.
        "article_selector": "article",
        "exclude_selectors": [],
        # BBC's opaque IDs (/news/articles/c80ny93xlrvo,
        # /cymrufyw/erthyglau/c77g8jdmdkyo) have neither hyphens nor a
        # long digit run, so the generic looks_like_article() heuristic
        # would find zero BBC articles from any listing page. This is
        # the fix for that -- path PREFIX match instead.
        "listing_path_prefixes": ["/news/articles/", "/cymrufyw/erthyglau/"],
    },
    "lefigaro.fr": {
        # inspect_selectors.py on a real Le Figaro article: 'main' (8276
        # chars, 4 suspect groups) vs 'article'/'main article' (both
        # identical, 4643 chars, 2 suspect groups). The tool's own
        # clean-alt check only considers a fallback candidate if it has
        # ZERO suspects, so it defaulted to keeping 'main' + excluding --
        # overridden by hand to 'article' instead, since 'article's
        # suspect set (share, tag) is a strict subset of 'main's (nav,
        # share x3, related, tag). The ~3.6k char gap between them is
        # almost certainly the breadcrumb nav and the "Sur le même
        # thème" related-content carousel (other headlines/summaries),
        # not real article prose -- so 'article' loses nothing and
        # entirely avoids ever needing the unnarrowed [class*='related']
        # wildcard the 'main' path would've required (the same shape of
        # risk as independent.co.uk's Taboola widget leaking
        # off-language content into the corpus).
        "article_selector": "article",
        "exclude_selectors": ["[class*='share']", "[class*='tag']"],
        "listing_path_prefixes": None,  # not checked by this tool -- verify against a listing page separately
    },
    "france24.com": {
        # inspect_selectors.py on a real France24 article: 'main' and
        # "[role='main']" returned identical text (18 paragraphs, 4849
        # chars) -- same node, not two real candidates -- and there's no
        # <article> tag on this template at all, so unlike lefigaro.fr
        # there was no tighter alternative to weigh against 'main'.
        # 24 separate "[class*='tag']" matches looked alarming at first
        # but were checked by hand one-by-one: suggested-keyword chips
        # (gtm-add-suggested-tag), category/recirc badges some of them
        # empty (m-master-tag), and the "Mots-clés associés" keyword
        # footer (t-content__tags) -- three unrelated class families all
        # containing "tag", none of them sentence-shaped, so the
        # wildcard's inability to narrow to one token reflects genuine
        # heterogeneity rather than a risk of eating real prose.
        # [class*='share'] was a single unambiguous "Partager" button.
        # CAVEAT (not yet confirmed against a real scrape): several
        # m-master-tag matches were empty, suggesting a recirc/"also
        # happening" module lives inside 'main' -- if a future sample
        # shows stray unrelated-topic sentences in
        # boilerplate_candidates.json for this domain, re-run the
        # inspector, since that module might carry real headline text
        # on some articles without a tag-ish class to catch it.
        "article_selector": "main",
        "exclude_selectors": ["nav", "[class*='share']", "[class*='tag']"],
        "listing_path_prefixes": None,  # not checked by this tool -- verify against a listing page separately
    },
    "apnews.com": {
        # inspect_selectors.py 2026-07-30 on a Fauci/Senate article:
        #   main                 -> 46 paras, 11037 chars, 4 suspect groups
        #                           (Author-socialLinks, jw-related video
        #                           shelf, jw accessibility chrome,
        #                           StoryPage byline + PagePromo read-time)
        #   [class*='RichText']  -> 38 paras, 9320 chars (~84% of main),
        #                           only PagePromo-byline chips ("5 MIN READ")
        # Tool auto-kept main (RichText not fully clean). Overridden by
        # hand to RichText -- same call as an earlier AP sample noted in
        # inspect_selectors.py's CLEAN_ALTERNATIVE_MIN_RATIO comment
        # (RichText preferred over main when close in size). Narrow
        # exclude for the promo read-time widgets only; avoid broad
        # [class*='byline'] which would also hit real author bylines if
        # a template puts those inside RichText later.
        "article_selector": "[class*='RichText']",
        "exclude_selectors": ["[class*='PagePromo-byline']"],
        "listing_path_prefixes": ["/article/"],
    },
    "theguardian.com": {
        # inspect_selectors.py on an ordinary (non-live) article
        # ("Japan earthquake stories"): article / main article / main
        # all identical -- 18 paras, 4684 chars, zero suspects. Picked
        # 'article' as both the fullest match and the clean one.
        # NOTE: does NOT cover theguardian.com/*/live/* -- confirmed
        # different template (main pulls the whole rolling thread,
        # repeats a per-update share/timestamp footer + pagination nav)
        # on a live Trump/Fauci/Iran updates page. Deliberately left
        # unhandled -- live-blog register is out of scope for now.
        "article_selector": "article",
        "exclude_selectors": [],
        "listing_path_prefixes": None,
    },
    "tagesschau.de": {
        # inspect_selectors.py 2026-07-31 on a CSD/Mauritania investigativ piece:
        #   main          -> 29 paras, 7630 chars, nav/social + taglist
        #   article       -> 26 paras, 7564 chars (~99% of main), same suspects
        # Prefer article (tighter). Hand-narrowed tag match to .taglist only.
        "article_selector": "article",
        "exclude_selectors": ["nav", ".taglist"],
        "listing_path_prefixes": None,
    },
    "nrk.no": {
        # inspect_selectors.py 2026-07-31 on a Meta/Microsoft quarterly piece:
        #   article / [role='main']  -> 50 paras, 5986 chars, related-sidebar junk
        #   .article-body            -> 49 paras, 5812 chars (97 %), clean
        "article_selector": ".article-body",
        "exclude_selectors": [],
        "listing_path_prefixes": None,
    },
    "nos.nl": {
        # inspect_selectors.py 2026-07-31 on a Bolle Jos / Belgian court piece:
        #   main -> 11 paras, 2591 chars, clean. No other candidates matched.
        "article_selector": "main",
        "exclude_selectors": [],
        "listing_path_prefixes": None,
    },
    "ansa.it": {
        # inspect_selectors.py 2026-07-31 on a Delmastro chat / Giunta piece:
        #   article                    -> 12 paras, 4898 chars, share + related + tags
        #   [itemprop='articleBody']   -> 12 paras, 4898 chars (100 %), clean
        "article_selector": "[itemprop='articleBody']",
        "exclude_selectors": [],
        "listing_path_prefixes": None,
    },
    "rtve.es": {
        # inspect_selectors.py 2026-07-31 on a forest-management / wildfires piece:
        #   article / main article -> 24 paras, 8862 chars, only shareBox
        #   main                   -> same text + extra nav + tag list
        "article_selector": "article",
        "exclude_selectors": [".shareBox"],
        "listing_path_prefixes": None,
    },
    "rtp.pt": {
        # inspect_selectors.py 2026-07-31 on an oil-company solidarity contribution piece:
        #   main            -> 41 paras, 6752 chars, social + share + ad frames
        #   .article-body   -> 3 paras, 3874 chars (57 %), clean but below 70 %
        "article_selector": "main",
        "exclude_selectors": [
            "[class*='social_buttons']",
            "[class*='sharethis']",
            "[class*='pub-frame']",
            "[class*='pub-advert']",
        ],
        "listing_path_prefixes": None,
    },
    "svt.se": {
        # inspect_selectors.py 2026-07-31 on a normal long-form local piece
        # (brand-pa-bat-i-vattern): article / main article / main all identical
        # (8 paras, 876 chars); only suspect is the ArticleFooter. Earlier
        # live-blog sample had no <article> and needed main + share excludes;
        # ordinary articles use <article>, so prefer that + footer only.
        # Existing END_OF_ARTICLE_MARKERS ("Så arbetar vi") still covers the
        # editorial-policy footer on other templates.
        "article_selector": "article",
        "exclude_selectors": ["footer"],
        "listing_path_prefixes": None,
    },
    "dw.com": {
        # inspect_selectors.py 2026-07-31 on a CSD Berlin / youth-judges piece:
        #   article -> 19 paras, 6798 chars; footer (author/feedback) + ad slots
        # No clean alternative. Hand-narrowed advert matches to .advertisement.
        "article_selector": "article",
        "exclude_selectors": ["footer", ".advertisement"],
        "listing_path_prefixes": None,
    },
    "newyddion.s4c.cymru": {
        # Inspected 2026-09-27 (inspect_selectors.py's container/suspect
        # logic, run in-browser) on a Ken Skates / Labour conference piece:
        #   main                    -> 17 paras, 2505 chars, clean
        #   [class*='ArticleBody']  -> 17 paras, 2505 chars, clean (identical)
        #   article / main article  -> matched, 0 paras
        # Tied on the text template, so the tool alone would keep 'main' --
        # overridden to ArticleBody on the evidence of the OTHER template:
        # on a video-only item (/article/36898) 'main' holds only video.js
        # player chrome ("This is a modal window." etc.) while ArticleBody
        # holds nothing, so ArticleBody never saves player text.
        # Sampled 7 random entries from server-sitemap.xml: 6 real text
        # articles (576-2836 chars in ArticleBody), 1 video-only.
        "article_selector": "[class*='ArticleBody']",
        "exclude_selectors": [],
        # Two URL forms point at the same articles: the homepage/topic pages
        # link /article/cy/<slug> (redirects to /article/<slug>), while the
        # sitemap uses canonical numeric /article/<id>. Prefix-match covers
        # both; short Welsh slugs would otherwise fail the generic
        # hyphen-count heuristic. Duplicates across the two forms are caught
        # by content-hash dedup.
        "listing_path_prefixes": ["/article/"],
    },
    "ycymro.cymru": {
        # Inspected 2026-09-27 on a news piece (Eisteddfod crowds) and an
        # opinion piece (/barn/, Senedd voting system):
        #   main / [role='main']  -> 25 paras, 4149 chars; share/social/
        #                            byline/footer/post-nav suspects, plus a
        #                            comments disclaimer paragraph
        #   article               -> 24 paras, 4003 chars; byline + share
        #   .entry-content        -> 24 paras, 4003 chars; share widget only
        # .entry-content chosen: same prose as 'article' minus the byline,
        # and excludes main's comment-section disclaimer. A site-wide
        # publisher footer + disclaimer still sits INSIDE .entry-content on
        # both samples -- handled by boilerplate_patterns.py (END_OF_ARTICLE
        # marker), not by a selector. Hand-narrowed the share suspect to its
        # Hustle plugin wrapper.
        "article_selector": ".entry-content",
        "exclude_selectors": ["[class*='hustle-social']"],
        # /<section>/YYYY/MM/<slug> -- the 4-digit year already satisfies
        # the generic heuristic; section/category/pagination pages don't.
        "listing_path_prefixes": None,
    },
}


def get_site_override(url):
    domain = urlparse(url).netloc.replace("www.", "").lower()
    return SITE_OVERRIDES.get(domain)


# get_expected_language(), is_language_locked(), known_locked_languages(),
# and resolve_listing_urls_for_languages() used to live here -- moved to
# language_detection.py alongside LANGUAGE_OVERRIDES, the table they
# actually read from. Imported below from that module instead.


_stanza_pipelines = {}  # lazily initialized per lang_code, only if --lemmatize is used


# --------------------------------------------------------------------------
# URL / filename helpers
# --------------------------------------------------------------------------

def normalize_url(url):
    """Strip query string / fragment / trailing slash so the same article
    reached via different tracking params still dedupes to one entry."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def safe_filename(url, variant=None, max_len=80):
    """variant, if given, is appended before the extension (e.g. 'lemma')
    so the raw and lemmatized copies of the same article get distinct
    filenames sharing the same domain/slug/hash prefix -- easy to spot
    as a pair, impossible to silently overwrite one another."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    slug = parsed.path.rstrip("/").split("/")[-1] or "index"
    slug = re.sub(r"[^a-zA-Z0-9\-_]", "_", slug)[:max_len]
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    suffix = f"__{variant}" if variant else ""
    return f"{domain}__{slug}__{url_hash}{suffix}.txt"


def looks_like_article(url, base_domain):
    """Generic heuristic for 'is this link probably a news article,
    as opposed to a nav/category/tag/video page' - domain agnostic.

    Checked FIRST against SITE_OVERRIDES' listing_path_prefixes, if the
    domain has one -- exact path-prefix matching is strictly more
    reliable than the hyphen/digit-ID guesswork below for any site whose
    real article URLs don't happen to fit that shape (see BBC's opaque
    IDs, documented in SITE_OVERRIDES above)."""
    parsed = urlparse(url)
    if parsed.netloc.replace("www.", "") != base_domain.replace("www.", ""):
        return False  # stay on the same site
    path_lower = parsed.path.lower()
    if path_lower.endswith(STATIC_EXTENSIONS):
        return False
    if any(hint in path_lower for hint in NON_ARTICLE_HINTS):
        return False

    override = get_site_override(url)
    if override and override.get("listing_path_prefixes"):
        return any(path_lower.startswith(prefix)
                   for prefix in override["listing_path_prefixes"])

    path = parsed.path.rstrip("/")
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    slug = segments[-1]

    hyphen_rich = slug.count("-") >= 3          # "biden-signs-new-bill-today"
    has_long_id = bool(re.search(r"\d{4,}", path))  # numeric article IDs
    deep_enough = len(segments) >= 2

    return deep_enough and (hyphen_rich or has_long_id)


# --------------------------------------------------------------------------
# Manifest (dedup across runs)
# --------------------------------------------------------------------------

def manifest_path(output_dir):
    return os.path.join(output_dir, MANIFEST_NAME)


def content_hash(text):
    """Hash of the normalized article body, so the same story reached via
    two different URLs (e.g. a retitled slug) is still recognized as a
    duplicate instead of being saved twice."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_manifest(output_dir):
    """Returns (seen_urls, seen_hashes). Tolerates the old one-column
    (URL-only) manifest format from before content-hash dedup existed.

    Kept as a plain loader with no filesystem verification -- reconcile_
    manifest() below is what run_auto/run_url_mode actually call at
    startup. Left standalone (rather than folded into reconcile_manifest)
    since it's also useful on its own wherever you just want to read the
    manifest as written, without triggering a disk walk + rewrite.
    """
    path = manifest_path(output_dir)
    seen_urls, seen_hashes = set(), set()
    if not os.path.exists(path):
        return seen_urls, seen_hashes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            seen_urls.add(parts[0])
            if len(parts) > 1 and parts[1]:
                seen_hashes.add(parts[1])
    return seen_urls, seen_hashes


def reconcile_manifest(output_dir):
    """
    Prunes manifest entries whose backing .txt file no longer exists on
    disk, and rewrites the manifest to match. Without this, deleting
    saved articles by hand (or via an external cleanup script) leaves
    the manifest's memory intact -- the scraper then silently treats
    those URLs as "already have it" forever, even though nothing is
    actually on disk to show for it.

    Matches on url_hash (see safe_filename) rather than the manifest's
    content_hash column: url_hash is deterministically recoverable from
    the URL alone, whereas confirming content_hash would mean re-reading
    and re-normalizing every surviving file's text just to compare a
    hash -- pure overhead over just checking whether a file with that
    hash exists. A URL counts as still-scraped if EITHER its raw or its
    lemmatized .txt exists anywhere under output_dir; which language
    subfolder or raw/lemmatized split it landed in doesn't matter here,
    only whether a file backing this URL still exists somewhere.

    Called once at the very start of run_auto/run_url_mode, before any
    scraping begins -- so a run right after a manual cleanup starts from
    an accurate picture instead of an accurate one only after the next
    full re-scrape completes.
    """
    path = manifest_path(output_dir)
    if not os.path.exists(path):
        return set(), set()

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Walk the output dir once up front -- O(1)-ish membership checks per
    # manifest line afterward instead of re-globbing per URL.
    existing_files = []
    for _root, _dirs, files in os.walk(output_dir):
        existing_files.extend(name for name in files if name.endswith(".txt"))

    kept_lines, seen_urls, seen_hashes = [], set(), set()
    pruned = 0
    for line in lines:
        parts = line.split("\t")
        url = parts[0]
        content_hash_value = parts[1] if len(parts) > 1 and parts[1] else None
        url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        # Matches "..._<hash>.txt" (raw) and "..._<hash>__lemma.txt" alike.
        still_exists = any(f"__{url_hash}" in fname for fname in existing_files)

        if still_exists:
            kept_lines.append(line)
            seen_urls.add(url)
            if content_hash_value:
                seen_hashes.add(content_hash_value)
        else:
            pruned += 1

    if pruned:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
        print(f"🧹 Manifest reconciled: {pruned} stale entr{'y' if pruned == 1 else 'ies'} "
              f"removed (backing file no longer on disk).")

    return seen_urls, seen_hashes


def _warn_if_legacy_layout(output_dir):
    """New articles always go to raw|lemmatized/<lang>/; articles saved
    before corpus_layout.py existed sit in <lang>/raw|lemmatized/. Dedup
    and backfill handle both, but the folder tree ends up split, so point
    at the one-off migration rather than migrating silently."""
    if has_legacy_layout(output_dir):
        print(f"⚠️  {output_dir} still has articles in the old <lang>/raw|lemmatized/ layout; "
              f"new ones go to raw|lemmatized/<lang>/. To move the old ones over: "
              f"python corpus_layout.py migrate --output-dir \"{output_dir}\"")


def append_to_manifest(output_dir, url, hash_value):
    with open(manifest_path(output_dir), "a", encoding="utf-8") as f:
        f.write(f"{url}\t{hash_value}\n")


def clear_downloaded_articles(output_dir, skip_confirm=False):
    """Physically DELETES every saved article .txt file under output_dir
    -- both the raw/ and lemmatized/ copies, across every per-language
    subfolder (news_corpus/raw/is/*.txt, news_corpus/lemmatized/en/*.txt,
    and any not-yet-migrated old-layout <lang>/raw/ files too, since this
    walks every .txt regardless of nesting). This is the actual downloaded
    content, not just a dedup record.

    scraped_urls.txt (MANIFEST_NAME) itself is deliberately excluded from
    the walk/delete below and is instead removed as a separate explicit
    step at the end: once the backing .txt files are gone, every manifest
    entry is stale anyway (reconcile_manifest() would otherwise prune them
    one by one, silently, the next time the script runs), so clearing it
    here up front avoids leaving a manifest full of dangling references
    and gives a genuinely clean slate.

    skip_confirm=True for scripted/CLI use (--yes); the interactive menu
    always confirms first since this can't be undone."""
    manifest = manifest_path(output_dir)
    txt_files = []
    if os.path.isdir(output_dir):
        for root, _dirs, files in os.walk(output_dir):
            for name in files:
                if not name.endswith(".txt"):
                    continue
                full = os.path.join(root, name)
                if full == manifest:
                    continue
                txt_files.append(full)

    manifest_exists = os.path.exists(manifest)
    if not txt_files and not manifest_exists:
        print(f"   Nothing found under {output_dir} -- nothing to clear.")
        return

    if not skip_confirm and not _prompt_yes_no(
        f"\nDelete {len(txt_files)} saved article .txt file(s) under {output_dir} "
        f"(raw + lemmatized) and reset {MANIFEST_NAME}? This permanently deletes "
        f"the downloaded articles themselves, not just the dedup record.",
        default=False,
    ):
        print("   Cancelled.")
        return

    for f in txt_files:
        os.remove(f)
    if manifest_exists:
        os.remove(manifest)
    print(f"   ✅ Deleted {len(txt_files)} article file(s)"
          f"{' and reset ' + MANIFEST_NAME if manifest_exists else ''}")


def delete_boilerplate_log(output_dir, skip_confirm=False):
    """Deletes boilerplate_candidates.json for output_dir -- the LLM
    review-queue log from boilerplate.py, not the actual
    boilerplate_patterns.py filter tables. Permanent: make sure any real
    hits have already been promoted into boilerplate_patterns.py by hand
    before clearing this, since nothing here reads it back out.

    skip_confirm=True for scripted/CLI use (--yes); the interactive menu
    always confirms first since this can't be undone."""
    path = os.path.join(output_dir, "boilerplate_candidates.json")
    if not os.path.exists(path):
        print(f"   No boilerplate_candidates.json found at {path} -- nothing to delete.")
        return
    if not skip_confirm and not _prompt_yes_no(
        f"\nDelete {path}? Make sure you've reviewed and promoted any real "
        f"patterns into boilerplate_patterns.py first -- this is permanent.",
        default=False,
    ):
        print("   Cancelled.")
        return
    os.remove(path)
    print(f"   ✅ Deleted {path}")


# --------------------------------------------------------------------------
# Lemmatization (optional, lazy)
# --------------------------------------------------------------------------

def ensure_lemmatizer(lang_code):
    """Initialize the Stanza pipeline for lang_code, up front, before any
    scraping happens. Previously this was lazy-loaded inside the
    per-article loop, so a bad language code or a download failure would
    surface as a generic 'Error' on article 1, get silently retried on
    every subsequent article, and never give a clear top-level failure.

    BUGFIX: _stanza_pipeline used to be a single global keyed by
    nothing -- ensure_lemmatizer() would no-op on any call after the
    first, regardless of what lang_code it was actually asked for.
    That's invisible within a single run (lang_code is fixed for the
    whole run), but the wizard supports doing multiple runs back-to-back
    in one process ("another run?"), and a second run with a different
    --lemmatize language would silently keep lemmatizing with the
    FIRST run's language/model instead of the one it was just told to
    use -- e.g. English AP articles run through a cached Welsh
    pipeline, producing garbled pseudo-Welsh text with no error or
    warning, since Stanza has no way to know the input is the "wrong"
    language -- it just does its best to lemmatize whatever text it's
    given. Fixed by keying the cache off lang_code, so each language
    gets (and keeps) its own pipeline instead of one shared slot."""
    global _stanza_pipelines
    if lang_code in _stanza_pipelines:
        return
    import stanza
    print(f"Loading Stanza lemmatizer for '{lang_code}'...")
    try:
        _stanza_pipelines[lang_code] = stanza.Pipeline(lang_code, processors="tokenize,lemma")
    except Exception:
        print(f"Model for '{lang_code}' not found locally, downloading...")
        stanza.download(lang_code)
        _stanza_pipelines[lang_code] = stanza.Pipeline(lang_code, processors="tokenize,lemma")
    print("Lemmatizer ready!\n")


def lemmatize(text, lang_code):
    ensure_lemmatizer(lang_code)
    doc = _stanza_pipelines[lang_code](text)
    lemmas = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.lemma and word.lemma.strip():
                lemmas.append(word.lemma)
    return " ".join(lemmas)


# --------------------------------------------------------------------------
# Language detection (drives per-language output subfolders)
# --------------------------------------------------------------------------

async def detect_language(text, url, use_gemini_tiebreak=False):
    """Runs the full language_detection.py panel (GlotLID, OpenLID-v3,
    lingua) on an article's RAW extracted text -- called before any
    --lemmatize step, since a lemmatized token stream is a worse
    detection input than natural text, same reasoning as before.

    The panel itself is synchronous (fastText/lingua are all
    blocking C++/native calls) so it's run via asyncio.to_thread() to
    avoid stalling the Playwright event loop while three models run --
    same reason detect_language_llm_sync() gets the to_thread()
    treatment in language_detection.py.

    url is used to look up a site-level language prior via
    get_expected_language(), AND to check is_language_locked() first --
    a LOCKED domain (e.g. ruv.is) skips the panel entirely and returns
    its mapped code directly with zero text inspection; everything else
    uses get_expected_language() the old way, as a corroborating
    judge_language() vote, never a sole decider. Both functions and the
    LANGUAGE_OVERRIDES table they read from live in language_detection.py
    now, not in this file's SITE_OVERRIDES -- see that table's own
    header comment for what locking does and doesn't guarantee.

    Returns (lang_code, verdict_or_None):
      - Locked domain, path resolves to a code -> (code, None) -- the
        panel never ran at all for this article, so there's genuinely no
        verdict to report (distinct from every other case below, where a
        verdict object always comes back once the panel has run, whether
        or not it disputed -- see corpus_metadata.py's record_for_article,
        which logs winner_share/votes for consensus articles too now,
        not just disputed ones).
      - Consensus (locally, or after a Gemini tiebreak) -> (code, verdict).
      - No judge had anything to say -> (UNKNOWN_LANG_FOLDER, verdict).
      - Still disputed after all available judges (including Gemini, if
        use_gemini_tiebreak) -> (DISPUTED_LANG_FOLDER, verdict).

    use_gemini_tiebreak mirrors the old detect_language_llm_flag --
    Gemini is ONLY called here, and ONLY when the local panel alone
    couldn't resolve, which is what actually cuts Gemini's call volume
    down (as opposed to just rate-limiting the same per-article call
    rate the old two-voice system made on every article).
    """
    if is_language_locked(url):
        locked_code = get_expected_language(url)
        if locked_code:
            # No judges touched, no models loaded on their account -- if
            # a run only ever scrapes locked domains, GlotLID/OpenLID's
            # lazy loaders in language_detection.py are simply never
            # called, so their ~3GB combined download/load never happens
            # at all this run, not just "happens but is fast".
            return locked_code, None
        # Locked domain, but this specific path isn't covered by its
        # expected_language mapping (a dict override missing a
        # "default", or a new path prefix nobody's added yet). Falling
        # through to the full panel here on purpose -- language_lock is
        # a claim that every path is mapped, and this is that claim
        # turning out false for this one URL. Don't silently guess;
        # treat it exactly like an unlocked domain instead, and let this
        # print serve as the signal that language_detection.py's
        # LANGUAGE_OVERRIDES needs an update.
        print(f"⚠️  {url} matched a language_lock'd domain but "
              "get_expected_language() returned nothing for this path -- "
              "falling back to the full detection panel. The "
              "language_lock mapping for this domain is missing a case; "
              "worth fixing in language_detection.py's LANGUAGE_OVERRIDES.")

    site_hint_code = get_expected_language(url)
    verdict = await asyncio.to_thread(judge_language, text, _lingua_detector, site_hint_code)

    if not verdict.disputed:
        return (verdict.code if verdict.code else UNKNOWN_LANG_FOLDER), verdict

    if use_gemini_tiebreak:
        gemini_result = await detect_language_llm(text)  # (code, confidence) or None
        verdict = reconcile_gemini_tiebreak(verdict, gemini_result)
        if not verdict.disputed:
            return verdict.code, verdict

    return DISPUTED_LANG_FOLDER, verdict


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

# Selectors tried, in order, when trafilatura fails and we fall back to
# raw <p> scraping. Most news CMS templates wrap the actual story body in
# one of these; restricting to the first one found on the page means the
# fallback never even sees footer/header/nav paragraphs living outside
# it. Only if NONE of these match anything do we fall through to
# page-wide <p> (last resort, then filtered by looks_like_boilerplate).
ARTICLE_CONTAINER_SELECTORS = [
    "article", "main article", "[itemprop='articleBody']",
    ".article-body", ".article-content", "#article-body",
    "main", "[role='main']",
]

# Generic boilerplate/junk-text tables (BOILERPLATE_PATTERNS,
# END_OF_ARTICLE_MARKERS, READ_MORE_SUFFIXES) live in boilerplate_patterns.py
# now, not inline here -- that list is expected to keep growing as new
# sites get scraped, and keeping it as pure data in its own file (same
# split as mutation_tables.py on the Welsh project) means adding a new
# pattern is a one-line edit there instead of hunting through extraction
# logic to find where to drop it in.
from boilerplate_patterns import (
    BOILERPLATE_PATTERNS,
    END_OF_ARTICLE_MARKERS,
    READ_MORE_SUFFIXES,
)
import boilerplate_patterns  # module object itself, not just its contents --
                               # boilerplate.run_review() needs
                               # .__file__ to know which file on disk to edit
from boilerplate import (
    queue_boilerplate_check,
    flush_boilerplate_queue,
    get_gemini_client,
    reset_quota_flag,
    run_review,
    offer_end_of_run_boilerplate_review,
    is_suspicious,
)
from corpus_metadata import record_for_article, append_metadata_record, backfill_missing
from corpus_layout import RAW, LEMMATIZED, variant_dir, has_legacy_layout
from language_detection import (
    detect_language_llm,
    reset_language_quota_flag,
    judge_language,
    reconcile_gemini_tiebreak,
    log_language_dispute,
    get_expected_language,
    is_language_locked,
    known_locked_languages,
    resolve_listing_urls_for_languages,
)
from term_ui import rule, wrap, half_width
import textwrap

# BUGFIX: re.compile("|".join([])) compiles to the EMPTY pattern, which
# matches at every position in every string -- so if any of these three
# tables ever gets emptied out (a pattern list that's *expected* to keep
# growing is also a list someone could accidentally empty during a
# refactor, or comment out the last entry of), _BOILERPLATE_RE/
# _END_OF_ARTICLE_RE would start silently classifying EVERY paragraph on
# EVERY article as boilerplate/an end-marker, rather than matching
# nothing -- a silent total-data-loss failure mode, not a no-op. Guard by
# falling back to a pattern that can never match instead of compiling an
# empty alternation.
_NEVER_MATCHES = r"(?!x)x"


def _compile_alternation(patterns):
    return re.compile("|".join(patterns) if patterns else _NEVER_MATCHES, re.IGNORECASE)


_END_OF_ARTICLE_RE = _compile_alternation(END_OF_ARTICLE_MARKERS)
_BOILERPLATE_RE = _compile_alternation(BOILERPLATE_PATTERNS)
_READ_MORE_RE = re.compile(
    r"\s*(?:" + "|".join(re.escape(s) for s in READ_MORE_SUFFIXES) + r")\s*$"
    if READ_MORE_SUFFIXES else _NEVER_MATCHES,
    re.IGNORECASE,
)


def truncate_at_end_markers(paragraphs):
    """Drops an END_OF_ARTICLE_MARKERS paragraph and everything after it.
    Returns (paragraphs, truncated) -- truncated is True only when a
    marker actually fired.

    That second return value exists specifically to gate
    strip_trailing_header_fragment() below (see its own docstring for
    why): this alone isn't enough when the related-content section sits
    BEFORE the footer instead of after it -- confirmed on a real SVT.se
    article, where a carousel title ("Framsteg runt om i världen") landed
    immediately before the "Så arbetar vi" marker rather than after, and
    survived as a trailing fragment in the saved article. But that's a
    narrow, specific failure mode -- a stranded fragment left over by
    THIS truncation -- not a general property of article endings, so the
    caller should only reach for that follow-up heuristic on the runs
    where this one actually cut something, not on every article.
    """
    for i, p in enumerate(paragraphs):
        if _END_OF_ARTICLE_RE.search(p.strip()):
            return paragraphs[:i], True
    return paragraphs, False


_SENTENCE_END_CHARS = ".!?…”\"'"


def strip_trailing_header_fragment(paragraphs):
    """Trims a bare heading-like paragraph off the END of the list --
    short (<=60 chars) and not ending in normal sentence-closing
    punctuation. Real prose almost always ends with one; a related-
    content carousel title like "Framsteg runt om i världen" (SVT.se,
    confirmed) doesn't.

    CALL THIS ONLY when truncate_at_end_markers() just reported
    truncated=True. The length+punctuation shape this checks for is
    common enough in ordinary short final sentences, quotes, or lines
    ending in something outside _SENTENCE_END_CHARS (a colon, a closing
    parenthesis, a language's own quote marks) that running it
    unconditionally on every article -- most of which never had an
    end-marker fire at all -- would risk trimming real content on the
    strength of a heuristic that was only ever validated against the one
    stranded-fragment failure mode it was built for. Gating it behind
    "did truncation actually happen" is what keeps it reasonable: it
    only runs when there's an actual reason to suspect a stranded
    fragment is sitting at the tail.

    Only ever trims from the tail, and never trims the list down to
    nothing, so even when it IS called, this can't eat every remaining
    paragraph -- worst case a short, unpunctuated final line stays
    untouched because trimming would have emptied the list.
    """
    result = list(paragraphs)
    while len(result) > 1:
        last = result[-1].strip()
        if last and len(last) <= 60 and last[-1] not in _SENTENCE_END_CHARS:
            result.pop()
        else:
            break
    return result


def looks_like_boilerplate(paragraph_text):
    return bool(_BOILERPLATE_RE.search(paragraph_text.strip()))


def dedupe_consecutive_paragraphs(paragraphs):
    """Drops back-to-back duplicate paragraphs and collapses truncated-
    teaser + full-text pairs from read-more widgets into just the full
    version. Two known real cases this fixes:

    1. A byline/credit block appearing twice in a row verbatim (confirmed
       on a NOS.nl article: "correspondent Israël en Palestijnse Gebieden"
       extracted as two separate identical paragraphs).
    2. DR.dk "vis mere" teaser widgets: the short version ending in "vis
       mere" is immediately followed by the same text again in full. Kept
       version is the fuller one (without the trailing "vis mere"), since
       that's the actual content -- the truncated copy is UI chrome, not
       a second sentence.

    Only compares immediate neighbors (not a corpus-wide dedup) since the
    duplication this targets is always adjacent in the DOM.
    """
    cleaned = []
    for p in paragraphs:
        p_stripped = p.strip()
        if cleaned:
            prev = cleaned[-1].strip()
            if p_stripped == prev:
                continue  # exact consecutive duplicate, drop the repeat
            prev_no_suffix = _READ_MORE_RE.sub("", prev).strip()
            if prev_no_suffix and prev_no_suffix == p_stripped:
                cleaned[-1] = p  # replace the truncated teaser with the full text
                continue
        cleaned.append(p)
    return cleaned


async def extract_via_override(page, override):
    """
    Deterministic extraction for a domain with a SITE_OVERRIDES entry:
    pull <p> text from inside article_selector only, dropping any
    paragraph that falls inside one of exclude_selectors first. Runs
    BEFORE trafilatura for domains that have an override, since a
    selector already confirmed against a real page (via
    inspect_selectors.py) is more trustworthy than trafilatura's
    general-purpose heuristics for that specific site.

    Returns None (not raising) on no match, so callers can cleanly fall
    through to the generic trafilatura/container path -- an override
    matching nothing on a given page (a template change, or a page type
    the override wasn't built against, like a live-blog vs. a standard
    article) should degrade gracefully, not break the whole scrape.

    BUGFIX: confirmed on a real scraped RUV article -- a cookie-consent
    placeholder for a blocked embed ("Hér á að vera innfellt efni...")
    rendered as its own <p> INSIDE .article-body, once per blocked
    embed (3x in that one piece). Container-scoping alone can't catch
    this since it's not a widget living outside the container, it's UI
    text sitting directly among the real paragraphs -- needs the same
    looks_like_boilerplate() text-content check the generic fallback
    already used, applied here too now rather than only there.
    """
    selector = override.get("article_selector")
    if not selector:
        return None
    exclude_selectors = override.get("exclude_selectors") or []
    try:
        texts = await page.eval_on_selector_all(
            f"{selector} p",
            """(paragraphs, excludeSelectors) => {
                return paragraphs
                    .filter(p => !excludeSelectors.some(sel => p.closest(sel)))
                    .map(p => p.innerText.trim())
                    .filter(t => t.length >= 20);
            }""",
            exclude_selectors,
        )
    except Exception:
        return None
    if not texts:
        return None
    texts, truncated = truncate_at_end_markers(texts)
    if truncated:
        texts = strip_trailing_header_fragment(texts)
    texts = dedupe_consecutive_paragraphs(texts)
    texts = [t for t in texts if not looks_like_boilerplate(t)]
    if not texts:
        return None
    return "\n\n".join(texts).strip()


async def fetch_html(page, url):
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
    try:
        # BUGFIX: this used to wait for a single <p> to attach, then grab
        # the page immediately. On bylined-article templates (RUV and
        # presumably others), the header/byline/subhead block often
        # renders first and satisfies that check well before the real
        # body has hydrated in -- so the page got snapshotted mid-load,
        # and trafilatura was handed just the teaser+metadata block.
        # Waiting for a real paragraph COUNT (an actual body almost never
        # renders as a single <p>) is a much stronger signal the article
        # body itself is present, not just page chrome.
        await page.wait_for_function(
            "document.querySelectorAll('p').length >= 3", timeout=8000
        )
    except Exception:
        pass  # some pages render text without <p> tags, or genuinely have <3; trafilatura may still work
    try:
        # Best-effort extra grace period for anything still hydrating in
        # after the paragraph-count check passes (images, related-content
        # widgets, etc. finishing up) -- short timeout, never blocks long.
        await page.wait_for_load_state("networkidle", timeout=4000)
    except Exception:
        pass
    return await page.content()


async def extract_article_text(page, url):
    """Returns (text, extraction_method) -- extraction_method is one of
    "override" / "trafilatura" / "fallback", recorded so
    corpus_metadata.py can log which extraction path actually produced
    each saved article (a real data-quality variable in its own right:
    e.g. articles pulled via the raw-<p> "fallback" path are the ones
    most likely to carry residual site chrome trafilatura/an override
    would otherwise have caught). Returns (None, None) when nothing
    usable was found at all."""
    html = await fetch_html(page, url)

    # If this domain has a proven selector (built from an actual
    # inspect_selectors.py run, see SITE_OVERRIDES), try it first --
    # more precise than trafilatura's general-purpose guessing for a
    # site we've already confirmed the structure of. Falls through to
    # the generic path below if it matches nothing (template changed,
    # or this is a page type the override wasn't built against).
    override = get_site_override(url)
    if override:
        override_text = await extract_via_override(page, override)
        if override_text and len(override_text) > 200:
            return override_text, "override"

    # Primary: trafilatura, which is built for exactly this - pulling clean
    # article text out of arbitrary news-site HTML while dropping nav/ads/
    # comments/boilerplate automatically, no site-specific rules needed.
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )

    # BUGFIX (second line of defense against the same render-timing race
    # fetch_html's paragraph-count wait targets): a result that clears the
    # 200-char floor but is still suspiciously short is usually a
    # byline/teaser block extracted successfully, not a genuinely short
    # article -- give the page a bit longer and retry extraction once
    # before accepting it. Only fires when the first attempt actually
    # improves, so a genuinely short article (e.g. the weather bulletin
    # style one-paragraph pieces) isn't churned pointlessly.
    if text and 200 < len(text.strip()) < 500:
        try:
            await page.wait_for_timeout(1500)
            html_retry = await page.content()
            text_retry = trafilatura.extract(
                html_retry, url=url, include_comments=False,
                include_tables=False, favor_recall=True,
            )
            if text_retry and len(text_retry.strip()) > len(text.strip()):
                text = text_retry
        except Exception:
            pass

    if text and len(text.strip()) > 200:
        # BUGFIX: trafilatura's own boilerplate detection targets nav/
        # ads/comments, not this specific cookie-consent-for-blocked-
        # embed placeholder -- it survives trafilatura's cleanup as a
        # plain paragraph in the middle of otherwise-real article text
        # (confirmed on RUV; nothing about the pattern is RUV-specific,
        # so this runs for every site on this path, not just overridden
        # ones). Split on paragraph breaks, drop any matching paragraph,
        # rejoin -- same check extract_via_override() and the page-wide
        # fallback below both use, kept in sync across all three paths.
        raw_paragraphs = text.strip().split("\n\n")
        truncated_paragraphs, was_truncated = truncate_at_end_markers(raw_paragraphs)
        if was_truncated:
            truncated_paragraphs = strip_trailing_header_fragment(truncated_paragraphs)
        deduped_paragraphs = dedupe_consecutive_paragraphs(truncated_paragraphs)
        cleaned_paragraphs = [p for p in deduped_paragraphs
                              if not looks_like_boilerplate(p)]
        cleaned = "\n\n".join(cleaned_paragraphs).strip()
        if cleaned:
            return cleaned, "trafilatura"

    # Fallback: raw <p> tag concatenation, for the rare page trafilatura
    # can't parse (e.g. unusual JS-rendered structure).
    #
    # BUGFIX: this used to query <p> across the WHOLE page, which pulls in
    # footer/header/sidebar paragraphs the primary path never sees (e.g.
    # RUV's address/phone footer block ending up saved as article text).
    # Now tries known article-container selectors first -- if the page
    # has one, only <p>s inside it are ever considered, so boilerplate
    # living OUTSIDE the container is structurally excluded, not just
    # filtered. looks_like_boilerplate() still runs on every paragraph
    # regardless of whether a container matched, though (see the BUGFIX
    # below it) -- a matched container is not a guarantee against
    # boilerplate living INSIDE it.
    paragraphs = []
    for selector in ARTICLE_CONTAINER_SELECTORS:
        container = await page.query_selector(selector)
        if container:
            paragraphs = await container.query_selector_all("p")
            if paragraphs:
                break

    if not paragraphs:
        paragraphs = await page.query_selector_all("p")

    # BUGFIX: this used to only run looks_like_boilerplate() when no
    # container matched, on the assumption that a matched container
    # (article/.article-body/etc.) can't contain boilerplate paragraphs.
    # extract_via_override() above disproves that for this exact code-
    # base: the RUV cookie-consent-for-blocked-embed placeholder rendered
    # as its own <p> INSIDE .article-body, not outside it, which is why
    # that function needed this same text-content check applied even
    # after container-scoping. This fallback had the identical blind spot
    # -- run the check unconditionally now, same as every other path.
    texts = []
    for p in paragraphs:
        t = (await p.inner_text()).strip()
        if len(t) < 40:
            continue
        if looks_like_boilerplate(t):
            continue
        texts.append(t)
    texts, truncated = truncate_at_end_markers(texts)
    if truncated:
        texts = strip_trailing_header_fragment(texts)
    texts = dedupe_consecutive_paragraphs(texts)
    fallback_text = "\n\n".join(texts).strip()
    return (fallback_text, "fallback") if fallback_text else (None, None)


_SITEMAP_LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)


async def _discover_from_sitemap(page, sitemap_url, base_domain):
    """An XML sitemap has no <a href> elements at all, so the normal
    listing-page path below silently discovers ZERO links from one (it
    waits for a[href], times out, and skips). Sitemaps are worth
    supporting anyway because they're usually the only way to reach a
    site's back catalogue rather than just today's front page -- e.g.
    newyddion.s4c.cymru's server-sitemap.xml lists a rolling month of
    articles (~500 URLs), versus ~70 on its homepage. Reads the raw
    response body (not page.content(), which would be Chromium's
    rendered XML viewer) and runs every <loc> through the same
    looks_like_article() filter as a normal listing page.

    Leaf sitemaps only: a <sitemapindex>'s <loc>s point at more .xml
    files, which looks_like_article() rejects, so an index yields nothing
    rather than recursing -- point listing_urls at the leaf sitemap."""
    try:
        response = await page.goto(sitemap_url, wait_until="domcontentloaded", timeout=30000)
        xml = await response.text() if response else ""
    except Exception:
        print(f"   Could not load sitemap {sitemap_url}, skipping")
        return []
    locs = [unescape(u) for u in _SITEMAP_LOC_RE.findall(xml)]
    candidates = [normalize_url(u) for u in locs if looks_like_article(u, base_domain)]
    return list(dict.fromkeys(candidates))


async def discover_article_links(page, listing_url):
    base_domain = urlparse(listing_url).netloc
    if urlparse(listing_url).path.lower().endswith(".xml"):
        return await _discover_from_sitemap(page, listing_url, base_domain)
    await page.goto(listing_url, wait_until="domcontentloaded", timeout=30000)
    try:
        await page.wait_for_selector("a[href]", state="attached", timeout=10000)
    except Exception:
        print(f"   Could not load links from {listing_url}, skipping")
        return []

    raw_links = await page.eval_on_selector_all(
        "a[href]", "elements => elements.map(el => el.href)"
    )

    candidates = []
    for link in raw_links:
        if looks_like_article(link, base_domain):
            candidates.append(normalize_url(link))

    return list(dict.fromkeys(candidates))  # dedupe, keep order


# --------------------------------------------------------------------------
# Scrape a single URL (used by both modes)
# --------------------------------------------------------------------------

async def scrape_one(page, url, output_dir, lemmatize_lang, seen_hashes, index, total,
                      detect_boilerplate=False, detect_language_llm_flag=False,
                      discovered_via=None):
    try:
        tqdm.write(f"[{index}/{total}] {url}")
        text, extraction_method = await extract_article_text(page, url)

        if not text:
            tqdm.write("   💥 No usable text found")
            return False

        # Content-based dedup: catches the same story republished under a
        # different slug/URL, which URL-only dedup can't see.
        raw_hash = content_hash(text)
        if raw_hash in seen_hashes:
            tqdm.write("   ↩️  Duplicate content (already have this story under another URL) - skipping")
            append_to_manifest(output_dir, normalize_url(url), raw_hash)
            return True

        # Detected on the raw extracted text, BEFORE lemmatization -- a
        # lemmatized token stream (already reduced to dictionary forms,
        # function words flattened) is a worse detection input than
        # natural running text.
        lang_code, verdict = await detect_language(text, url, use_gemini_tiebreak=detect_language_llm_flag)
        # BUGFIX: this used to gate on `verdict is not None`, which was
        # equivalent to "disputed" back when detect_language() only ever
        # returned a verdict for disputed articles. Now that verdict is
        # populated for every article the panel actually ran on (see
        # detect_language()'s docstring -- corpus_metadata.py wants the
        # winner_share/votes for consensus articles too), that check would
        # fire on every non-locked article and mislog every consensus
        # result into language_disputes.json. Gate on the actual outcome
        # instead.
        if lang_code == DISPUTED_LANG_FOLDER and verdict is not None:
            logged = log_language_dispute(output_dir, url, verdict)
            if logged:
                vote_summary = ", ".join(f"{name}={v.code}({v.confidence:.2f})"
                                          for name, v in verdict.votes.items())
                tqdm.write(f"   🗣️  Language panel disputed ({vote_summary}) -> filed under "
                           f"'{DISPUTED_LANG_FOLDER}', logged to language_disputes.json")
        # BUGFIX: this used to lemmatize text IN PLACE and only ever save
        # one variant -- meaning scraping a URL raw, then later re-running
        # the same URL with --lemmatize, just got skipped as a duplicate
        # (dedup is by URL/content-hash, computed before lemmatization),
        # so there was no way to actually end up with both versions of
        # the same article side by side. Now: the raw text is always
        # saved first, and a lemmatized copy is saved ADDITIONALLY (same
        # single extraction, no re-scraping) when --lemmatize is given --
        # raw/<lang>/ and lemmatized/<lang>/ (see corpus_layout.py), with
        # the lemmatized filename carrying a '__lemma' suffix so the two
        # never collide and are easy to spot as a matched pair.
        raw_dir = variant_dir(output_dir, RAW, lang_code)
        os.makedirs(raw_dir, exist_ok=True)
        raw_filename = safe_filename(url)
        raw_filepath = os.path.join(raw_dir, raw_filename)
        with open(raw_filepath, "w", encoding="utf-8") as f:
            f.write(text)
        saved_note = f"{RAW}/{lang_code}/{raw_filename}"

        if lemmatize_lang:
            tqdm.write("   Lemmatizing...")
            lemma_text = lemmatize(text, lemmatize_lang)
            lemma_dir = variant_dir(output_dir, LEMMATIZED, lang_code)
            os.makedirs(lemma_dir, exist_ok=True)
            lemma_filename = safe_filename(url, variant="lemma")
            lemma_filepath = os.path.join(lemma_dir, lemma_filename)
            with open(lemma_filepath, "w", encoding="utf-8") as f:
                f.write(lemma_text)
            saved_note += f"  +  {LEMMATIZED}/{lang_code}/{lemma_filename}"

        # Optional LLM pass over the just-saved text, looking for
        # boilerplate that slipped past BOILERPLATE_PATTERNS /
        # END_OF_ARTICLE_MARKERS. This never touches what got saved above
        # -- it only logs candidates to boilerplate_candidates.json for
        # you to review and promote into boilerplate_patterns.py by hand,
        # same role lemma_cache.json plays on the Welsh project (a
        # persistent record building up across runs), except this one is
        # a review queue rather than a reusable cache: entries don't get
        # consumed, they sit there until you've looked at them.
        #
        # Doesn't call Gemini directly -- queue_boilerplate_check() only
        # queues articles its own suspicion filter thinks are worth a
        # look, and sends them in batches rather than one call per
        # article; see boilerplate.py's module docstring. This
        # means candidates from a SUSPICIOUS article might not get
        # logged/printed until a later article in this same run triggers
        # the batch to flush (or the run ends and flush_boilerplate_queue()
        # catches the remainder) -- output no longer arrives strictly
        # per-article the way it used to.
        if detect_boilerplate:
            await queue_boilerplate_check(output_dir, url, text)

        # One row per saved article in corpus_metadata.jsonl -- word/
        # sentence/paragraph counts, cheap register-proxy stats (TTR,
        # avg sentence/word length), the extraction path used, and the
        # full language-panel vote breakdown (not just disputed cases --
        # see corpus_metadata.py's module docstring). Meant to be loaded
        # straight into pandas for exploratory analysis once there's an
        # actual research question to point it at; appending here rather
        # than at the top of the function means a failed/duplicate/
        # no-text article never gets a row for text that was never
        # actually saved.
        append_metadata_record(output_dir, record_for_article(
            url=url,
            domain=urlparse(url).netloc.replace("www.", ""),
            language_code=lang_code,
            language_locked=is_language_locked(url),
            language_verdict=verdict,
            extraction_method=extraction_method,
            was_suspicious=is_suspicious(text),
            text=text,
            lemmatized_lang=lemmatize_lang,
            discovered_via=discovered_via,
        ))

        seen_hashes.add(raw_hash)
        append_to_manifest(output_dir, normalize_url(url), raw_hash)
        tqdm.write(f"   ✅ Saved: {saved_note}")
        return True

    except Exception as e:
        tqdm.write(f"   💥 Error: {e}")
        return False


async def scrape_batch(page, urls, output_dir, lemmatize_lang, seen_hashes, delay,
                        detect_boilerplate=False, detect_language_llm_flag=False,
                        discovered_via=None):
    """discovered_via: optional {url: listing_url} map from run_auto,
    recorded per article in corpus_metadata.jsonl. None in url mode."""
    discovered_via = discovered_via or {}
    failed = []
    saved = 0
    with tqdm(total=len(urls), desc="Scraping", unit="article") as pbar:
        for i, url in enumerate(urls):
            ok = await scrape_one(page, url, output_dir, lemmatize_lang, seen_hashes, i + 1, len(urls),
                                   detect_boilerplate=detect_boilerplate,
                                   detect_language_llm_flag=detect_language_llm_flag,
                                   discovered_via=discovered_via.get(url))
            if ok:
                saved += 1
            else:
                failed.append(url)
            pbar.set_postfix({"✅": saved, "💥": len(failed)})
            pbar.update(1)
            if delay > 0 and i < len(urls) - 1:
                await asyncio.sleep(delay + random.uniform(0, delay * 0.5))

    if failed:
        tqdm.write(f"\n🔁 Retrying {len(failed)} failed URLs...\n")
        still_failed = []
        with tqdm(total=len(failed), desc="Retrying", unit="article") as pbar:
            for i, url in enumerate(failed):
                ok = await scrape_one(page, url, output_dir, lemmatize_lang, seen_hashes, i + 1, len(failed),
                                       detect_boilerplate=detect_boilerplate,
                                       detect_language_llm_flag=detect_language_llm_flag,
                                       discovered_via=discovered_via.get(url))
                if ok:
                    saved += 1
                else:
                    still_failed.append(url)
                pbar.set_postfix({"✅": saved, "💥": len(still_failed)})
                pbar.update(1)
                if delay > 0 and i < len(failed) - 1:
                    await asyncio.sleep(delay + random.uniform(0, delay * 0.5))
        if still_failed:
            tqdm.write(f"\n💥 {len(still_failed)} URLs could not be scraped after retry:")
            for u in still_failed:
                tqdm.write(f"   {u}")

    if detect_boilerplate:
        # Catches whatever's left in the queue below BATCH_SIZE -- without
        # this, a run whose suspicious-article count isn't an exact
        # multiple of BATCH_SIZE would silently lose that remainder's
        # candidates when the process exits. Safe to call unconditionally
        # (no-op on an empty queue), but only worth calling at all when
        # this run actually turned the feature on.
        await flush_boilerplate_queue()

        # Human gate at the end of every detection-enabled run: review
        # now (approve → boilerplate_patterns.py), leave in the JSON for
        # later, or discard. Replaces the previous unconditional
        # run_review() call so a long scrape doesn't force a full review
        # session when you only wanted to save candidates. Non-TTY
        # (scripted) runs skip the prompt and leave the log as-is.
        # Hooked here because both run_auto() and run_url_mode() funnel
        # through scrape_batch -- one call site covers both entry points.
        offer_end_of_run_boilerplate_review(output_dir, boilerplate_patterns)

    return saved


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

async def run_auto(listing_urls, output_dir, lemmatize_lang, delay, detect_boilerplate=False,
                    detect_language_llm_flag=False):
    os.makedirs(output_dir, exist_ok=True)
    already_urls, seen_hashes = reconcile_manifest(output_dir)
    _warn_if_legacy_layout(output_dir)

    if lemmatize_lang:
        ensure_lemmatizer(lemmatize_lang)  # fail fast on a bad language code
    if detect_boilerplate or detect_language_llm_flag:
        get_gemini_client()  # fail fast on missing package/API key -- shared
                              # by both LLM features, same client either way
    if detect_boilerplate:
        reset_quota_flag()   # a previous run in this same interactive-menu
                              # session shouldn't leave detection disabled here
    if detect_language_llm_flag:
        reset_language_quota_flag()  # same reasoning, separate flag

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(30000)

        all_links = []
        # First listing page to link a URL wins attribution, matching the
        # dict.fromkeys() order below -- so pass specific section/topic
        # pages BEFORE general ones (homepage, sitemap) to get the most
        # informative discovered_via value in corpus_metadata.jsonl.
        discovered_via = {}
        for listing_url in listing_urls:
            print(f"Collecting links from: {listing_url}")
            links = await discover_article_links(page, listing_url)
            print(f"   Found {len(links)} candidate article links")
            all_links.extend(links)
            for link in links:
                discovered_via.setdefault(link, listing_url)
            if delay > 0:
                await asyncio.sleep(delay)

        all_links = list(dict.fromkeys(all_links))
        new_links = [u for u in all_links if u not in already_urls]

        print(f"\n{len(new_links)} new articles to scrape ({len(already_urls)} already in corpus)")
        print(f"Saving corpus to: {output_dir}\n")

        if not new_links:
            print("💥 Nothing new - come back later!")
            await browser.close()
            return

        saved = await scrape_batch(page, new_links, output_dir, lemmatize_lang, seen_hashes, delay,
                                    detect_boilerplate=detect_boilerplate,
                                    detect_language_llm_flag=detect_language_llm_flag,
                                    discovered_via=discovered_via)
        await browser.close()
        print(f"\n🎉 Done! Saved {saved} new articles to {output_dir}/")


async def run_url_mode(article_urls, output_dir, lemmatize_lang, delay, detect_boilerplate=False,
                        detect_language_llm_flag=False):
    os.makedirs(output_dir, exist_ok=True)
    article_urls = [normalize_url(u) for u in article_urls]
    article_urls = list(dict.fromkeys(article_urls))
    _, seen_hashes = reconcile_manifest(output_dir)
    _warn_if_legacy_layout(output_dir)

    if lemmatize_lang:
        ensure_lemmatizer(lemmatize_lang)  # fail fast on a bad language code
    if detect_boilerplate or detect_language_llm_flag:
        get_gemini_client()  # fail fast on missing package/API key -- shared
                              # by both LLM features, same client either way
    if detect_boilerplate:
        reset_quota_flag()   # a previous run in this same interactive-menu
                              # session shouldn't leave detection disabled here
    if detect_language_llm_flag:
        reset_language_quota_flag()  # same reasoning, separate flag

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(30000)

        saved = await scrape_batch(page, article_urls, output_dir, lemmatize_lang, seen_hashes, delay,
                                    detect_boilerplate=detect_boilerplate,
                                    detect_language_llm_flag=detect_language_llm_flag)
        await browser.close()
        print(f"\n🎉 Done! Saved {saved} article(s) to {output_dir}/")


# --------------------------------------------------------------------------
# Interactive menu (launched with no CLI arguments -- run with args, e.g.
# `python icelandic_text_extractor.py auto <url>`, to skip straight to the
# argparse path below instead, for scripting/automation)
# --------------------------------------------------------------------------

class UserQuit(Exception):
    """Raised when the user quits the interactive session entirely: typing
    'q' at the very first wizard step (output_dir -- there's nothing
    earlier to step back to, so 'q' there really does mean quit) or at
    the "another run?" prompt after a scrape finishes, outside the wizard
    altogether. Caught once in main()'s while-loop, so nothing else needs
    its own try/except for a real quit."""
    pass


class GoBack(UserQuit):
    """Raised by the _prompt_* helpers when the user types 'q' at any
    step OTHER than output_dir. interactive_menu()'s step loop catches
    this itself and re-asks the previous step instead of exiting the
    program -- that's the whole point of splitting this out from
    UserQuit.

    It's a UserQuit *subclass*, not a fully separate exception, so that a
    GoBack raised from a _prompt_* call OUTSIDE the wizard (e.g. the
    "another run?" prompt in main(), which isn't part of the step stack
    and has nowhere to step back to) still unwinds all the way out to
    main()'s `except UserQuit` and quits -- without the shared _prompt_*
    helpers needing to know whether they're being called from inside the
    wizard or not."""
    pass

def _print_banner():
    print(rule())
    print("  📰  News Article Scraper")
    print(rule())


def _prompt_choice(question, options):
    """options: list of (key, label) tuples. Returns the chosen key.

    'q' is always accepted here too, on top of whatever's in options --
    typing it raises GoBack (a step back, not a full quit -- see GoBack's
    docstring) rather than being treated as an invalid choice, so
    quitting doesn't need to be an explicit option on every single call
    site."""
    print(f"\n{wrap(question)}")
    for key, label in options:
        prefix = f"  {key}) "
        # subsequent_indent is spaces matching the prefix's length, not
        # the prefix text itself -- a wrapped continuation line should
        # align under the label, not repeat "  review) " on every line.
        print(textwrap.fill(label, width=half_width(),
                             initial_indent=prefix,
                             subsequent_indent=" " * len(prefix)))
    print("  q) Back")
    valid_keys = [k.lower() for k, _ in options]
    while True:
        choice = input("> ").strip().lower()
        if choice == "q":
            raise GoBack
        if choice in valid_keys:
            return choice
        print(f"   Please enter one of: {', '.join(k for k, _ in options)}, q")


def _prompt_urls(prompt_text):
    print(f"\n{wrap(prompt_text)}")
    print(wrap("(paste one or more URLs -- comma-separated, or one per line; "
                "blank line when done, q to go back)"))
    urls = []
    while True:
        line = input("> ").strip()
        if line.lower() == "q":
            raise GoBack
        if not line:
            if urls:
                break
            print("   Need at least one URL.")
            continue
        urls.extend(p.strip() for p in line.split(",") if p.strip())
    return urls


def _prompt_language_codes():
    """Wizard step for mode 3 ("scrape by language") -- typed codes are
    resolved straight to a list of listing/section URLs via
    resolve_listing_urls_for_languages(), so the rest of the pipeline
    downstream never has to know this run started from language codes
    instead of pasted URLs; it's still just an "auto" mode run with a
    given set of listing pages, exactly like mode 1.

    Loops (re-asking, not GoBack) on a malformed code or a code nothing
    is currently configured for -- those are recoverable typos/mismatches
    worth a second try in place, unlike a real 'q' which still goes back
    a step as usual."""
    print(f"\n{wrap('Which language(s) do you want to scrape? Enter ISO 639-1 codes, comma-separated if more than one (e.g. is, or is,cy).')}")
    known = known_locked_languages()
    print(wrap("Currently configured: " +
               (", ".join(known) if known else "(none yet -- add listing_urls to a locked LANGUAGE_OVERRIDES entry in language_detection.py first)")))
    while True:
        line = input("> ").strip().lower()
        if line == "q":
            raise GoBack
        codes = [c.strip() for c in line.split(",") if c.strip()]
        if not codes or not all(re.fullmatch(r"[a-z]{2}", c) for c in codes):
            print("   Enter one or more 2-letter codes, comma-separated (e.g. is,cy).")
            continue
        urls = resolve_listing_urls_for_languages(codes)
        if not urls:
            print(f"   No locked domain currently serves {', '.join(codes)}. "
                  f"Configured: {', '.join(known) if known else '(none)'}.")
            continue
        print(f"   Found {len(urls)} listing page(s) across locked domains for {', '.join(codes)}.")
        return urls


def _prompt_yes_no(question, default=False):
    suffix = "Y/n" if default else "y/N"
    print(wrap(question))
    while True:
        ans = input(f"[{suffix}/q] ").strip().lower()
        if ans == "q":
            raise GoBack
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("   Please answer y, n, or q.")


def interactive_menu():
    _print_banner()

    # Step-stack wizard: `history` holds the names of steps actually
    # visited, in order, so a GoBack can pop back to whichever step was
    # really asked before this one -- including skipping over
    # "lemmatize_lang" on the way back if it was never asked (do_lemmatize
    # was No), since it's simply never on the stack in that case. `state`
    # holds each step's collected answer, keyed by step name, so
    # re-visiting a step on the way back doesn't lose the OTHER answers
    # already given.
    history = []
    state = {}
    step = "output_dir"

    while step != "done":
        history.append(step)
        try:
            if step == "output_dir":
                # The one real quit point -- asked FIRST (used to be
                # asked last) since clear/delete need to know which
                # output dir's files to act on, and both are reachable
                # from the mode-choice step below, so the directory has
                # to be known before that step runs, not after. There's
                # nothing before this step, so 'q' here raises UserQuit
                # directly rather than GoBack -- see UserQuit vs GoBack's
                # docstrings for why that split exists.
                output_dir = input(f"Output directory [{DEFAULT_OUTPUT}] (q to quit): ").strip()
                if output_dir.lower() == "q":
                    raise UserQuit
                state["output_dir"] = output_dir or DEFAULT_OUTPUT
                step = "mode"

            elif step == "mode":
                output_dir = state["output_dir"]
                mode_key = _prompt_choice(
                    "What would you like to do?",
                    [("1", "Auto-discover articles from a listing/section page"),
                     ("2", "Scrape specific article URL(s) directly"),
                     ("3", "Scrape by language code(s) -- auto-picks known locked "
                           "outlets' listing pages, no URLs needed"),
                     ("review", f"Review boilerplate_candidates.json for '{output_dir}' and promote approved patterns into boilerplate_patterns.py"),
                     ("backfill", f"Backfill corpus_metadata.jsonl for already-saved articles under '{output_dir}' scraped before this feature existed"),
                     ("clear", f"Delete downloaded article .txt files for '{output_dir}' (raw + lemmatized; resets scraped_urls.txt too)"),
                     ("delete", f"Delete boilerplate_candidates.json for '{output_dir}' (the LLM review log)")],
                )
                # review/clear/delete are one-off maintenance actions, not
                # a mode to scrape in -- do the action, then re-show this
                # SAME step rather than advancing. Popping the just-
                # appended "mode" back off history before looping keeps
                # the stack accurate: re-entering the top of the while
                # loop appends "mode" again, so it isn't double-counted.
                if mode_key == "review":
                    run_review(output_dir, boilerplate_patterns)
                    history.pop()
                    continue
                if mode_key == "backfill":
                    added = backfill_missing(output_dir)
                    print(f"   ✅ Added {added} row(s) to corpus_metadata.jsonl "
                          f"(already-covered URLs were skipped).")
                    history.pop()
                    continue
                if mode_key == "clear":
                    clear_downloaded_articles(output_dir)
                    history.pop()
                    continue
                if mode_key == "delete":
                    delete_boilerplate_log(output_dir)
                    history.pop()
                    continue
                if mode_key == "3":
                    # Still an "auto" run under the hood -- language_codes
                    # resolves straight to listing-page URLs, same shape
                    # of input run_auto() already expects from mode 1.
                    # The distinction only matters up here in the wizard.
                    state["mode"] = "auto"
                    step = "language_codes"
                    continue
                state["mode"] = "auto" if mode_key == "1" else "url"
                step = "urls"

            elif step == "language_codes":
                state["urls"] = _prompt_language_codes()
                step = "lemmatize_yn"

            elif step == "urls":
                mode = state["mode"]
                state["urls"] = _prompt_urls(
                    "Enter the listing/section page URL(s):" if mode == "auto"
                    else "Enter the article URL(s):"
                )
                step = "lemmatize_yn"

            elif step == "lemmatize_yn":
                state["do_lemmatize"] = _prompt_yes_no(
                    "\nLemmatize the saved text (reduce every word to dictionary form)?",
                    default=False,
                )
                # lemmatize_lang only gets a turn on the stack -- and
                # therefore only gets landed on by a later GoBack -- when
                # it was actually asked.
                step = "lemmatize_lang" if state["do_lemmatize"] else "boilerplate"

            elif step == "lemmatize_lang":
                lemmatize_lang = input("   Stanza language code (e.g. is, en, cy) (q to go back): ").strip()
                if lemmatize_lang.lower() == "q":
                    raise GoBack
                state["lemmatize_lang"] = lemmatize_lang or None
                step = "boilerplate"

            elif step == "boilerplate":
                state["detect_boilerplate"] = _prompt_yes_no(
                    "\nUse Gemini's free API tier to flag possible leftover boilerplate for review? "
                    "(only articles that look suspiciously short/list-shaped are checked, and those "
                    "are sent in batches rather than one call per article, to go easier on the free "
                    "tier's per-minute request cap; requires GEMINI_API_KEY; you'll be asked to "
                    "approve or reject each candidate at the end of THIS run, and nothing is ever "
                    "auto-applied to boilerplate_patterns.py without that)",
                    default=False,
                )
                step = "language_llm"

            elif step == "language_llm":
                state["detect_language_llm_flag"] = _prompt_yes_no(
                    "\nAllow Gemini to break ties when the local language-ID panel "
                    "(GlotLID, OpenLID-v3, lingua) genuinely disagrees on an "
                    "article? Only called for the disputed subset, not every article "
                    "-- same free tier/key as boilerplate detection above.",
                    default=False,
                )
                step = "done"

        except GoBack:
            # Discard the step we just backed out of, then pop the step
            # before it off the stack to become current again -- it'll be
            # re-appended at the top of the next iteration, so this isn't
            # a permanent removal, just "rewind the stack pointer by one".
            history.pop()
            if not history:
                # Backed out past the very first step -- nothing earlier
                # to return to, so this IS a real quit (the "except when
                # it should" case: only output_dir raises UserQuit
                # directly, but backing all the way past mode/urls/etc.
                # in a row lands here too, and should behave the same
                # way).
                raise UserQuit
            step = history.pop()

    output_dir = state["output_dir"]
    mode = state["mode"]
    urls = state["urls"]
    lemmatize_lang = state.get("lemmatize_lang")
    detect_boilerplate = state["detect_boilerplate"]
    detect_language_llm_flag = state["detect_language_llm_flag"]

    print(f"\n{rule('=')}")
    print(f"  Mode:       {'Auto-discover' if mode == 'auto' else 'Direct URL(s)'}")
    print(f"  URLs:       {len(urls)} given")
    print(f"  Lemmatize:  {lemmatize_lang if lemmatize_lang else 'No'}")
    print(f"  Detect boilerplate (LLM): {'Yes' if detect_boilerplate else 'No'}")
    print(f"  Language ID Gemini tiebreak: {'Yes' if detect_language_llm_flag else 'No'}")
    print(f"  Output dir: {output_dir}")
    print(f"{rule('=')}\n")

    return mode, urls, output_dir, lemmatize_lang, detect_boilerplate, detect_language_llm_flag


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    # No arguments at all -> interactive menu. Any arguments -> argparse
    # path below, unchanged, so scripted/automated calls keep working
    # exactly as before.
    if len(sys.argv) == 1:
        delay = 1.5
        # BUGFIX: this used to run interactive_menu() once and exit after
        # the scrape finished, even though it's presented as an
        # interactive session -- meant doing a second run required
        # relaunching the script from scratch. Now loops back to the menu
        # after each run and asks whether to go again, until you say no.
        # UserQuit is only raised for a REAL quit now: typing 'q' at the
        # very first wizard step (output_dir), backing out past it via a
        # string of GoBacks, or 'q' at the "another run?" prompt below.
        # Every other 'q' inside interactive_menu() raises GoBack instead,
        # which is caught internally by the wizard's own step loop and
        # turned into "re-ask the previous step" -- it never reaches this
        # except block. Since GoBack subclasses UserQuit, this one
        # except still catches both: a GoBack from the "another run?"
        # prompt (which isn't part of the wizard's step stack) unwinds
        # here exactly like a real UserQuit would.
        try:
            while True:
                (mode, urls, output_dir, lemmatize_lang, detect_boilerplate,
                 detect_language_llm_flag) = interactive_menu()
                if mode == "auto":
                    asyncio.run(run_auto(urls, output_dir, lemmatize_lang, delay,
                                          detect_boilerplate=detect_boilerplate,
                                          detect_language_llm_flag=detect_language_llm_flag))
                else:
                    asyncio.run(run_url_mode(urls, output_dir, lemmatize_lang, delay,
                                              detect_boilerplate=detect_boilerplate,
                                              detect_language_llm_flag=detect_language_llm_flag))
                if not _prompt_yes_no("\nBack to the main menu for another run?", default=True):
                    print("\n👋 Bye!")
                    break
        except UserQuit:
            print("\n👋 Bye!")
        return

    parser = argparse.ArgumentParser(description="Generic news article scraper")
    sub = parser.add_subparsers(dest="mode", required=True)

    auto_p = sub.add_parser("auto", help="Crawl listing/section pages, auto-discover and scrape articles")
    auto_p.add_argument("urls", nargs="+", help="One or more listing/section page URLs")
    auto_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    auto_p.add_argument("--lemmatize", default=None, help="Stanza language code, e.g. en, cy, is")
    auto_p.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between requests (politeness/rate-limit avoidance, default 1.5)")
    auto_p.add_argument("--detect-boilerplate", action="store_true",
                         help="Use Gemini's free API tier to flag possible leftover boilerplate per article for review "
                              "(logs candidates to boilerplate_candidates.json; at the end of THIS run you'll be "
                              "prompted per candidate to approve or reject before it's promoted into "
                              "boilerplate_patterns.py -- never auto-applied; requires GEMINI_API_KEY)")
    auto_p.add_argument("--detect-language-llm", action="store_true",
                         help="Get a second language-ID opinion from Gemini alongside the built-in lingua "
                              "detector (same free tier/key as --detect-boilerplate); on disagreement, "
                              "Gemini's answer is used and the split is logged to language_disagreements.json")

    url_p = sub.add_parser("url", help="Scrape one or more specific article URLs directly")
    url_p.add_argument("urls", nargs="+", help="One or more direct article URLs")
    url_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    url_p.add_argument("--lemmatize", default=None, help="Stanza language code, e.g. en, cy, is")
    url_p.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between requests (politeness/rate-limit avoidance, default 1.5)")
    url_p.add_argument("--detect-boilerplate", action="store_true",
                        help="Use Gemini's free API tier to flag possible leftover boilerplate per article for review "
                             "(logs candidates to boilerplate_candidates.json; at the end of THIS run you'll be "
                             "prompted per candidate to approve or reject before it's promoted into "
                             "boilerplate_patterns.py -- never auto-applied; requires GEMINI_API_KEY)")
    url_p.add_argument("--detect-language-llm", action="store_true",
                        help="Get a second language-ID opinion from Gemini alongside the built-in lingua "
                             "detector (same free tier/key as --detect-boilerplate); on disagreement, "
                             "Gemini's answer is used and the split is logged to language_disagreements.json")

    clear_p = sub.add_parser("clear", help="Delete downloaded article .txt files (raw + lemmatized) and reset scraped_urls.txt")
    clear_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    clear_p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    delete_p = sub.add_parser("delete", help="Delete boilerplate_candidates.json (the LLM review-queue log)")
    delete_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    delete_p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")

    review_p = sub.add_parser("review", help="Interactively review boilerplate_candidates.json and promote approved patterns into boilerplate_patterns.py")
    review_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)

    backfill_p = sub.add_parser("backfill-metadata", help="Compute corpus_metadata.jsonl rows for already-saved articles scraped before this feature existed "
                                 "(word/sentence counts, TTR, URL topic hint -- extraction method and language-panel vote breakdown aren't recoverable "
                                 "from disk alone and are left null for these rows; see corpus_metadata.py)")
    backfill_p.add_argument("--output-dir", default=DEFAULT_OUTPUT)

    args = parser.parse_args()

    if args.mode == "auto":
        asyncio.run(run_auto(args.urls, args.output_dir, args.lemmatize, args.delay,
                              detect_boilerplate=args.detect_boilerplate,
                              detect_language_llm_flag=args.detect_language_llm))
    elif args.mode == "url":
        asyncio.run(run_url_mode(args.urls, args.output_dir, args.lemmatize, args.delay,
                                  detect_boilerplate=args.detect_boilerplate,
                                  detect_language_llm_flag=args.detect_language_llm))
    elif args.mode == "clear":
        clear_downloaded_articles(args.output_dir, skip_confirm=args.yes)
    elif args.mode == "delete":
        delete_boilerplate_log(args.output_dir, skip_confirm=args.yes)
    elif args.mode == "review":
        run_review(args.output_dir, boilerplate_patterns)
    elif args.mode == "backfill-metadata":
        added = backfill_missing(args.output_dir)
        print(f"   ✅ Added {added} row(s) to corpus_metadata.jsonl "
              f"(already-covered URLs were skipped).")


if __name__ == "__main__":
    main()