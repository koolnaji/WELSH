"""
corpus_metadata.py
===================
Per-article metadata capture for the news corpus -- a single flat,
append-only table (corpus_metadata.jsonl, top level of each output_dir,
same spot as scraped_urls.txt) with one row per saved article, meant to
be loaded straight into pandas/R for exploratory correlation-hunting
across whatever variables the eventual research question turns out to
need: word count, language, a couple of cheap register/complexity
proxies, source/outlet, extraction quality, and the FULL language-panel
vote breakdown.

That last one is worth calling out: language_detection.py's
log_language_dispute() previously only ever ran for DISPUTED verdicts --
a consensus verdict's winner_share (how confidently the panel agreed) was
computed and then thrown away. That's exactly the kind of already-computed,
continuous signal worth keeping for every article, not just the disputed
ones (e.g. it's a plausible proxy for code-switched or borderline text
even when it didn't cross the dispute threshold). record_for_article()
below keeps it either way.

Deliberately scoped to fields computable from data the pipeline ALREADY
has in hand at save time (the extracted text itself, the URL, the
language panel's own verdict) -- no new network calls, no new heavy
dependencies, nothing that meaningfully slows down a scrape. "Register"
in the sociolinguistic sense (formality, genre) isn't something this
computes directly -- that needs either manual coding or a trained
classifier, both real time investments this file doesn't assume are
available before a hard deadline. What it computes instead are standard,
cheap, quantitative PROXIES corpus linguistics actually uses for exactly
that axis: type-token ratio and average sentence/word length as rough
lexical-diversity/complexity signals. Treat them as one plausible axis to
correlate against, not a verified register label -- the honest place to
start when the actual research question isn't picked yet.

Two entry points:
  record_for_article()  -- called once per newly scraped article, from
                            icelandic_text_extractor.py's scrape_one().
  backfill_missing()     -- one-off pass over an output_dir's already-
                            saved raw/*.txt files, computing whatever
                            subset of these fields is recoverable from a
                            saved .txt + its URL alone, for a corpus
                            scraped BEFORE this module existed. Extraction
                            method and the live language-panel vote
                            breakdown are NOT recoverable this way (they
                            only exist at scrape time) -- backfilled rows
                            are flagged ("backfilled": True) rather than
                            silently padded with fabricated values, so
                            they're visibly distinguishable once loaded.
"""

import os
import re
import json
import hashlib
from datetime import datetime, timezone
from urllib.parse import urlparse

from corpus_layout import iter_raw_files

METADATA_FILENAME = "corpus_metadata.jsonl"

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
# Letters only (any Unicode script), excludes bare digit runs and
# underscores -- good enough to tokenize "words" for a rough type-token
# ratio across languages this project already handles (Icelandic þ/ð,
# Welsh mutated forms, etc. are all \w under Python's default Unicode
# matching), not a substitute for a real per-language tokenizer.
_WORD_RE = re.compile(r"\b[^\W\d_]+\b", re.UNICODE)


def compute_text_stats(text):
    """Cheap, dependency-free stats from already-extracted article text.
    Every value here is O(n) over the text with no external calls -- safe
    to run on every single article without meaningfully slowing a scrape.

    Sentence splitting is a blunt regex (.!? followed by whitespace/end),
    not a real sentence tokenizer -- good enough for a rough
    avg_sentence_length signal averaged across a large corpus, not
    precise enough to trust on any one article in isolation. Same honesty
    as this module's docstring on "register": a usable proxy, not a claim
    of linguistic precision.
    """
    stripped = text.strip()
    if not stripped:
        return {
            "char_count": 0, "word_count": 0, "sentence_count": 0,
            "paragraph_count": 0, "avg_sentence_length": 0.0,
            "avg_word_length": 0.0, "type_token_ratio": 0.0,
        }

    words = stripped.split()
    word_count = len(words)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    # A text with no terminal punctuation at all is still one "sentence"
    # for averaging purposes, not zero (which would make
    # avg_sentence_length a division by zero).
    sentence_count = max(1, len(sentences))

    paragraphs = [p for p in stripped.split("\n\n") if p.strip()]

    letter_tokens = [w.lower() for w in _WORD_RE.findall(stripped)]
    type_token_ratio = (len(set(letter_tokens)) / len(letter_tokens)) if letter_tokens else 0.0
    avg_word_length = (sum(len(w) for w in letter_tokens) / len(letter_tokens)) if letter_tokens else 0.0

    return {
        "char_count": len(stripped),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": len(paragraphs) if paragraphs else 1,
        "avg_sentence_length": round(word_count / sentence_count, 2),
        "avg_word_length": round(avg_word_length, 2),
        "type_token_ratio": round(type_token_ratio, 4),
    }


def url_topic_hint(url):
    """First meaningful path segment of the URL, as a free, rough
    section/topic proxy (e.g. ".../sport/football/..." -> "sport") --
    most news CMSs put the section in the URL, so this costs nothing
    beyond a URL parse. Not a substitute for real topic modeling, just
    another cheap column to correlate against while the real research
    question is still being picked.

    Returns None when the path has no real section segment to read: a
    bare single-segment slug, a segment that's mostly a numeric ID, or an
    implausibly long "segment" (almost always the article slug itself
    sitting first in the path, not a section name) -- comparable in
    spirit to looks_like_article()'s own digit/hyphen heuristics in
    icelandic_text_extractor.py.
    """
    path = urlparse(url).path.strip("/")
    if not path:
        return None
    segments = path.split("/")
    first = segments[0].lower()
    if len(segments) == 1 or re.search(r"\d{4,}", first) or len(first) > 30:
        return None
    return first


def metadata_path(output_dir):
    return os.path.join(output_dir, METADATA_FILENAME)


def record_for_article(*, url, domain, language_code, language_locked,
                        language_verdict, extraction_method, was_suspicious,
                        text, lemmatized_lang=None, discovered_via=None):
    """Builds one corpus_metadata.jsonl row for a just-saved article.
    Called once per article from scrape_one(), after extraction and
    language detection have both already run -- every argument here is
    something scrape_one() already has in hand; nothing is recomputed
    except the cheap text stats.

    language_verdict: the LanguageVerdict from detect_language() -- may
    be None when the domain is language_lock'd (the panel never ran at
    all for this article, not just "reached a quiet consensus"; those are
    distinguishable via language_locked=True with language_votes=None).
    When present, its FULL vote breakdown is recorded regardless of
    whether the panel disputed or reached consensus -- see this module's
    docstring for why that's a change from the old dispute-only logging.

    discovered_via: the listing page (or sitemap) this URL was found on in
    auto mode -- None for direct-URL (mode 2) scrapes. When one article is
    linked from several listing pages in the same run, the FIRST listing
    URL passed in wins (see run_auto), so list the most specific pages
    (section/topic pages) before general ones (homepage, sitemap) to get
    the most informative attribution. This is the only topic signal for
    sites whose URLs carry none (newyddion.s4c.cymru: every article is
    /article/..., so url_topic_hint is uninformative there).
    """
    stats = compute_text_stats(text)
    record = {
        "url": url,
        "domain": domain,
        "discovered_via": discovered_via,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "language_code": language_code,
        "language_locked": language_locked,
        "extraction_method": extraction_method,
        "was_suspicious_for_boilerplate_check": was_suspicious,
        "lemmatized_lang": lemmatized_lang,
        "url_topic_hint": url_topic_hint(url),
        "backfilled": False,
        **stats,
    }
    if language_verdict is not None:
        record["language_disputed"] = language_verdict.disputed
        record["language_winner_share"] = round(language_verdict.winner_share, 4)
        record["language_votes"] = {
            name: {"code": v.code, "confidence": round(v.confidence, 4)}
            for name, v in language_verdict.votes.items()
        }
    else:
        record["language_disputed"] = False
        record["language_winner_share"] = None
        record["language_votes"] = None
    return record


def append_metadata_record(output_dir, record):
    with open(metadata_path(output_dir), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_metadata(output_dir):
    """Returns (records list, set of URLs already present). Tolerates a
    missing file (empty result) and skips any individual line that fails
    to parse rather than losing the whole file to one bad line -- a crash
    mid-write is the realistic way a single line ends up truncated, not
    the whole file, same tolerance every other JSON log in this project
    already gives corrupt/partial data."""
    path = metadata_path(output_dir)
    records = []
    if not os.path.exists(path):
        return records, set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    seen_urls = {r.get("url") for r in records if isinstance(r, dict) and r.get("url")}
    return records, seen_urls


def backfill_missing(output_dir):
    """Walks every saved raw .txt (either folder layout -- see
    corpus_layout.iter_raw_files), computes the recoverable-from-disk
    subset of fields for any file whose URL isn't already in
    corpus_metadata.jsonl, and appends those rows. Recovers each file's real URL from scraped_urls.txt (matched by
    the same 8-hex-char url_hash safe_filename() already embeds in every
    filename, the same match reconcile_manifest() uses) rather than
    trying to reverse the slug in the filename, since the manifest is the
    one place the real URL is guaranteed to still be recorded.

    Returns the number of rows added. Safe to re-run -- URLs already
    logged (live or backfilled) are skipped, so running this after every
    future scrape too (not just once) is harmless.
    """
    manifest_path = os.path.join(output_dir, "scraped_urls.txt")
    if not os.path.exists(manifest_path):
        return 0

    hash_to_url = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            url = line.split("\t")[0]
            url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
            hash_to_url[url_hash] = url

    _, seen_urls = load_metadata(output_dir)

    added = 0
    for lang_code, path in iter_raw_files(output_dir):
        match = re.search(r"__([0-9a-f]{8})\.txt$", os.path.basename(path))
        if not match:
            continue
        url = hash_to_url.get(match.group(1))
        if not url or url in seen_urls:
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        stats = compute_text_stats(text)
        record = {
            "url": url,
            "domain": urlparse(url).netloc.replace("www.", ""),
            "discovered_via": None,    # not recoverable from disk alone
            "scraped_at": None,        # not recoverable from disk alone
            "language_code": lang_code,
            "language_locked": None,   # not recoverable from disk alone
            "extraction_method": None,  # not recoverable from disk alone
            "was_suspicious_for_boilerplate_check": None,
            "lemmatized_lang": None,
            "url_topic_hint": url_topic_hint(url),
            "language_disputed": None,
            "language_winner_share": None,
            "language_votes": None,
            "backfilled": True,
            **stats,
        }
        append_metadata_record(output_dir, record)
        seen_urls.add(url)
        added += 1
    return added


if __name__ == "__main__":
    # Standalone entry point so backfilling never needs the scraper's own
    # dependencies (playwright, trafilatura, tqdm) -- icelandic_text_
    # extractor.py imports playwright at module level, so its own
    # backfill-metadata subcommand fails in any Python env that can read
    # a corpus but was never set up to scrape one. This file is stdlib-only.
    import argparse

    parser = argparse.ArgumentParser(description="corpus_metadata.jsonl maintenance (stdlib only)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    backfill_p = sub.add_parser("backfill", help="Add rows for already-saved articles missing from corpus_metadata.jsonl")
    backfill_p.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.cmd == "backfill":
        n = backfill_missing(args.output_dir)
        print(f"Added {n} row(s) to {metadata_path(args.output_dir)} (already-covered URLs skipped).")
