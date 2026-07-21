"""
cysill_client.py
=================
Everything about talking to the Cysill (techiaith.cymru) API: connection
handling, retries, rate-limit handling, the circuit breaker that disables
Cysill for the rest of a run after repeated failures, and the two things
we actually ask it for -- POS tags and lemmas.

Nothing in here knows about mutations, Whisper, or CSVs. If a call site
needs a POS tag or a lemma for a chunk of Welsh text, this is the file
that gets it for them (falling back to spaCy/simplemma happens one layer
up, in mutation_engine.py).
"""
import os
import random
import time

import requests
from tqdm import tqdm

from mutation_tables import MUTATION_TAG_MAP, GENDER_BEARING_PREFIXES

TECHIAITH_API_KEY         = os.getenv("WELSH_LEMMATIZER", "").strip()
WELSH_LEMMATIZER_ENDPOINT = "https://api.techiaith.cymru/cysill/v1/lemmatizer/v1"
WELSH_POS_ENDPOINT        = "https://api.techiaith.cymru/cysill/v1/pos/v1"

# How big a chunk of preprocessed words we'll hand to fetch_pos_for_chunk()
# in one request. Lives here (not in mutation_engine.py) because it's a
# Cysill API limit, not a linguistic constant.
POS_CHUNK_MAX_CHARS = 2700

# PATCH: distinguishable sentinel for "the call itself failed" (rate limit,
# timeout, circuit breaker open, unparseable response) as opposed to a
# genuine `None`/empty result, which means the API was reached and
# responded, it just had nothing to offer. Callers (currently just
# get_welsh_lemma) need this distinction to decide what's safe to cache
# permanently vs. what should be retried on a future run. Using a sentinel
# object rather than a string/None avoids any risk of colliding with a
# real API response value.
CYSILL_CALL_FAILED = object()

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
    interactive choice (e.g. choice 1 -> local MP3s) and the user then picks
    choice 3 (queue processing) in the same session, the breaker would remain
    open and silently disable Cysill for the second run.  Call this at the
    start of each top-level processing branch in main() to give each run a
    clean slate.
    """
    global _cysill_consecutive_failures, _cysill_disabled_for_run
    _cysill_consecutive_failures = 0
    _cysill_disabled_for_run     = False


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
                # BUGFIX: Retry-After is technically allowed by HTTP to be
                # either delta-seconds ("120") or an HTTP-date
                # ("Wed, 21 Oct 2015 07:28:00 GMT"). A bare int(...) on a
                # date string raises ValueError, which used to fall through
                # to the generic `except Exception` below and get logged as
                # a misleading "Request error" -- functionally harmless
                # (it still backed off and retried) but it silently threw
                # away the server's actual guidance and mislabeled the
                # cause. Parsed defensively here instead: try int, then
                # float-rounded-to-int, then fall back to our own backoff
                # for anything else (e.g. a date string) without raising.
                retry_after_header = resp.headers.get("Retry-After")
                retry_after = None
                if retry_after_header is not None:
                    try:
                        retry_after = int(retry_after_header)
                    except ValueError:
                        try:
                            retry_after = int(float(retry_after_header))
                        except ValueError:
                            retry_after = None  # e.g. an HTTP-date -- not parsed, use default below
                if retry_after is None:
                    retry_after = int(base_delay * attempt * 2)
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
        # PATCH: this is a genuine call failure (retries exhausted, or the
        # circuit breaker is open) -- not the same thing as the API
        # answering "no lemma". Distinguish it so the caller doesn't
        # permanently cache a word that we simply never managed to ask.
        return CYSILL_CALL_FAILED
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("success") is True:
            return data.get("result")
        tqdm.write(f" ⚠️ Lemma API success=False: {data}")
    except Exception as e:
        tqdm.write(f" ⚠️ Lemma API parse error: {e}")
        # PATCH: a response we couldn't parse is closer to "we didn't get
        # a usable answer" than "the API confirmed there's nothing" --
        # treat it the same as a call failure so it gets retried later
        # rather than calcified as a permanent null.
        return CYSILL_CALL_FAILED
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