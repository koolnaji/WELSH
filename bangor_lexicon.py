r"""
bangor_lexicon.py
==================
Local, offline lookup against Techiaith's own Bangor lexicon
(github.com/techiaith/lecsicon-cymraeg-bangor, CC0 license) -- ~830k
wordform-level entries (wordform, lemma, UD POS, UD morphological
features), generated directly from Cysill's own internal lexicon data.
This is NOT a third-party approximation with its own reliability
tradeoff -- it's the same institution's own data, released as a static
file specifically "in the hope that it will stimulate the development
of Welsh language technologies" (repo README). Confirmed against the
live file (cloned 2026-08, dated Mar 2025 per the archive's internal
timestamp): 496,818 unique wordforms, of which 96.3% resolve to a
single lemma even when multiple POS/morph readings exist for the same
surface form (measured directly against the downloaded file, not an
estimate).

Why this exists: the Cysill POS endpoint (fetch_pos_for_chunk() in
cysill_client.py) is the one actually hitting the 429 / 3600s-Retry-
After rate limit in real runs. Every wordform this lookup can resolve
WITHOUT ambiguity is one word (or, at the chunk level, potentially one
whole API call) that doesn't need to go through that endpoint at all.
See get_welsh_lemma() and enrich_words() in mutation_engine.py for the
two integration points -- lemma lookup (this module's biggest win,
96.3% coverage) and whole-chunk POS/mutation-type/gender resolution (a
more conservative, all-or-nothing-per-chunk win -- see enrich_words()'s
own comments for why it's gated that way).

*** WHAT THIS MODULE DELIBERATELY DOES NOT DO ***
Cysill's live POS endpoint returns POS tags in Cysill's OWN compact tag
scheme (the kind of string cysill_coarse_pos() and pos_compatible() in
mutation_engine.py pattern-match against, and the scheme
GENDER_BEARING_PREFIXES / MUTATION_TAG_MAP in cysill_client.py are keyed
to). This lexicon uses Universal Dependencies tags instead -- the repo
README says so explicitly: the data has been "converted to use part of
speech tags and features based on the Universal Dependencies framework".
There is no published mapping from one scheme to the other, and
guessing one is exactly the kind of silent-scheme-mismatch bug this
project has already been bitten by more than once (see mutation_engine
.py's own comments on the tagger_agreement / DOM-Vnoun-conflation bugs
-- both were exactly this shape of error: two fields that LOOK
comparable but aren't, silently compared as if they were). So:

  - cysill_pos is NEVER populated from this module -- only ever left
    None, the same value it already has whenever Cysill itself fails or
    is disabled for a word. Every downstream consumer already has to
    tolerate that value, by design (three-layer detection: spaCy
    primary, Cysill secondary, heuristic tertiary).
  - cysill_mutation_type and cysill_gender ARE populated from this
    lexicon, because those two fields already store TRANSLATED,
    scheme-independent values (friendly strings like "soft"/"nasal"/
    "aspirate"/"h-mutation", and "feminine"/"masculine") -- the exact
    same target vocabulary spaCy's own SPACY_MUTATION_MAP already
    translates into (see BANGOR_MUTATION_MAP below, built FROM
    SPACY_MUTATION_MAP rather than duplicating its strings). Reusing
    the shared vocabulary, not the raw tag scheme, is what makes this
    safe.

Usage:
    1. Download the data (~4MB zipped, ~56MB unzipped):
       https://github.com/techiaith/lecsicon-cymraeg-bangor
       -> lecsicon_cc0.zip -> unzip -> lecsicon_cc0.txt
    2. Place it wherever you like and either pass the path to load(), or
       set the BANGOR_LEXICON_PATH environment variable before the first
       call -- loading is lazy and happens once per process either way.
"""
import os
import re
from pathlib import Path

from mutation_tables import MUTATION_TAG_MAP, SPACY_MUTATION_MAP

# Cross-referenced from mutation_tables.py rather than hardcoded, so this
# stays in sync automatically if the friendly mutation-type vocabulary
# ever changes there -- one source of truth for what these strings
# actually are, same discipline as EVALUABLE_STATUSES living in exactly
# one place. HM (h-mutation) is the one mutation category the lexicon
# tags that SPACY_MUTATION_MAP doesn't cover (spaCy's own Mutation
# feature apparently never surfaces HM in this project's usage so far --
# TH is Cysill's own code for the same thing, per MUTATION_TAG_MAP).
BANGOR_MUTATION_MAP = dict(SPACY_MUTATION_MAP)          # SM/NM/AM -> soft/nasal/aspirate
BANGOR_MUTATION_MAP["HM"] = MUTATION_TAG_MAP["TH"]       # HM -> "h-mutation"

DEFAULT_LEXICON_PATH = Path(os.getenv("BANGOR_LEXICON_PATH", "bangor_lexicon/lecsicon_cc0.txt"))

# Matches one UD-style feature=value pair at a time, INCLUDING comma-
# separated multi-values like "Gender=Fem,Masc" (real, attested Welsh
# epicene nouns -- "aberth", "abid" -- confirmed against the actual
# file). A narrower [A-Za-z]+-only value pattern would silently truncate
# "Fem,Masc" down to just "Fem" and misreport a genuinely dual-gender
# noun as unambiguously feminine -- caught before shipping by checking
# the real data first, not assumed.
_MORPH_FEATURE_RE = re.compile(r"([A-Za-z]+)=([A-Za-z]+(?:,[A-Za-z]+)*)")

_lexicon = None  # wordform -> list of {"lemma", "pos", "morph"} dicts, loaded lazily


def _parse_morph(morph_str):
    return dict(_MORPH_FEATURE_RE.findall(morph_str)) if morph_str else {}


def load(path=None):
    """
    Loads the lexicon into memory (module-level, once per process).
    Safe to call repeatedly -- a no-op after the first successful load,
    regardless of which path triggered it.

    Raises FileNotFoundError with the download URL in the message if the
    file isn't where expected, rather than silently building an empty
    lexicon and letting every lookup() quietly degrade to "nothing is
    ever resolved" -- that failure mode would be far harder to notice
    (everything still runs, just slower and hitting Cysill as much as
    before) than a loud one at startup.
    """
    global _lexicon
    if _lexicon is not None:
        return

    lex_path = Path(path) if path else DEFAULT_LEXICON_PATH
    if not lex_path.exists():
        raise FileNotFoundError(
            f"Bangor lexicon not found at {lex_path}. Download it from "
            f"https://github.com/techiaith/lecsicon-cymraeg-bangor "
            f"(lecsicon_cc0.zip -> unzip -> lecsicon_cc0.txt) and place it "
            f"there, or pass an explicit path to load(), or set the "
            f"BANGOR_LEXICON_PATH environment variable."
        )

    lexicon = {}
    with open(lex_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue  # malformed line -- skip rather than abort the whole load
            wordform, lemma, pos = parts[0], parts[1], parts[2]
            morph = _parse_morph(parts[3]) if len(parts) > 3 else {}
            lexicon.setdefault(wordform, []).append(
                {"lemma": lemma, "pos": pos, "morph": morph})

    _lexicon = lexicon


def is_loaded():
    return _lexicon is not None


def wordform_count():
    """0 if not loaded, else the number of unique wordforms -- exists so
    callers (load_bangor_lexicon() in mutation_engine.py) can print a
    real confirmation number without reaching into the private
    _lexicon module variable directly."""
    return len(_lexicon) if _lexicon is not None else 0


def lookup(word):
    """
    Returns the list of lexicon entries for this EXACT wordform. Empty
    list if the lexicon isn't loaded yet, or if the word is genuinely
    OOV. No normalization applied beyond the case-recovery fallback
    below -- this project already has its own normalize_word() with its
    own conventions (stripped punctuation, curly-quote handling, "'r"/
    "'n" expansion), and this module shouldn't duplicate or second-guess
    that; callers own normalization, this just looks up whatever string
    they hand it.

    Falls back to a Title-cased retry ONLY when the exact form misses --
    the lexicon stores proper nouns capitalized (e.g. "Cymru", not
    "cymru"), but this project's normalize_word() lowercases everything
    before words ever reach here, so a straight lowercase-only lookup
    would silently miss every capitalized entry in the lexicon (not a
    small gap -- 8,852 PROPN entries in the file). This fallback is a
    case-recovery step, not a content guess: it doesn't change which
    letters the word has, just retries assuming the leading
    capitalization the caller's own normalization already stripped.
    """
    if _lexicon is None:
        return []
    hit = _lexicon.get(word)
    if hit:
        return hit
    if word and word[0].islower():
        return _lexicon.get(word[0].upper() + word[1:], [])
    return []


def lemma_if_unambiguous(word):
    """
    Returns a single lemma (lowercased, matching this project's
    LEMMA_CACHE convention) if EVERY reading for this wordform agrees on
    lemma -- even when readings disagree on POS/morph, since that's
    still an unambiguous answer to "what's the lemma" specifically
    (measured: 96.3% of wordforms in the lexicon are lemma-unambiguous
    this way). Returns None for OOV words, and for the ~3.7% that are
    genuinely lemma-ambiguous -- e.g. "clo", which is either the noun
    "clo" (lock) or an inflected form of the verb "cloi" (to lock)
    depending on context this lookup doesn't have -- those defer to
    Cysill/simplemma exactly as before, since picking one candidate
    lemma over another here would be a guess dressed up as a lookup.
    """
    entries = lookup(word)
    if not entries:
        return None
    lemmas = {e["lemma"].lower() for e in entries}
    return next(iter(lemmas)) if len(lemmas) == 1 else None


def resolved_tag_if_unambiguous(word):
    """
    Returns {"mutation_type": ..., "gender": ..., "number": ...} if this
    wordform has EXACTLY ONE reading in the lexicon -- not just lemma-
    unambiguous, POS/morph-unambiguous too, a stricter bar than
    lemma_if_unambiguous() needs since this stands in for Cysill's own
    per-token tag output rather than just a lemma string. Returns None
    for OOV or multi-reading words -- deliberately conservative, since a
    wrong guess here feeds directly into mutation classification and
    confidence scoring, not just a corpus-analysis lemma column.

    mutation_type/gender use the SAME friendly-string vocabulary
    cysill_mutation_type/cysill_gender already store (see module
    docstring) -- safe to assign directly to those fields. Does NOT
    return a "pos" key -- see module docstring for why cysill_pos is
    never populated from here. Gender=Fem,Masc (real, epicene Welsh
    nouns) is treated the same as "no gender feature at all": Cysill
    itself couldn't hand back a single answer for that word either, so
    None here matches what an honest tagger output would look like, not
    a guess in either direction.
    """
    entries = lookup(word)
    if len(entries) != 1:
        return None
    morph = entries[0]["morph"]

    gender_raw = morph.get("Gender", "")
    gender = {"Masc": "masculine", "Fem": "feminine"}.get(gender_raw)  # None if "" or "Fem,Masc"

    number_raw = morph.get("Number", "")
    number = {"Sing": "singular", "Plur": "plural"}.get(number_raw)

    mutation_type = BANGOR_MUTATION_MAP.get(morph.get("Mutation"))

    return {"mutation_type": mutation_type, "gender": gender, "number": number}