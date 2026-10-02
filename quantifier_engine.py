"""
quantifier_engine.py
=====================
Detection for the quantifier_* branch (see quantifier_tables.py for the
linguistics): after a quantity word + "o" ("llawer o", "digon o", "lot o",
and since 2026-10-02 the partitives "un o'r", "rhai o'r", "dau o'r", "y rhan
fwyaf o'r"), a count noun should be plural in both Welsh and English -- the
second predicted-to-RESIST branch, alongside plural_engine.py ("rhai" +
plural); the analyzer also reports the two as one combined control.

Same gates as plural_engine.py and numeral_engine.py, applied to correct
and eroded cases alike so none can tilt the rate:
  - the quantifier must be followed (skipping hesitations, never across a
    comma or utterance end) by "o", not tagged as a pronoun;
  - the noun after "o" (article/clitic tokens skipped) must pass
    spacy_tagging.is_noun_target(), must not be code-switched or a
    discourse particle, and must not start a self-correction
    (noun_phrase_interrupted);
  - its number must come from the Bangor lexicon (number_source "lexicon");
  - mass nouns are excluded by lemma (MASS_NOUN_LEMMAS: "llawer o waith"
    is singular in English too), and so is any noun the lexicon never
    lists in the plural -- there was no singular/plural choice to make.
    Both exclusions are by noun, so they remove singular and plural
    outcomes alike.

Rows: plural -> correct; a collective noun ("llawer o bobl") -> correct;
singular -> erosion_unverified, a candidate for manual review only (see
the comment at that branch: in speech these were all mass or degree
readings, singular in English too). Every row records the quantifier and whether
it is the English loan "lot" (is_loan_quantifier), so results can be
reported with and without it.

Does not use mark_consumed(): the noun is routinely also a mutation-branch
target (soft mutation after "o"), a separate phenomenon.

Imports only shared infrastructure (spacy_tagging.py, bangor_lexicon.py),
never another branch.
"""
import bangor_lexicon
from spacy_tagging import is_noun_target, noun_number, noun_phrase_interrupted
from quantifier_tables import (QUANTIFIER_FORMS, LOAN_QUANTIFIERS, LINKING_WORDS,
                               ARTICLE_FORMS, MASS_NOUN_LEMMAS, COLLECTIVE_NOUN_LEMMAS,
                               WELSH_FILLERS, DISCOURSE_PARTICLES)


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("‘", "'").replace("’", "'") \
                     .replace("“", '"').replace("”", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def _next_real(words_list, idx, max_ahead=3):
    """Index of the next word after idx that isn't synthetic or a hesitation,
    or None at a clause/utterance boundary or beyond max_ahead."""
    if words_list[idx].get("_clause_boundary_after"):
        return None
    k = idx + 1
    while k < len(words_list) and k - idx <= max_ahead:
        node = words_list[k]
        if node.get("synthetic") or normalize_word(node["word"]) in WELSH_FILLERS:
            if node.get("_clause_boundary_after"):
                return None
            k += 1
            continue
        return k
    return None


def _build_quantifier_row(current_node, target_node, quantifier, status, is_erosion,
                          number_found, number_source, lemma, note):
    return {
        "timestamp":            f"{current_node.get('start')}s - {target_node.get('end')}s",
        "quantifier":           quantifier,
        "quantifier_surface":   normalize_word(current_node["word"]),
        "is_loan_quantifier":   quantifier in LOAN_QUANTIFIERS,
        "following_word":       normalize_word(target_node["word"]),
        "noun_lemma":           lemma,
        "number_found":         number_found,
        "number_source":        number_source,
        "status":               status,
        "is_erosion":           is_erosion,
        "note":                 note,
        "trigger_confidence":   current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "rule":                 "plural_after_quantifier",
    }


def process_quantifier_plurals(words_list):
    rows = []
    for i, current_node in enumerate(words_list):
        if current_node.get("synthetic") or current_node.get("confidence", 0.0) < 0.65:
            continue
        quantifier = QUANTIFIER_FORMS.get(normalize_word(current_node["word"]))
        if quantifier is None:
            continue

        link_idx = _next_real(words_list, i, max_ahead=2)
        if link_idx is None:
            continue
        link = words_list[link_idx]
        if normalize_word(link["word"]) not in LINKING_WORDS or \
                (link.get("spacy_token") or {}).get("pos") == "PRON":
            continue   # "llawer gwell" (much better), or "o" = "he"

        target_idx = _next_real(words_list, link_idx)
        if target_idx is not None and normalize_word(words_list[target_idx]["word"]) in ARTICLE_FORMS:
            target_idx = _next_real(words_list, target_idx)   # "o (y)r plant" -- see ARTICLE_FORMS
        if target_idx is None:
            continue
        target = words_list[target_idx]
        if target.get("confidence", 0.0) < 0.65 or target.get("_code_switch"):
            continue
        target_norm = normalize_word(target["word"])
        if target_norm in DISCOURSE_PARTICLES or not is_noun_target(target):
            continue
        if noun_phrase_interrupted(words_list, i, target):
            continue

        lemmas = bangor_lexicon.noun_lemmas(target_norm)
        if not lemmas or lemmas & MASS_NOUN_LEMMAS:
            continue   # unknown to the lexicon, or a mass noun -- see module docstring
        lemma = sorted(lemmas)[0]
        number, number_source = noun_number(target)
        if number_source != "lexicon":
            continue   # dictionary-confirmed number only -- see numeral_engine.py

        if lemmas & COLLECTIVE_NOUN_LEMMAS or number == "plural":
            kind = "Plural" if number == "plural" else "Collective (plural-meaning)"
            rows.append(_build_quantifier_row(
                current_node, target, quantifier, "correct_mutation", False, number,
                number_source, lemma, f"{kind} noun after '{quantifier} o': {target_norm}"))
        elif number == "singular":
            # A noun never listed in the plural had no choice to make.
            if not any(bangor_lexicon.noun_lemma_has_plural(l) for l in lemmas):
                continue
            # NOT counted as erosion automatically. In the first full Siarad
            # run (2026-09-28) all ~78 singular rows were grammatical in
            # English too: abstract nouns used as mass nouns ("llawer o
            # wahaniaeth" = a lot of difference, "cymaint o bwys", "mwy o
            # ymdrech") -- the dictionary lists plurals for these, so no
            # mass-noun list can cover them -- or degree readings ("gormod o
            # babi" = too much of a baby, "mwy o hogyn" = more of a lad).
            # The four checked by hand against the transcript were all such
            # readings, none a slip. So a singular is a CANDIDATE for manual
            # review, not a finding: erosion_unverified is outside
            # EVALUABLE_STATUSES, and only a hand-confirmed slip should be
            # counted.
            rows.append(_build_quantifier_row(
                current_node, target, quantifier, "erosion_unverified", False, number,
                number_source, lemma,
                f"Singular after '{quantifier} o': {target_norm} -- a mass/abstract or "
                f"degree reading ('too much of a baby') unless checked by hand to be "
                f"a slip; not counted"))
    return rows
