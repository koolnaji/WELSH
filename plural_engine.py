"""
plural_engine.py
==================
Detection for the plural_* branch (see plural_tables.py for the
linguistics): after "rhai" ("some"), the noun should be plural in both
Welsh and English -- the predicted-to-RESIST comparison point for the
numeral_* branch.

Detects PRESENCE of plural marking only (the noun's Number tag), not which
plural allomorph was used -- Welsh plural formation is too irregular to
verify "the correct plural" without a full plural lexicon.

Same gates as numeral_engine.py, applied to correct and eroded cases alike:
the next word (skipping hesitation sounds, never across a comma or an
utterance/segment end) must pass spacy_tagging.is_noun_target() (the same
noun gate as numeral_engine.py), must not be code-switched, and must have a
known Number --
unknown means no row. Number comes from spacy_tagging.noun_number(), the
same reader numeral_engine.py uses (lexicon first, then spaCy). "rhai" as a pronoun ("mae rhai yn meddwl") is
followed by a non-noun and so produces nothing.

Standalone: imports only spacy_tagging.py.
"""
from spacy_tagging import is_noun_target, noun_number
from plural_tables import RHAI_FORMS, COLLECTIVE_NOUNS, WELSH_FILLERS


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("‘", "'").replace("’", "'") \
                     .replace("“", '"').replace("”", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def _find_noun_target(i, words_list):
    lookahead = 1
    while lookahead <= 3 and (i + lookahead) < len(words_list):
        candidate = words_list[i + lookahead]
        norm = normalize_word(candidate["word"])
        if candidate.get("synthetic") or norm in WELSH_FILLERS:
            if candidate.get("_clause_boundary_after"):
                return None
            lookahead += 1
            continue
        return candidate
    return None


def _build_plural_row(current_node, target_node, status, is_erosion, number_found,
                      number_source, note):
    return {
        "timestamp":            f"{current_node.get('start')}s - {target_node.get('end')}s",
        "trigger_word":         normalize_word(current_node["word"]),
        "following_word":       normalize_word(target_node["word"]),
        "number_found":         number_found,
        "number_source":        number_source,
        "status":               status,
        "is_erosion":           is_erosion,
        "note":                 note,
        "trigger_confidence":   current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "rule":                 "plural_after_rhai",
    }


def process_plural_marking(words_list):
    rows = []
    for i, current_node in enumerate(words_list):
        if current_node.get("synthetic") or current_node.get("confidence", 0.0) < 0.65:
            continue
        if normalize_word(current_node["word"]) not in RHAI_FORMS:
            continue
        if current_node.get("_clause_boundary_after"):
            continue

        target = _find_noun_target(i, words_list)
        if target is None or target.get("confidence", 0.0) < 0.65:
            continue
        if target.get("_code_switch") or not is_noun_target(target):
            continue

        target_norm = normalize_word(target["word"])
        number, number_source = noun_number(target)
        if number == "plural" or (number == "singular" and target_norm in COLLECTIVE_NOUNS):
            rows.append(_build_plural_row(
                current_node, target, "correct_mutation", False, number, number_source,
                f"Plural noun after 'rhai': {target_norm}"))
        elif number == "singular":
            rows.append(_build_plural_row(
                current_node, target, "erosion", True, number, number_source,
                f"**EROSION**: singular noun after 'rhai' (both Welsh and "
                f"English require a plural here): {target_norm}"))
    return rows
