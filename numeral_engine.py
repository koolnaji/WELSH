"""
numeral_engine.py
==================
Detection for the numeral_* branch (see numeral_tables.py for the
linguistics): a numeral followed directly by a noun should take the
SINGULAR in Welsh; a plural there is the English-style form.

Gates, all applied identically to correct and eroded cases so none of them
can tilt the rate:
  - the trigger must be tagged as a numeral (spaCy NUM or Cysill CARD) --
    "deg" is also soft-mutated "teg" (fair), "mil" also "animal";
  - the next word (skipping hesitation sounds, never across a comma) must
    be tagged NOUN, must not be the partitive "o", and must not be a
    code-switched word (an English noun keeps English plural morphology);
  - its singular/plural value must be known -- unknown means no row. Read
    by spacy_tagging.noun_number(): the Bangor lexicon's dictionary value
    first, spaCy's morph tag where the lexicon has none; number_source in
    each row says which one decided.

Deliberately does NOT check or set mark_consumed(): the same noun is
routinely also a mutation-branch target ("tri chi"), and noun NUMBER is a
separate phenomenon from its MUTATION -- skipping it because another branch
used it would silently drop this branch's data.

Standalone: imports only spacy_tagging.py, same convention as the other
non-mutation branches.
"""
from spacy_tagging import noun_number
from numeral_tables import NUMERAL_FORMS, PARTITIVE_WORDS, WELSH_FILLERS


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("‘", "'").replace("’", "'") \
                     .replace("“", '"').replace("”", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def _tagged_as(node, spacy_pos, cysill_prefixes):
    spacy_tok = node.get("spacy_token") or {}
    if spacy_tok.get("pos") == spacy_pos:
        return True
    return (node.get("cysill_pos") or "").upper().startswith(cysill_prefixes)


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


def _build_numeral_row(current_node, target_node, numeral, status, is_erosion,
                       number_found, number_source, note):
    return {
        "timestamp":            f"{current_node.get('start')}s - {target_node.get('end')}s",
        "numeral":              numeral,
        "numeral_surface":      normalize_word(current_node["word"]),
        "following_word":       normalize_word(target_node["word"]),
        "number_found":         number_found,
        "number_source":        number_source,
        "status":               status,
        "is_erosion":           is_erosion,
        "note":                 note,
        "trigger_confidence":   current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "rule":                 "numeral_singular_noun",
    }


def process_numeral_agreement(words_list):
    rows = []
    for i, current_node in enumerate(words_list):
        if current_node.get("synthetic") or current_node.get("confidence", 0.0) < 0.65:
            continue
        numeral = NUMERAL_FORMS.get(normalize_word(current_node["word"]))
        if numeral is None or current_node.get("_clause_boundary_after"):
            continue
        if not _tagged_as(current_node, "NUM", ("CARD",)):
            continue

        target = _find_noun_target(i, words_list)
        if target is None or target.get("confidence", 0.0) < 0.65:
            continue
        if normalize_word(target["word"]) in PARTITIVE_WORDS:
            continue
        if target.get("_code_switch"):
            continue
        if not _tagged_as(target, "NOUN", ("N",)):
            continue

        number, number_source = noun_number(target)
        if number == "singular":
            rows.append(_build_numeral_row(
                current_node, target, numeral, "correct_mutation", False, "singular",
                number_source, f"Singular noun after '{numeral}' (Welsh pattern)"))
        elif number == "plural":
            rows.append(_build_numeral_row(
                current_node, target, numeral, "erosion", True, "plural",
                number_source,
                f"**EROSION**: plural noun after '{numeral}' (English pattern; "
                f"Welsh takes the singular here)"))
    return rows
