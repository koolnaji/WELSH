"""
plural_engine.py
==================
Detection engine for the plural-marking branch: after a plural
demonstrative ("y rhain"/"y rheina"/"y rheiny" -- see plural_tables.py),
the following noun is grammatically required to carry plural marking.
Unlike mutation or conjugated prepositions, English ALSO obligatorily
marks plurality on nouns -- different mechanism, same requirement -- so
an English-dominant bilingual's other language reinforces this category
rather than undermining it. Per this project's hypothesis, this is the
predicted-to-RESIST-erosion comparison point against the other two
branches (see plural_tables.py's module docstring for the full argument).

First pass deliberately detects PRESENCE of plural marking only (is the
noun tagged Number=Plur at all), not which specific allomorph was used --
Welsh plural formation is far more irregular than mutation (dozens of
suffix/vowel-mutation patterns vs. mutation's three clean tables), so
verifying "is this the CORRECT plural form" would need a full plural-
paradigm lexicon this project doesn't have. "Is it marked as plural at
all" is directly answerable off spaCy's own Number morph tag via
spacy_tagging.extract_number_from_spacy -- the same function the
mutation branch already uses for its own plural gating (require_target_
not_plural), relocated there specifically so this branch could use it
without importing mutation_engine.py's internals.

Same trigger -> expected -> evaluate shape as the other two branches, but
the trigger itself is a BIGRAM ("y" + "rhain"/"rheina"/"rheiny"), not a
single word -- mutation_engine.layer_1_trigger_detection's shape assumes
a single trigger word, so this does its own small scan rather than reuse
it. Deliberately standalone (see prep_engine.py's docstring for the same
reasoning): imports only spacy_tagging.py, not mutation_engine.py or
mutation_tables.py.
"""
from spacy_tagging import mark_consumed, was_consumed, extract_number_from_spacy
from plural_tables import PLURAL_DEMONSTRATIVE_TRIGGERS, WELSH_FILLERS


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("‘", "'").replace("’", "'") \
                     .replace("“", '"').replace("”", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def _find_noun_target(i, words_list):
    """Lookahead for the next non-filler, non-synthetic word after the
    two-word trigger -- mirrors the other branches' lookahead shape (same
    reasoning: absorb a filler word or two without losing the match)."""
    lookahead = 1
    while lookahead <= 3 and (i + lookahead) < len(words_list):
        candidate = words_list[i + lookahead]
        norm = normalize_word(candidate["word"])
        if candidate.get("synthetic") or norm in WELSH_FILLERS:
            lookahead += 1
            continue
        return candidate, lookahead
    return None, lookahead


def _build_plural_row(current_node, trigger_word2, target_node, trigger_gloss,
                       status, is_erosion, note):
    return {
        "timestamp":            f"{current_node.get('start')}s - {target_node.get('end')}s",
        "trigger_word":         f"y {trigger_word2}",
        "trigger_gloss":        trigger_gloss,
        "following_word":       normalize_word(target_node["word"]),
        "status":               status,
        "is_erosion":           is_erosion,
        "note":                 note,
        "trigger_confidence":   current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "rule":                 "plural_after_demonstrative",
    }


def process_plural_marking(words_list):
    """
    Walks the same flat, preprocessed word-stream every branch consumes.
    Returns one row per plural-demonstrative context found: the following
    noun either carries plural marking (correct_mutation, not erosion) or
    doesn't (erosion -- a singular/unmarked form used where plural
    marking is grammatically obligatory in both Welsh and English).

    Only fires when the target's Number morph feature is confidently
    known (Plur or Sing) -- an unknown/missing Number feature produces no
    row at all, same "don't guess when uncertain" discipline as the
    mutation branch's own require_target_not_plural gate.
    """
    rows = []
    i = 0
    n = len(words_list)
    while i < n:
        current_node = words_list[i]
        if current_node.get("synthetic") or was_consumed(current_node):
            i += 1
            continue
        if current_node.get("confidence", 0.0) < 0.65:
            i += 1
            continue
        norm1 = normalize_word(current_node["word"])
        if norm1 != "y" or (i + 1) >= n:
            i += 1
            continue

        next_node = words_list[i + 1]
        if next_node.get("synthetic") or next_node.get("confidence", 0.0) < 0.65:
            i += 1
            continue
        norm2 = normalize_word(next_node["word"])
        gloss = PLURAL_DEMONSTRATIVE_TRIGGERS.get((norm1, norm2))
        if gloss is None:
            i += 1
            continue

        # Trigger found -- mark the second trigger word ("rhain" etc.)
        # consumed too, same reasoning as marking a mutation TARGET
        # consumed: it's already been used as half of this trigger and
        # shouldn't be independently re-evaluated as something else's
        # target by a different branch walking the same word stream.
        mark_consumed(next_node)

        target, lookahead = _find_noun_target(i + 1, words_list)
        if target is None or target.get("confidence", 0.0) < 0.65:
            i += 1
            continue

        spacy_tok = target.get("spacy_token")
        number = extract_number_from_spacy(spacy_tok) or target.get("cysill_number")
        if number is None:
            i += 1
            continue  # genuinely uncertain -- no row, not a guess either way

        target_norm = normalize_word(target["word"])
        if number == "plural":
            row = _build_plural_row(
                current_node, norm2, target, gloss,
                status="correct_mutation", is_erosion=False,
                note=f"Plural correctly marked after 'y {norm2}': {target_norm}")
        else:
            row = _build_plural_row(
                current_node, norm2, target, gloss,
                status="erosion", is_erosion=True,
                note=(f"**EROSION**: plural marking expected after 'y {norm2}' "
                      f"(has an English counterpart -- both languages require "
                      f"plural marking here), singular/unmarked form used: "
                      f"{target_norm}"))
        rows.append(row)
        mark_consumed(target)
        i += 1

    return rows
