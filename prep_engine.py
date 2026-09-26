"""
prep_engine.py
================
Detection engine for the conjugated-preposition branch: Welsh prepositions
that fuse with a following pronoun into a single inflected word (e.g. "ar"
+ 1st singular -> "arna i") are a grammatical category with NO counterpart
in English -- English prepositions never conjugate ("to him"/"to me" is
always invariant "to" + pronoun, full stop). Per this project's
hypothesis (dominant-language contact pressure erodes structures the
dominant language has nothing corresponding to, faster than structures it
does), this branch is the second data point alongside mutation erosion:
the predicted colloquial drift is the SAME shape English already has --
bare preposition + independent pronoun ("ar fi") replacing the fused form
("arna i") -- which is exactly the failure mode to detect.

Same trigger -> expected -> evaluate shape as mutation_engine.py's Layer
1A, but simpler: there's no reverse-mutation-table reconstruction needed,
just a direct lookup of whether the word matches this person's correctly-
conjugated form (prep_tables.PREP_CONJUGATED_FORMS) or the bare radical
(prep_tables.PREP_BARE_FORMS) -- both keyed off which independent pronoun
follows.

Deliberately standalone (see prep_tables.py's own docstring): this module
does not import mutation_engine.py or mutation_tables.py, so the
prep_* branch has no dependency on the mutation branch's internals.
Shares only spacy_tagging.py (the tagger-output module every branch
consumes) and its mark_consumed/was_consumed consumption-tracking helper,
specifically so this branch can't independently reintroduce the double-
counting bug found and fixed in the mutation branch (see
mutation_engine.py's PATCH (2.2) comment for the full story).
"""
from spacy_tagging import mark_consumed, was_consumed
from prep_tables import (
    PERSONS, INDEPENDENT_PRONOUNS, PREP_BARE_FORMS, PREP_CONJUGATED_FORMS,
    PREP_LEXEME_ALIASES, WELSH_FILLERS,
)


def _matches_any_form(word, valid_forms):
    """
    True if `word` equals one of `valid_forms` exactly, OR equals one of
    them with its trailing "-f" dropped -- the near-universal colloquial
    Welsh phonological reduction (unstressed final -f is regularly not
    pronounced in casual/spoken registers) that prep_tables.py
    deliberately does NOT enumerate a separate entry for (see that
    module's own docstring). This is a different phenomenon from the
    fusion-loss erosion this branch measures -- treating a dropped final
    -f as erosion would badly miscalibrate this branch against exactly
    the spoken-register data it's meant to analyze.
    """
    if word in valid_forms:
        return True
    return any(f.endswith("f") and word == f[:-1] for f in valid_forms)


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("‘", "'").replace("’", "'") \
                     .replace("“", '"').replace("”", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def _person_for_pronoun(pronoun_norm):
    for person, forms in INDEPENDENT_PRONOUNS.items():
        if pronoun_norm in forms:
            return person
    return None


def _find_pronoun_target(i, words_list):
    """Lookahead for the next non-filler, non-synthetic word -- mirrors
    mutation_engine._find_lookahead_target's shape (same reasoning: absorb
    a filler word or two between the preposition and its pronoun without
    losing the match, don't look arbitrarily far ahead). Never skips across
    a comma/clause boundary."""
    lookahead = 1
    while lookahead <= 3 and (i + lookahead) < len(words_list):
        candidate = words_list[i + lookahead]
        norm = normalize_word(candidate["word"])
        if candidate.get("synthetic") or norm in WELSH_FILLERS:
            if candidate.get("_clause_boundary_after"):
                return None, lookahead
            lookahead += 1
            continue
        return candidate, lookahead
    return None, lookahead


def _tagged_as(node, spacy_pos, cysill_prefixes):
    spacy_tok = node.get("spacy_token") or {}
    if spacy_tok.get("pos") == spacy_pos:
        return True
    return (node.get("cysill_pos") or "").upper().startswith(cysill_prefixes)


def _build_prep_row(current_node, target_node, prep, person, status,
                     is_erosion, note, mutation_found, expected_forms):
    return {
        "timestamp":          f"{current_node.get('start')}s - {target_node.get('end')}s",
        "preposition":        prep,
        "person":             person,
        "surface_form":       normalize_word(current_node["word"]),
        "following_pronoun":  normalize_word(target_node["word"]),
        "expected_forms":     "|".join(sorted(expected_forms)),
        "mutation_found":     mutation_found,
        "status":             status,
        "is_erosion":         is_erosion,
        "note":               note,
        "trigger_confidence": current_node.get("confidence", 0.0),
        "following_confidence": target_node.get("confidence"),
        "rule":               "conjugated_preposition",
    }


def process_preposition_erosion(words_list):
    """
    Walks the same flat, preprocessed word-stream every branch consumes
    (see mutation_engine.process_comprehensive_mutations for the sibling
    walk over the mutation branch's own rules) and returns one row per
    conjugated-preposition context found -- correct (fused form used) or
    eroded (bare preposition + independent pronoun used instead, where
    this person genuinely has a fused form to have used).

    Deliberately does NOT touch/consume words this branch doesn't itself
    classify: a bare preposition followed by a pronoun for a person that
    prep_tables.PREP_CONJUGATED_FORMS says doesn't conjugate at all (e.g.
    "i" + 1st/2nd person) produces no row -- that's the grammatically
    correct form, not an erosion candidate, nothing to measure.
    """
    rows = []
    for i, current_node in enumerate(words_list):
        if current_node.get("synthetic") or was_consumed(current_node):
            continue
        conf = current_node.get("confidence", 0.0)
        if conf < 0.65:
            continue
        norm = normalize_word(current_node["word"])
        if len(norm) < 1:
            continue
        # PATCH: false-positive guards, all confirmed live 2026-09-26 -- every
        # erosion this branch had produced came from one of these:
        #   - a "yn" split out of a contraction ("o'n i" = oeddwn i, "I was");
        #   - a comma right after the trigger ("O, ti'n..." -- interjection);
        #   - the stem of a split contraction read as a pronoun ("i o'n").
        if current_node.get("_from_contraction") or current_node.get("_clause_boundary_after"):
            continue

        target, lookahead = _find_pronoun_target(i, words_list)
        if target is None or target.get("_contraction_stem"):
            continue
        if target.get("confidence", 0.0) < 0.65:
            continue
        target_norm = normalize_word(target["word"])
        person = _person_for_pronoun(target_norm)
        if person is None:
            continue  # next word isn't a recognized independent pronoun -- not this phenomenon
        # "i", "o", "ni", "fi" are also pronouns/particles/interjections, so the
        # word after the preposition must actually be tagged as a pronoun --
        # applied to correct AND eroded cases alike, so it can't tilt the rate.
        if not _tagged_as(target, "PRON", ("PRON",)):
            continue

        # Case A: current word IS a correctly-conjugated form for this
        # exact person, under some preposition (exact match, or the same
        # form with a colloquially-dropped final -f -- see
        # _matches_any_form's docstring).
        matched_prep = None
        for prep, forms_by_person in PREP_CONJUGATED_FORMS.items():
            valid_forms = forms_by_person.get(person)
            if valid_forms and _matches_any_form(norm, valid_forms):
                matched_prep = prep
                break
        if matched_prep:
            valid_forms = PREP_CONJUGATED_FORMS[matched_prep][person]
            row = _build_prep_row(
                current_node, target, matched_prep, person,
                status="correct_mutation", is_erosion=False,
                note=f"Correctly conjugated '{matched_prep}' ({person}): {norm}",
                mutation_found=norm, expected_forms=valid_forms)
            rows.append(row)
            mark_consumed(target)
            continue

        # Case B: current word is the BARE preposition, and this person
        # DOES have a genuine conjugated form to erode from (excludes "i"
        # 1st/2nd person, which correctly stays bare -- see
        # PREP_CONJUGATED_FORMS's own docstring). Resolved through
        # PREP_LEXEME_ALIASES first (currently only trwy -> drwy) so a
        # spelling variant checks against its lexeme's real paradigm.
        # The bare forms ("i", "o", "yn", "am"...) are homographs of pronouns,
        # interjections and particles, so they must be tagged as a
        # preposition. Conjugated forms (Case A: "iddo", "arna", "ohono") are
        # unambiguous words and aren't gated this way.
        if norm in PREP_BARE_FORMS and _tagged_as(current_node, "ADP", ("PREP", "CPREP")):
            lexeme = PREP_LEXEME_ALIASES.get(norm, norm)
            valid_forms = PREP_CONJUGATED_FORMS.get(lexeme, {}).get(person)
            if valid_forms:
                row = _build_prep_row(
                    current_node, target, norm, person,
                    status="erosion", is_erosion=True,
                    note=(f"**EROSION**: expected conjugated '{norm}' ({person}: "
                          f"{'/'.join(sorted(valid_forms))}), bare preposition + "
                          f"independent pronoun used instead"),
                    mutation_found="none", expected_forms=valid_forms)
                rows.append(row)
                mark_consumed(target)
                continue
            # else: this preposition doesn't conjugate for this person at
            # all (e.g. "i" + 1sg/2sg/1pl/2pl) -- bare + pronoun is
            # correct here, not erosion, no row.

    return rows
