"""
plural_tables.py
=================
Pure data: the plural-demonstrative triggers the plural_* branch detects
retention/erosion of plural marking after. No functions, no logic, no
side effects -- same spirit as mutation_tables.py.

This is the "HAS an English counterpart" data point in this project's
hypothesis (contrast mutation_tables.py's TRIGGERS/RADICAL_TO_MUTATED,
and prep_tables.py's PREP_CONJUGATED_FORMS -- both "has NO counterpart"
phenomena). Both Welsh and English obligatorily mark plurality on the
noun -- different mechanism (Welsh's varied suffixation/apophony vs.
English's regular -s), same grammatical requirement. An English-dominant
bilingual's other language REINFORCES "you must mark this," rather than
undermining it the way it does for mutation or conjugated prepositions
(neither of which has any English counterpart at all) -- predicted by
this project's hypothesis to RESIST erosion, in direct contrast to the
other two branches.

PLURAL_DEMONSTRATIVE_TRIGGERS is a closed, deliberately small set: "y
rhain" (these) and "y rheina"/"y rheiny" (those -- both spellings
attested) -- TWO-WORD triggers, architecturally different from the
single-word TRIGGERS dict in mutation_tables.py or the single-word
PREP_BARE_FORMS in prep_tables.py, which is why plural_engine.py can't
reuse mutation_engine.layer_1_trigger_detection's shape and instead does
its own small bigram scan (see that module's own docstring).
"""

# (first_word, second_word) -> plain-English gloss, for the note field on
# any row this trigger produces.
PLURAL_DEMONSTRATIVE_TRIGGERS = {
    ("y", "rhain"):  "these",
    ("y", "rheina"): "those",
    ("y", "rheiny"): "those",
}

# ========================= DUPLICATED, STANDALONE-CONVENTION HELPERS =====
# Small and stable enough to duplicate rather than import -- keeps this
# branch independent of mutation_engine.py/mutation_tables.py's internals
# (see prep_tables.py's own comment on this same convention).
WELSH_FILLERS = {"ym", "er", "ah", "iawn", "gwybod", "chdi", "te", "ffeil"}
