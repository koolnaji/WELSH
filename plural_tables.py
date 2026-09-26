"""
plural_tables.py
=================
Pure data for the plural_* branch: plural marking after "rhai" ("some").

This is the "HAS an English counterpart" data point: Welsh "rhai" and
English "some" both take a PLURAL noun ("rhai llyfrau" / "some books"), so
English reinforces the Welsh requirement rather than undermining it --
predicted to RESIST erosion. It is the partner of the numeral_* branch
(numeral + SINGULAR noun, no English counterpart): both measure the same
thing, the noun's singular/plural tag, so their rates are directly
comparable.

PATCH (2026-09-26): the original trigger, "y rhain"/"y rheina"/"y rheiny" +
noun, was a design error -- "y rhain" is a pronoun ("these ones") and is
almost never followed by a noun ("these dogs" is "y cŵn 'ma/hyn"), so the
branch produced zero rows on every real transcript.
"""

# "rai" is the soft-mutated form ("i rai pobl").
RHAI_FORMS = {"rhai", "rai"}

# Grammatically singular nouns with plural meaning that are standard after
# "rhai" ("rhai pobl" = some people) -- their singular tag is not erosion.
COLLECTIVE_NOUNS = {"pobl", "bobl"}

# ========================= DUPLICATED, STANDALONE-CONVENTION HELPERS =====
# Small and stable enough to duplicate rather than import -- keeps this
# branch independent of mutation_engine.py/mutation_tables.py's internals
# (see prep_tables.py's own comment on this same convention).
# Hesitation sounds only -- "iawn"/"gwybod"/"te"/"chdi" are real words and
# skipping them attaches the trigger to the wrong word (same fix as prep_tables).
WELSH_FILLERS = {"ym", "er", "ah"}
