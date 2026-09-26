"""
prep_tables.py
===============
Pure data: the conjugated-preposition paradigms the prep_* branch detects
erosion in. No functions, no logic, no side effects -- same spirit as
mutation_tables.py. If you're adding a newly-confirmed paradigm or
correcting a form, it belongs here.

Sourced from https://welearnwelsh.com/blog/welsh-prepositions-part-1/ and
.../welsh-prepositions-part-2/ (colloquial spoken forms), cross-checked
against https://en.wikibooks.org/wiki/Welsh/Grammar/Prepositions (literary
citation forms, generally given with the final -f: "arnaf", "ataf",
"amdanaf"...), all directly reviewed with the researcher rather than
assumed (conversation history, 2026-09-25) -- getting this table wrong
would silently poison every row this branch ever produces, the same way
an uncorrected error in mutation_tables.py would.

PREP_CONJUGATED_FORMS below stores the SOURCED forms as given (favoring
the literary -f form where a source gives one) -- it deliberately does
NOT also enumerate a separate "-f dropped" entry for every person of
every preposition. Final unstressed -f is dropped as a matter of regular,
near-universal COLLOQUIAL WELSH PHONOLOGY ("arnaf" -> spoken "arna"),
completely unrelated to the fusion-loss erosion this branch actually
measures (the fused form disappearing entirely in favour of a bare
preposition + independent pronoun, e.g. "ar fi"). Conflating the two
would badly miscalibrate this branch against exactly the spoken-register
data it exists to analyze -- treating routine, universal casual-speech
phonology as if it were the specific syntactic erosion under study.
Instead, prep_engine.py's matching applies the -f-drop rule generically
to every sourced form at match time (one rule, not one exception list per
preposition) -- see its own docstring for why that's the more principled
design here, and the deliberate exceptions below where a colloquial form
is a genuinely different STEM (not just -f-dropped) and needed its own
explicit entry.

Things worth flagging for anyone extending this table:

  - "gan" has real, extensive North/South dialect variation for 1st/2nd/
    3rd person (gen/gin i, gennyn/ganddon/gynnon/gynddon nhw, etc.) --
    every attested variant across both source pages is included in
    PREP_CONJUGATED_FORMS[gan] as its own set, not just one "standard"
    pick, so a speaker using a different valid regional form isn't
    wrongly counted as erosion.
  - "heb"'s 3rd-person-plural form was independently confirmed as
    "hebddyn nhw" by TWO separate sources -- an earlier screenshot from
    the first source had shown "heb" 3rd-plural with the SAME multi-
    variant text as "gan"'s own 3rd-plural cell (almost certainly a
    copy-paste error on that page), which the Wikibooks source's clean
    "Hebddyn nhw" then confirmed was wrong to keep.
  - "drwy"/"trwy" (through): the Wikibooks source's "Nhw" cell for this
    preposition reads "Drwyddon nhw" -- but EVERY other preposition in
    that same source's tables uses a "-yn" suffix for 3rd plural (arnyn,
    atyn, amdanyn, danyn, drostyn, hebddyn, rhyngddyn), and "drwyddon" is
    identical to that same table's OWN 1st-plural cell ("Ni: Drwyddon
    ni") -- so this reads as the same class of copy-paste artifact as
    the "heb"/"gan" one above, not a genuine irregularity. Encoded here
    as "drwyddyn", matching the paradigm's own internal pattern; flag for
    correction if a better source turns up. "trwy" is treated as the
    same lexeme as "drwy" (a mutation-conditioned spelling variant, not
    an independently-conjugated preposition) -- both appear in
    PREP_BARE_FORMS, but there is only one conjugated-forms entry ("drwy").

"gyda" is deliberately NOT included: it has no live, productive conjugated
paradigm in colloquial Welsh (already used with independent pronouns by
default -- "gyda fi", "gyda ti" -- so there is no inflected form to
measure erosion FROM).

Standalone convention, same as fetch_captions.py/manual_editing.py before
it: a couple of small, stable constants (filler words) are duplicated
here rather than imported from mutation_tables.py/mutation_engine.py, so
this branch has no dependency on the mutation branch's internals -- see
corpus_io.py's CURATED_CHANNELS relocation comment for why that
independence matters for this project's branch-per-prefix convention.
"""

# ========================= PERSON/NUMBER =========================
PERSONS = ["1sg", "2sg", "3sg_m", "3sg_f", "1pl", "2pl", "3pl"]

# The independent pronoun(s) that mark each person -- both the reinforcing
# pronoun that correctly follows a fused conjugated preposition ("arnaf i")
# and the one that follows a BARE preposition in the eroded, analytic
# pattern this branch exists to detect ("ar fi" instead of "arnaf i").
INDEPENDENT_PRONOUNS = {
    # PATCH: "mi" was missing entirely -- confirmed live via phrase-test
    # ("i mi" produced no row at all, traced to _person_for_pronoun
    # returning None for "mi"), even though "mi" is literally the FIRST
    # variant every source gives for 1st-singular "i" ("i mi / i fi").
    # NOTE: "mi" also has a completely separate grammatical role in Welsh
    # -- the pre-verbal affirmative particle ("mi welais i" = "I saw"),
    # no person-marking meaning at all. Not expected to collide with this
    # branch in practice: that role only ever immediately precedes a
    # conjugated VERB, never follows one of PREP_BARE_FORMS/
    # PREP_CONJUGATED_FORMS's words, which is the only context this
    # module ever inspects "mi" in -- flagged as a known dual-role word,
    # same spirit as "yn"'s note above, not exhaustively proven safe
    # against every possible input.
    "1sg":   {"i", "fi", "mi"},
    "2sg":   {"ti", "chdi"},
    "3sg_m": {"fe", "fo", "e", "o"},
    "3sg_f": {"hi"},
    "1pl":   {"ni"},
    "2pl":   {"chi"},
    "3pl":   {"nhw"},
}

# The bare, unconjugated form(s) of each preposition this branch covers --
# what a genuinely-eroded (analytic) usage looks like: this word,
# unconjugated, immediately followed by an independent pronoun instead of
# the fused form below. "trwy" is included alongside "drwy" (see module
# docstring -- same lexeme, mutation-conditioned spelling), both erode the
# same way and both check against PREP_CONJUGATED_FORMS["drwy"].
PREP_BARE_FORMS = {
    "i", "o", "ar", "at", "am", "wrth", "dan", "dros", "drwy", "trwy",
    "heb", "gan", "rhwng", "yn",
}

# A bare surface spelling variant -> the PREP_CONJUGATED_FORMS key it
# should be checked against, for prepositions with more than one bare
# spelling (currently only trwy/drwy). Every other bare form maps to
# itself; this dict only needs the exceptions.
PREP_LEXEME_ALIASES = {"trwy": "drwy"}

# Each preposition's correctly-CONJUGATED WORD form(s) per person, as
# directly sourced -- the single fused token itself, NOT including its
# trailing reinforcing pronoun (that's INDEPENDENT_PRONOUNS above;
# "arnaf i" is two words, "arnaf" is the conjugated form, "i" is the
# reinforcing pronoun). See module docstring for why colloquial -f-drop
# variants are NOT separately enumerated here.
#
# A person absent from a preposition's dict means that preposition does
# NOT fuse for that person at all -- "i" is the one case of this among the
# set covered here: it's mixed-conjugation, only 3rd person genuinely
# fuses (iddo/iddi/iddyn); 1st/2nd/1st-plural/2nd-plural correctly stay
# bare "i" + pronoun. Those persons are deliberately absent from "i"'s
# entry below: "i" + pronoun IS the grammatically correct form there, not
# erosion, so they must never be evaluated as an erosion candidate at all.
PREP_CONJUGATED_FORMS = {
    "i": {
        "3sg_m": {"iddo"}, "3sg_f": {"iddi"}, "3pl": {"iddyn"},
    },
    "ar": {
        "1sg": {"arnaf"}, "2sg": {"arnat"},
        "3sg_m": {"arno"}, "3sg_f": {"arni"},
        "1pl": {"arnon"}, "2pl": {"arnoch"}, "3pl": {"arnyn"},
    },
    # PATCH: Wikibooks gives BOTH -af/-at AND -of/-ot stem variants for
    # "at" explicitly ("ataf i, atof i" / "atat ti, atot ti") -- a real,
    # independently-attested alternation, not the generic -f-drop rule
    # (that rule only strips a trailing -f, it doesn't explain the a/o
    # vowel difference) -- so both stems are stored explicitly here
    # rather than derived.
    "at": {
        "1sg": {"ataf", "atof"}, "2sg": {"atat", "atot"},
        "3sg_m": {"ato"}, "3sg_f": {"ati"},
        "1pl": {"aton"}, "2pl": {"atoch"}, "3pl": {"atyn"},
    },
    "am": {
        "1sg": {"amdanaf"}, "2sg": {"amdanat"},
        "3sg_m": {"amdano"}, "3sg_f": {"amdani"},
        "1pl": {"amdanon"}, "2pl": {"amdanoch"}, "3pl": {"amdanyn"},
    },
    # PATCH: an earlier AI-summarized fetch of a "Part 2" page gave garbled
    # text for this preposition ("Wrth o fi / Wrth a i", with suspicious
    # mid-word spaces) -- flagged as an unreliable extraction at the time
    # and now superseded by a clean screenshot confirming "wrthof"/
    # "wrthot", matching every other preposition's -of/-ot pattern.
    "wrth": {
        "1sg": {"wrthof"}, "2sg": {"wrthot"},
        "3sg_m": {"wrtho"}, "3sg_f": {"wrthi"},
        "1pl": {"wrthon"}, "2pl": {"wrthoch"}, "3pl": {"wrthyn"},
    },
    "dan": {
        "1sg": {"danaf"}, "2sg": {"danat"},
        "3sg_m": {"dano"}, "3sg_f": {"dani"},
        "1pl": {"danon"}, "2pl": {"danoch"}, "3pl": {"danyn"},
    },
    "o": {
        "1sg": {"ohonof", "ohona"}, "2sg": {"ohonot", "ohonat"},
        "3sg_m": {"ohono"}, "3sg_f": {"ohoni"},
        "1pl": {"ohonon"}, "2pl": {"ohonoch"}, "3pl": {"ohonyn"},
    },
    # PATCH: "yn" (in) is heavily overloaded in Welsh grammar -- it's
    # already a mutation TRIGGER in the mutation branch (predicate
    # particle -> soft mutation, preposition "in" -> nasal mutation,
    # aspectual particle -> no mutation at all; see mutation_tables.py's
    # TRIGGERS and layer_1_trigger_detection's "yn" special-case). This is
    # the SAME lexeme in its fourth role: conjugating with a pronoun
    # object the same way ar/at/am/etc. do. Checked for interaction with
    # the mutation branch's own "yn" handling: the mutation branch's
    # unmutable-initial exemption would decline to evaluate a pronoun
    # target under "yn" (a pronoun's initial cluster is never a valid
    # nasal/soft_limited radical), so in the one case where the two
    # branches could both look at the same "yn" + pronoun span (the ERODED
    # bare "yn" case), the mutation branch should produce no row and
    # therefore mark nothing consumed -- not exhaustively proven safe
    # against every possible input, just traced through the current logic.
    "yn": {
        "1sg": {"ynddof"}, "2sg": {"ynddot"},
        "3sg_m": {"ynddo"}, "3sg_f": {"ynddi"},
        "1pl": {"ynddon"}, "2pl": {"ynddoch"}, "3pl": {"ynddyn"},
    },
    # PATCH: same "two independently-attested stems" situation as "at" --
    # welearnwelsh.com gave "drosta i" (a-stem), Wikibooks gave
    # "drostof i" (o-stem, with -f). Both kept explicitly for 1sg;
    # 2sg similarly has both drostat (a-stem) and drostot (o-stem, from
    # Wikibooks) attested.
    "dros": {
        "1sg": {"drosta", "drostof"}, "2sg": {"drostat", "drostot"},
        "3sg_m": {"drosto"}, "3sg_f": {"drosti"},
        "1pl": {"droston"}, "2pl": {"drostoch"}, "3pl": {"drostyn"},
    },
    "drwy": {
        "1sg": {"drwyddof"}, "2sg": {"drwyddot"},
        "3sg_m": {"drwyddo"}, "3sg_f": {"drwyddi"},
        "1pl": {"drwyddon"}, "2pl": {"drwyddoch"},
        "3pl": {"drwyddyn"},  # see module docstring -- source said "drwyddon"
    },
    # PATCH: "gan" conjugates identically for 1st/2nd singular ("gen"/
    # "gin" + pronoun) -- person is disambiguated entirely by the
    # following pronoun ("gen i" vs "gen ti"), unlike ar/at/am/wrth/dan/
    # dros/drwy, which each have a distinct 1st vs 2nd singular stem.
    # Confirmed against the source table, not a transcription slip.
    "gan": {
        "1sg": {"gen", "gin"}, "2sg": {"gen", "gin"},
        "3sg_m": {"ganddo", "gynno"}, "3sg_f": {"ganddi", "gynni"},
        "1pl": {"gynnon", "gennyn", "ganddon"},
        "2pl": {"gynnoch", "gennych", "ganddoch"},
        "3pl": {"ganddyn", "gennyn", "gynnon", "gynddon"},
    },
    "heb": {
        # PATCH: "hebdda" (welearnwelsh.com) is a genuinely different stem
        # from "hebddof" (Wikibooks) -- not derivable from it by -f-drop
        # (which would give "hebddo", already the 3sg_m form) -- kept as
        # its own explicit variant rather than assumed derivable.
        "1sg": {"hebddof", "hebdda"}, "2sg": {"hebddot", "hebddat"},
        "3sg_m": {"hebddo"}, "3sg_f": {"hebddi"},
        "1pl": {"hebddon"}, "2pl": {"hebddoch"},
        "3pl": {"hebddyn"},  # confirmed independently by BOTH sources
    },
    "rhwng": {
        "1sg": {"rhyngddof"}, "2sg": {"rhyngddot"},
        "3sg_m": {"rhyngddo"}, "3sg_f": {"rhyngddi"},
        "1pl": {"rhyngddon"}, "2pl": {"rhyngddoch"}, "3pl": {"rhyngddyn"},
    },
}

# ========================= DUPLICATED, STANDALONE-CONVENTION HELPERS =====
# Small and stable enough to duplicate rather than import -- keeps this
# branch independent of mutation_engine.py/mutation_tables.py's internals.
# Hesitation sounds only. "iawn"/"gwybod"/"te" are real words, and skipping
# them let "(o'n) iawn i ddweud" pair "yn" with "i"; "chdi" is the northern
# 2sg pronoun (now in INDEPENDENT_PRONOUNS). mutation_tables.WELSH_FILLERS
# keeps the longer list because corpus_formality uses it as a filler measure.
WELSH_FILLERS = {"ym", "er", "ah"}
