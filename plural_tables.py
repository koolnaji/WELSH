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

# "rai" is the soft-mutated form ("i rai pobl"); "rhei"/"rei" are the northern
# spellings the Siarad transcribers use -- 183 Siarad lines have "rei/rhei" +
# a word against 84 with "rhai/rai", so without them the branch read about a
# third of the data (2026-10-02).
RHAI_FORMS = {"rhai", "rai", "rhei", "rei"}

# Grammatically singular nouns with plural meaning that are standard after
# "rhai" ("rhai pobl" = some people) -- their singular tag is not erosion.
COLLECTIVE_NOUNS = {"pobl", "bobl", "pobol", "bobol"}   # pobol: southern spelling (CorCenCC)

# Mass nouns are excluded from this branch (decision 2026-09-26). The
# "English agrees" premise only holds for countable nouns: English "some"
# takes the SINGULAR with mass nouns ("some time", "some money"), and
# standard Welsh uses "peth", not "rhai", there ("peth amser"). So "rhai
# amser" (davies1.cha: "fi yn iawn fel rhai amser") copies English rather
# than failing a rule both languages share -- counted here it would make the
# control look like it erodes for an English-driven reason. Singular and
# plural forms are both listed, so the exclusion is by noun, not by outcome.
MASS_NOUNS = {
    "amser", "amserau", "arian", "bwyd", "bwydydd", "dŵr", "dwr", "dyfroedd",
    "gwaith", "gweithiau", "help", "cariad", "cariadon", "tywydd",
    # 2026-10-01: Patagonia's 5 rhai "erosions" were "rhai cig", "rhai caws",
    # "rhai Sbaeneg" (some meat / cheese / Spanish) and two non-contexts --
    # so the list now matches quantifier_tables.MASS_NOUN_LEMMAS (duplicated
    # per the convention below; "rhai" doesn't mutate its noun, so radical
    # forms are enough), plus language names and "newid" (some change).
    "pres", "sŵn", "swn", "sbort", "hwyl", "sbri", "stwff", "trafferth",
    "gwybodaeth", "profiad", "sylw", "croeso", "lwc", "cyffro", "straen",
    "cwrw", "gwin", "coffi", "llaeth", "llefrith", "bara", "cig", "caws",
    "siwgr", "halen", "menyn", "gwaed", "petrol",
    "glaw", "eira", "gwynt", "haul", "golau", "tân", "mwg", "mwd", "baw",
    "llwch", "sbwriel", "tir", "tywod", "glo", "pren", "papur", "gwair",
    "gwellt", "gwallt", "gwres", "oerfel",
    "amynedd", "hyder", "parch", "cefnogaeth", "cymorth", "gofal", "egni",
    "ynni", "ymchwil", "addysg", "iechyd", "ofn", "poen", "pwysau", "cwsg",
    "diddordeb", "ymarfer", "newid", "metel",
    "cymraeg", "saesneg", "sbaeneg", "ffrangeg", "almaeneg", "eidaleg",
}

# Pronouns that echo a possessive after its noun ("tad fi" = my dad). Same
# list as mutation_tables.ECHO_PRONOUNS, duplicated per the convention below.
POSSESSIVE_ECHO_PRONOUNS = {"fi", "ti", "di", "fe", "e", "hi", "ni", "chi", "nhw"}

# Not nouns after pronoun "rhai" (2026-09-28): the tag particle "te" (isn't
# it?) in "mae yna rai te" (there are some, aren't there -- fusser19.cha), and
# "fan"/"man" + yma/yna/hyn/acw = "here/there" ("rai fan yna" = some over
# there, fusser23.cha). Same list as mutation_tables.DISCOURSE_PARTICLES.
DISCOURSE_PARTICLES = {"te", "tê", "de", "ta", "ynte", "yntê"}
# More pronoun "rhai" + a word that isn't its noun, from the first run with the
# northern spellings (2026-10-02): "yr hen rei gyd" (the old ones, all -- "i
# gyd"), "rei pawb arall" (everyone else's), "rei tu mewn" (ones inside),
# "rhai tro nesa" (some next time). Dropped whatever their number.
NOT_RHAI_NOUNS = {"gyd", "pawb", "tu", "tro", "dro",
                  # CorCenCC (2026-10-02): "rai popeth", "rhai lan" (put some up),
                  # "rhai pedwar deg gwaith" (some forty times), "rai fan yn heddiw"
                  # (some here today -- "fan hyn"), "fan 'na"
                  "popeth", "lan", "fan", "man"}
# "y rhai" = "the ones", always the pronoun: "y rei côr" (the choir ones), "y
# rei cyngerdd" (the concert ones), "yr hen rei" -- the word after it is an
# attributive noun, singular in English too ("the choir ones"). Same 2026-10-02
# run: 4 of the 8 new Siarad/Patagonia "erosions" were this.
RHAI_PRONOUN_ARTICLES = {"y", "yr", "'r", "r"}
PLACE_ADVERB_NOUNS = {"fan", "man"}
PLACE_ADVERB_FOLLOWERS = {"yma", "yna", "hyn", "acw", "ma", "na", "'ma", "'na"}

# ========================= DUPLICATED, STANDALONE-CONVENTION HELPERS =====
# Small and stable enough to duplicate rather than import -- keeps this
# branch independent of mutation_engine.py/mutation_tables.py's internals
# (see prep_tables.py's own comment on this same convention).
# Hesitation sounds only -- "iawn"/"gwybod"/"te"/"chdi" are real words and
# skipping them attaches the trigger to the wrong word (same fix as prep_tables).
WELSH_FILLERS = {"ym", "er", "ah"}
