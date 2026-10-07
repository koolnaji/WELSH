"""
quantifier_tables.py
=====================
Pure data for the quantifier_* branch: plural marking after a quantifier +
"o" ("llawer o lyfrau" = lots of books, "digon o bethau" = enough things).

The second "HAS an English counterpart" data point, alongside the plural_*
branch ("rhai" + plural): English "lots of / enough / too many / more"
takes a plural count noun exactly as Welsh does, so English reinforces the
Welsh requirement -- predicted to RESIST erosion. Added 2026-09-28 because
"rhai" + noun is too rare in speech to carry that prediction alone; this
keeps the design symmetric (two branches predicted to erode on the noun's
number or form, two predicted to resist). It measures the same thing as
numeral_* and plural_* -- the noun's singular/plural tag, from the
lexicon -- so all three rates compare directly.

Quantifiers are listed with their mutated forms ("yn lawer", "mwy" ->
"fwy" after a soft trigger), mapped to one canonical form for reporting.
"""

# surface form -> canonical quantifier
QUANTIFIER_FORMS = {
    "llawer": "llawer", "lawer": "llawer",
    "digon": "digon", "ddigon": "digon",
    "gormod": "gormod", "ormod": "gormod",
    "mwy": "mwy", "fwy": "mwy",
    "llai": "llai", "lai": "llai",
    "faint": "faint",
    "cymaint": "cymaint", "gymaint": "cymaint", "chymaint": "cymaint",
    "nifer": "nifer",
    "rhagor": "rhagor", "ragor": "rhagor",
    "llwyth": "llwyth", "lwyth": "llwyth", "llwythi": "llwyth", "lwythi": "llwyth",
    "cannoedd": "cannoedd", "gannoedd": "cannoedd", "channoedd": "cannoedd",
    "miloedd": "miloedd", "filoedd": "miloedd",
    "degau": "degau", "ddegau": "degau",
    # English loan quantifier (decision 2026-09-28): counted, but flagged in
    # its own column (is_loan_quantifier) so results can be reported with
    # and without it. Whether a borrowed quantifier behaves like native
    # "llawer o" is itself of interest.
    "lot": "lot", "lots": "lot",
    # Partitives, added 2026-10-02 to make the "English agrees" control
    # sturdier: "un o'r pethau" (one of the things), "rhai o'r plant" (some
    # of the children), "dau o'r hogiau", "y rhan fwyaf o'r bobl" -- English
    # takes the plural in exactly the same place. The "o" is required (see
    # LINKING_WORDS), so bare "un ferch" / "tri phlentyn" never get here.
    # "un o" + a singular is often "one FROM" ("un o'r ardal") -- like every
    # singular in this branch it's only a candidate for hand review.
    "un": "un",
    "rhai": "rhai", "rai": "rhai", "rhei": "rhai", "rei": "rhai",
    "dau": "dau", "ddau": "dau", "dwy": "dwy", "ddwy": "dwy",
    "tri": "tri", "dri": "tri", "thri": "tri", "tair": "tair", "dair": "tair",
    "pedwar": "pedwar", "bedwar": "pedwar", "pedair": "pedair", "bedair": "pedair",
    "pump": "pump", "bump": "pump", "chwech": "chwech", "saith": "saith",
    "wyth": "wyth", "naw": "naw", "deg": "deg", "ddeg": "deg",
    "mwyaf": "mwyaf", "fwyaf": "mwyaf", "mwya": "mwyaf", "fwya": "mwyaf",
}
LOAN_QUANTIFIERS = {"lot"}

# The linking preposition -- "llawer o'r plant" normalizes to "o" too.
LINKING_WORDS = {"o"}

# A definite article between "o" and the noun: Siarad transcribes the reduced
# article as its own word ("lot o (y)r plant" -> "o", "yr", "plant"), which
# used to be taken as the noun and the phrase skipped -- 117 Siarad lines
# (2026-10-02). In Whisper/news text "o'r" is split with a synthetic 'r,
# which was already stepped over.
ARTICLE_FORMS = {"y", "yr", "r"}

# Mass nouns, by LEMMA (so mutated and plural forms are excluded alike --
# the exclusion is by noun, not by outcome, as in plural_tables.MASS_NOUNS).
# English "lots of / enough / more" takes the SINGULAR with these too ("a
# lot of time", "enough money"), so a singular here isn't the English
# pattern winning -- it's both languages agreeing on singular. Counted, they
# would all look like erosion. Far more common after quantifiers than after
# "rhai" ("llawer o waith", "digon o amser"), so the list is longer; the
# precision audit should check what it misses. Includes plural_tables'
# MASS_NOUNS lemmas, duplicated per the standalone-branch convention.
MASS_NOUN_LEMMAS = {
    # plural_tables.MASS_NOUNS
    "amser", "arian", "bwyd", "dŵr", "dwr", "gwaith", "help", "cariad", "tywydd",
    # money, noise, fun, stuff, trouble, knowledge
    "pres", "sŵn", "swn", "sbort", "hwyl", "sbri", "stwff", "trafferth",
    "gwybodaeth", "profiad", "sylw", "lle", "croeso", "lwc", "cyffro", "straen",
    # drink, food, substances
    "cwrw", "gwin", "te", "coffi", "llaeth", "llefrith", "bara", "cig", "caws",
    "siwgr", "halen", "menyn", "gwaed", "petrol",
    # weather, nature, materials
    "glaw", "eira", "gwynt", "haul", "golau", "tân", "tan", "mwg", "mwd", "baw",
    "llwch", "sbwriel", "tir", "tywod", "glo", "pren", "papur", "gwair",
    "gwellt", "gwallt", "gwres", "oerfel",
    # abstract
    "amynedd", "hyder", "parch", "cefnogaeth", "cymorth", "gofal", "egni",
    "ynni", "ymchwil", "addysg", "iechyd", "ofn", "poen", "pwysau", "cwsg",
    "diddordeb", "ymarfer", "cymraeg", "saesneg",
    # fixed: "faint o'r gloch" = what time
    "cloch",
}

# Grammatically singular nouns with plural meaning, standard after a
# quantifier ("llawer o bobl" = many people) -- same as plural_tables.
COLLECTIVE_NOUN_LEMMAS = {"pobl", "pobol"}   # pobol: southern spelling (CorCenCC)

# Not a quantity phrase at all, so skipped singular and plural alike (found
# listing the CorCenCC partitive candidates, 2026-10-05): "o gwmpas" is the
# preposition "around" ("rhai o gwmpas" = some around), and a number word
# used as a noun ("un o'r chwech" = one of the six) has no plural to choose.
NON_COUNT_TARGET_LEMMAS = {
    "cwmpas",
    "un", "dau", "dwy", "tri", "tair", "pedwar", "pedair", "pump", "pum",
    "chwech", "chwe", "saith", "wyth", "naw", "deg", "deuddeg", "pymtheg",
    "ugain", "cant", "mil",
}

# A capitalised target is part of a name ("un o Sir Fôn" = one FROM
# Anglesey; CorCenCC's anonymised "un o Lleoliad" = one from [place]) and is
# skipped whatever its number -- except these peoples and nationalities,
# common nouns that are capitalised and take the plural like any other
# ("rhai o'r Cymry"). By lemma.
CAPITALISED_COUNT_LEMMAS = {
    "cymro", "cymraes", "sais", "saesnes", "gwyddel", "albanwr", "americanwr",
    "almaenwr", "ffrancwr", "eidalwr", "sbaenwr", "ewropead", "iddew", "cristion",
}

# ========================= DUPLICATED, STANDALONE-CONVENTION HELPERS =====
# Same lists as plural_tables.py (see the convention note there).
WELSH_FILLERS = {"ym", "er", "ah"}
DISCOURSE_PARTICLES = {"te", "tê", "de", "ta", "ynte", "yntê"}
