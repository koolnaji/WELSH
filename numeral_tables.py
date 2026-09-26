"""
numeral_tables.py
==================
Pure data for the numeral_* branch: numeral + noun number agreement.

Welsh puts a SINGULAR noun directly after a numeral ("tri chi", "dau
blentyn", "pum mlynedd"); English uses a plural ("three dogs"). This is the
branch's "NO English counterpart" structure -- the syntactic-assimilation
prediction is drift toward English-style numeral + PLURAL ("tri cŵn").
Its "HAS a counterpart" partner is plural_* ("rhai" + plural noun, where
Welsh and English agree): both branches measure the same thing (the noun's
singular/plural tag), so tagger errors hit both sides equally.

Only the numeral IMMEDIATELY followed by its noun counts. The partitive
"tri o'r plant" ("three of the children") correctly takes a plural and is
native Welsh, so a following "o"/"o'r" means no row. "un" is excluded:
English "one dog" is singular too, so there is no contrast to measure.
"""

# surface form (including soft/nasal-mutated spellings) -> citation form
NUMERAL_FORMS = {
    "dau": "dau", "ddau": "dau",
    "dwy": "dwy", "ddwy": "dwy",
    "tri": "tri", "dri": "tri",
    "tair": "tair", "dair": "tair",
    "pedwar": "pedwar", "bedwar": "pedwar",
    "pedair": "pedair", "bedair": "pedair",
    "pump": "pump", "pum": "pump", "bump": "pump", "bum": "pump",
    "chwech": "chwech", "chwe": "chwech",
    "saith": "saith",
    "wyth": "wyth",
    "naw": "naw",
    "deg": "deg", "ddeg": "deg", "deng": "deg", "ddeng": "deg",
    "deuddeg": "deuddeg", "ddeuddeg": "deuddeg",
    "pymtheg": "pymtheg", "bymtheg": "pymtheg",
    "deunaw": "deunaw", "ddeunaw": "deunaw",
    "ugain": "ugain",
    "cant": "cant", "gant": "cant",
    "mil": "mil", "fil": "mil",
}

# A following "o"/"o'r" is the native partitive ("tri o'r plant"), plural
# by design -- not a numeral + noun context.
PARTITIVE_WORDS = {"o", "o'r", "or"}

# Hesitation sounds only -- same convention as prep_tables/plural_tables.
WELSH_FILLERS = {"ym", "er", "ah"}
