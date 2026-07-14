"""
mutation_tables.py
===================
Pure data: every Welsh mutation table, trigger list, and lexicon the
pipeline uses. No functions, no logic, no side effects -- just the facts
about the language. If you're adding a newly-discovered linguistic rule
(a new trigger, a new exemption list, a new lexicon), it almost certainly
belongs here, not in mutation_engine.py.

Split out of mutation_engine.py so that "what does the pipeline believe
about Welsh mutation" is one file you can read top to bottom without also
reading API-retry logic, hallucination filtering, or CLI plumbing.
"""

# ========================= MUTATION TABLES =========================
SOFT_MUTATION = {
    "p": "b", "t": "d", "c": "g", "b": "f", "d": "dd", "g": "",
    "ll": "l", "rh": "r", "m": "f"
}
SOFT_MUTATION_LIMITED = {k: v for k, v in SOFT_MUTATION.items()
                         if k not in ("ll", "rh")}
NASAL_MUTATION    = {"p": "mh", "t": "nh", "c": "ngh", "b": "m", "d": "n", "g": "ng"}
ASPIRATE_MUTATION = {"p": "ph", "t": "th", "c": "ch"}
COLLOQUIAL_AFFRICATE_MUTATION = {"ts": "j"}

RADICAL_TO_MUTATED = {
    "soft": SOFT_MUTATION, "soft_limited": SOFT_MUTATION_LIMITED,
    "nasal": NASAL_MUTATION, "aspirate": ASPIRATE_MUTATION,
    "colloquial_affricate": COLLOQUIAL_AFFRICATE_MUTATION,
}

MUTATION_MAP = {}
for mtype, mapping in RADICAL_TO_MUTATED.items():
    for radical, mutated in mapping.items():
        if mutated and mutated not in MUTATION_MAP:
            MUTATION_MAP[mutated] = mtype
MUTATION_MAP[""] = "soft"

MUTATION_TAG_MAP  = {"TM": "soft", "TT": "nasal", "TL": "aspirate", "TH": "h-mutation"}
SPACY_MUTATION_MAP= {"SM": "soft", "NM": "nasal", "AM": "aspirate"}

SELECTIVE_INVARIANT_MAP = {
    "nasal":        {"m", "n", "ll", "rh", "f", "s", "ch", "h", "j"},
    "aspirate":     {"m", "n", "ll", "rh", "b", "d", "g", "f", "s", "h", "j"},
    "soft":         {"n", "f", "s", "ch", "h", "j"},
    "soft_limited": {"n", "f", "s", "ch", "h", "j"},
}
ASPIRATE_INITIALS = set(ASPIRATE_MUTATION.keys())

# ========================= TRIGGER TABLE =========================
TRIGGERS = {
    "am": "soft", "ar": "soft", "at": "soft", "dan": "soft",
    "dros": "soft", "drwy": "soft", "heb": "soft", "wrth": "soft",
    "gan": "soft", "i": "soft", "o": "soft", "hyd": "soft",
    "tua": "aspirate", "thua": "aspirate",
    "neu": "soft", "pan": "soft", "ail": "soft",
    "ni": "soft|aspirate", "nid": "soft|aspirate",
    "na": "soft|aspirate", "oni": "soft|aspirate",
    "beth": "soft", "pa": "soft", "pwy": "soft", "sut": "soft",
    "rhy": "soft", "lled": "soft", "pur": "soft_limited",
    "reit": "soft", "hollol": "soft", "gweddol": "soft",
    "go": "soft", "llwyr": "soft",
    # PATCH: pre-posed adjectives -- when these precede their noun (instead
    # of the usual noun+adjective order), the noun takes soft mutation
    # regardless of gender/number, e.g. "hen ddyn", "rhyw brynhawn",
    # "ychydig fisoedd", "amryw bethau", "prif fachgen".
    "hen": "soft", "rhyw": "soft", "ychydig": "soft",
    "amryw": "soft", "prif": "soft",
    "mor": "soft_limited", "cyn": "soft_limited",
    "mae": "soft", "ydy": "soft", "oes": "soft",
    "sy": "soft", "sydd": "soft",
    "dyma": "soft", "dyna": "soft", "yna": "soft",
    "mai": "soft", "taw": "soft", "pe": "soft",
    "fy": "nasal", "dy": "soft",
    "ei": "soft|aspirate",
    "ein": "h-mutation", "eu": "h-mutation", "u": "h-mutation",
    "tri": "aspirate", "chwe": "aspirate",
    "a": "soft|aspirate", "â": "aspirate",
    "gyda": "aspirate", "tra": "aspirate",
    "yn": "nasal|soft_limited", "ym": "nasal", "yng": "nasal",
    "yr": "soft_limited",
    "fe": "soft", "mi": "soft",
    "dau": "soft", "dwy": "soft",
}

DEFINITE_ARTICLE_FORMS             = {"y", "yr", "r"}
# PATCH: per Wiktionary's Welsh mutations appendix, ll and rh do NOT undergo
# soft mutation after these triggers, even though other soft-mutable
# consonants do. A pipeline that doesn't know this will see the correct,
# un-mutated "llyfr"/"rhyd" after one of these triggers and wrongly flag it
# as erosion, inflating the erosion rate. yn/cyn/mor/pur are exempt for any
# POS; yr/y/r/un are exempt specifically for nouns (adjectives still mutate,
# e.g. "y lonnaf", "un ryfedd" -- Wiktionary's own examples show ll/rh DO
# mutate after these when the target is an adjective, only nouns are
# exempt). "un llaw" (one hand) is Wiktionary's explicit un+noun example --
# "un" was missing from this set entirely, so a genuine "un llaw"/"un rhyd"
# would have wrongly expected mutation and flagged the correct radical as
# erosion.
LL_RH_SOFT_EXEMPT_TRIGGERS_ANY_POS = {"yn", "cyn", "mor", "pur"}
LL_RH_SOFT_EXEMPT_TRIGGERS_NOUN_ONLY = {"y", "yr", "r", "un"}

# PATCH: per Wiktionary's Welsh mutations appendix, bare present-tense
# bod-forms do NOT trigger mutation on the word immediately following them
# in the general case -- that word is normally the SUBJECT ("Mae ci yn
# cysgu" -- "ci" stays radical, correctly). Bod-forms only trigger soft
# mutation via two much narrower environments: the predicate particle "yn"
# (already handled on its own via the "yn" branch in
# layer_1_trigger_detection) and a fronted predicate before a b-initial
# bod-form (rare, not modelled here). "sy"/"sydd" are NOT in this set --
# the relative sy(dd) genuinely does trigger soft mutation of a directly
# following predicate, that's a different, legitimate rule.
BOD_SUBJECT_EXEMPT_TRIGGERS = {"mae", "ydy", "oes"}
NUMERAL_FEM_SOFT_LIMITED_TRIGGERS  = {"un"}
NUMERAL_GENERAL_SOFT_TRIGGERS      = {"dau", "dwy"}
NASAL_NUMERAL_TRIGGERS             = {
    "pump", "saith", "wyth", "naw", "deng",
    "12", "15", "18", "20", "100",
    "deuddeg", "pymtheg", "deunaw", "ugain", "cant",
}
NASAL_NUMERAL_VALID_TARGETS        = {"blynedd", "blwydd", "diwrnod"}
MIXED_MUTATION_TRIGGERS            = {"ni", "nid", "na", "oni"}
# PATCH (Bucket 2): preposed adjectives -- when one of these precedes its
# noun (the inverse of normal Welsh noun-adjective order), the noun takes
# soft mutation. Kept as a closed lexicon (not just any ADJ) because the
# dependency check alone can't rule out other ADJ positions reliably, and
# because some of these words are polysemous (e.g. "rhyw" can be a noun
# meaning "sex/gender"); detection still requires the spaCy dependency
# check below to confirm the adjective genuinely modifies the following
# noun, not just lexical co-occurrence. "cyntaf" and other preposed
# superlatives are deliberately excluded -- Wiktionary notes they do not
# (reliably) trigger mutation. "prif" is excluded too: it has its own
# extra mutation-of-the-adjective-itself rule after a feminine article
# noun (e.g. "y brif fynedfa") that this simple noun-only layer can't
# represent correctly.
PREPOSED_ADJECTIVE_LEXICON         = {"hen", "rhyw", "ychydig", "amryw",
                                      "unig", "annwyl", "gau", "unrhyw", "holl", "ambell", "aml",
                                      }
# PATCH (Bucket 3b): known Welsh compound nouns whose second element takes
# soft mutation when fused (e.g. "creigardd" = craig + (g)ardd). When
# spoken/transcribed correctly the word surfaces as ONE fused token, so
# there's nothing to flag -- correct usage is invisible to a token-pair
# detector by design, which is fine (no false erosion). The only erosion
# signature this layer can catch is the compound "coming apart" back into
# two separate radical-form words (e.g. "craig gardd" instead of
# "creigardd"). This is a small seed list, not exhaustive -- expand as
# more compounds turn up in the corpus.
COMPOUND_NOUN_SECOND_ELEMENT       = {
    "craig": "gardd",     # creigardd "rock garden"
    "haf":   "dydd",      # hafddydd "summer's day"
    "bar":   "morwyn",    # barforwyn "barmaid"
    "hwyl":  "pren",      # hwylbren "mast"
    "rhwyd": "gwaith",    # rhwydwaith "network"
    "modur": "tŷ",        # modurdy "garage"
}
OBJ_DEPS                           = {"obj", "iobj"}
VOCAT_DEPS                         = {"vocative"}

PHANTOM_CONTEXT_TAGS = {
    "NF": "soft", "NM": "soft", "VBF": "soft", "VB": "soft",
    "VB+NM": "soft", "NM+VBF": "soft", "PLACE": "soft", "PERSON": "soft",
}
GENDER_BEARING_PREFIXES  = ("N", "PRON", "CARD", "ORD")
KNOWN_HOMOGRAPH_COLLISIONS = {
    "chi": "Pronoun 'chi' (you) collides with aspirate-mutated 'ci' (dog -> chi).",
}
PREP_TAG_PREFIXES = ("PREP", "CPREP")
PREDYN_TAGS       = {"PREDYN", "VERBADJ", "PREDYN+VERBADJ"}
REL_INT_TAGS      = {"PRONREL", "PART", "EXCL"}

DIGRAPHS      = ["ngh", "mh", "nh", "dd", "ff", "ll", "ph", "rh", "th",
                 "ch", "ng", "ts", "j"]
WELSH_FILLERS = {"ym", "er", "ah", "iawn", "gwybod", "chdi", "te", "ffeil"}
WELSH_VOWELS  = {"a", "e", "i", "o", "u", "w", "y", "â", "ê", "î", "ô", "û", "ŵ", "ŷ"}

# High-confidence English function words and common spoken insertions.
# These are checked before Welsh lemmatisation/tagger evidence because Cysill
# can occasionally Welshify English tokens (e.g. "the" -> "te", "for" -> "mor").
ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "an", "and", "or", "but", "of", "to", "for", "with", "that",
    "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "as", "at", "from", "by", "if", "then", "than",
    "not", "yes", "you", "your", "they", "he", "she", "them", "his", "her",
    "their", "my", "got", "get", "just", "what", "who", "when", "where",
    "why", "how", "can", "could", "would", "should", "will", "do", "does",
    "did", "have", "has", "had", "because", "about", "into", "over",
    "after", "before", "there", "here", "thing", "things", "people",
    "that's", "it's", "we're", "you're", "they're", "i'm", "don't",
    "doesn't", "didn't", "can't", "couldn't", "wouldn't", "shouldn't",
    "won't",
})
WELSH_ENGLISH_HOMOGRAPHS = frozenset({
    "a", "am", "i", "in", "mi", "no", "un",
})

# PATCH: Welsh orthography only uses combining marks for a closed set of
# seven vowels â ê î ô û ŵ ŷ. Acute accents appear in a small number of
# loanwords (café, acíwt) and grave accents in a handful of poetry conventions.
# All other combining marks -- umlaut/diaeresis (ö ü), tilde (õ ã), breve,
# caron, cedilla, ring, ogonek -- are completely absent from native Welsh.
#
# When Whisper is forced into cy mode on low-energy or non-Welsh audio it
# hallucinates tokens containing these foreign diacritics (e.g. töii, tõii).
# This table lets us reject those tokens orthographically rather than relying
# on confidence scores, which are unreliable for hallucinated tokens.
WELSH_LEGAL_DIACRITIC_CHARS = frozenset(
    "âêîôûŵŷ"   # circumflex (to bach) -- canonical Welsh diacritic
    "áéíóú"     # acute -- attested in loanwords, tolerated
    "àèìòù"     # grave -- rare but attested in poetry / some loanword spellings
)

# The combining marks that correspond to the legal diacritics above.
# Any other combining mark signals a non-Welsh character.
_COMBINING_LEGAL = frozenset({
    "\u0302",   # combining circumflex  (â ê î ô û ŵ ŷ)
    "\u0301",   # combining acute       (á é í ó ú)
    "\u0300",   # combining grave       (à è ì ò ù)
})

# ========================= BOD SUPPRESSION =========================
# All surface realisations of 'bod' (to be) suppressed as *mutation targets*.
# These are either suppletive (mae, yw, oedd -- no morphophonological link to
# a predictable radical) or genuinely mutated but in a verbal context that
# is not triggered by the preceding trigger word's mutation slot (fydd→fod,
# fu→bu). Analysing them as targets produces systematic false erosion counts.
#
# They are NOT suppressed as triggers -- mae/ydy/oes/dyma/dyna/sy/sydd
# remain in TRIGGERS and correctly predict soft mutation on the following
# noun/adjective. (Per Wiktionary's Welsh mutations appendix: "The verb
# form sydd, sy triggers soft mutation of a predicate noun or adjective" --
# sy/sydd were previously missing from TRIGGERS despite sitting right here
# in this set, silently undercounting every genuine sydd/sy-triggered
# context in the corpus.)
BOD_SURFACE_FORMS = frozenset({
    # Present tense
    "mae", "maen", "maent",
    "yw", "ydy", "ydyw",
    "oes",
    "sy", "sydd",
    # 1st/2nd/3rd person present
    "wyf", "wyt", "ydych", "ydym", "ydyn",
    # Imperfect / past
    "oedd", "oedden", "oeddech", "oeddet", "oeddwn", "oeddem",
    "roedd", "roedden", "roeddech", "roeddet", "roeddwn", "roeddem",
    # Preterite
    "bu", "buodd", "buon", "buoch", "buost", "bues", "buom",
    "fu", "fuodd", "fuon", "fuoch", "fuost", "fues", "fuom",
    # Future / conditional
    "bydd", "byddan", "byddwch", "byddi", "byddwn", "byddem", "byddent",
    "fydd", "fyddan", "fyddwch", "fyddi", "fyddwn", "fyddem", "fyddent",
    "byddai", "fyddai",
    # Subjunctive / literary conditional / pluperfect
    "byddaf", "byddo", "byddech",
    "bo", "boed", "foed",
    "buasai", "buasem", "buasent", "buaset", "buasech", "buaswn",
    "fuasai", "fuasem", "fuasent", "fuaset", "fuasech", "fuaswn",
    # Verbal noun (radical + soft-mutated form)
    "bod", "fod",
    # Colloquial / contracted forms
    "dw", "dwi",          # dw i (present 1sg)
    "on", "oni",          # o'n i (imperfect) -- 'oni' also a trigger but rare as target
    "dan", "dyn",         # dan ni / dyn ni
    "sa", "san", "set",   # sa i / conditional colloquial
    "se", "sen",          # southern conditional
})

# Welsh contractions that Whisper may keep joined but Cysill/spaCy will split.
# Format: surface_form -> [sub_token_1, sub_token_2, ...]
# Only the first sub-token inherits the original word's timestamp/confidence.
WELSH_CONTRACTION_SPLITS = {
    "i'r":  ["i", "'r"],
    "a'r":  ["a", "'r"],
    "o'r":  ["o", "'r"],
    "yn y": ["yn", "y"],
    "i'w":  ["i", "'w"],
    "a'i":  ["a", "'i"],
    "o'i":  ["o", "'i"],
}

# PATCH: known suppletive Welsh comparative/superlative adjectives. Welsh
# lemmatizers (Cysill and simplemma) collapse comparative/superlative forms
# to their citation (positive-grade) lemma for dictionary purposes -- e.g.
# "llai" (smaller) lemmatizes to "bach" (small), "gwell" (better) to "da"
# (good). That's correct lexicographically, but this pipeline uses
# get_welsh_lemma()'s result as a RADICAL for mutation purposes
# (initial_cluster(lemma), expected_surface_forms(lemma, ...)), and each
# comparative/superlative degree is its own suppletive radical with its own
# mutation behaviour -- "llai" itself is ll-initial and exempt from soft
# mutation after "yn", but the positive-grade lemma "bach" is b-initial and
# expects soft mutation. Using the positive-grade lemma here silently
# misclassified every comparative/superlative occurrence of these suppletive
# lexemes (confirmed live in the corpus: "yn llai" was being evaluated as if
# it were "bach"). Fixed by special-casing these forms to return their own
# surface degree as the "lemma"/radical, before the normal lookup path runs
# (see get_welsh_lemma in mutation_engine.py).
SUPPLETIVE_COMPARATIVE_SUPERLATIVE_RADICALS = {
    "gwell", "gorau",       # da (good) -> better, best
    "gwaeth", "gwaethaf",   # drwg (bad) -> worse, worst
    "mwy", "mwyaf",         # mawr (big) -> bigger, biggest
    "llai", "lleiaf",       # bach (small) -> smaller, smallest
    "nes", "nesaf",         # agos (near) -> nearer, nearest
}