"""
spacy_tagging.py
=================
Everything about the Welsh spaCy dependency parser (cy_ud_cy_ccg): loading
the model once and turning a spaCy Doc into the plain-dict token format
the rest of the pipeline works with (so nothing outside this file needs
to import spacy or know about Doc/Token objects directly).
"""
from tqdm import tqdm

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not installed -- dependency parser disabled.")

SPACY_NLP = None


def load_spacy():
    global SPACY_NLP
    if not SPACY_AVAILABLE:
        return False
    if SPACY_NLP is not None:
        return True
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SPACY_NLP = spacy.load("cy_ud_cy_ccg")
        print("✅ Welsh dependency parser loaded.")
        return True
    except Exception as e:
        print(f"⚠️  Could not load Welsh dependency parser: {e}")
        return False


def mark_consumed(word_dict):
    """Marks a word (a dict from the shared preprocessed-word-stream list
    every detection branch walks) as already having been scored as a
    mutation TARGET by some rule this pass, so a later rule walking the
    same list doesn't score the same physical mutation event a second
    time as if the word were untouched.

    Internal bookkeeping only, same convention as the existing "_seg_id"/
    "_clause_boundary_after" fields in mutation_engine.py's preprocessing:
    never leaks into CSV output, since output rows are built from explicit
    key-selection dicts, not from dumping a word dict wholesale.

    Lives here (not in mutation_engine.py) because every branch that walks
    this same word stream -- prep_engine.py, plural_engine.py, not just
    mutation_engine.py -- needs the same protection from the same class of
    bug: a rule that lands on a word as a TARGET, then advances the loop
    onto that exact word, where a different, independently-triggered rule
    could re-score it as if it were untouched. See mutation_engine.py's
    process_comprehensive_mutations for the confirmed case this fixes
    (Layer 1A's target reused as Layer 1H/1I's current_node)."""
    word_dict["_consumed_as_target"] = True


def was_consumed(word_dict):
    """True if mark_consumed() was already called on this word dict this
    pass."""
    return bool(word_dict.get("_consumed_as_target"))


def extract_gender_from_spacy(spacy_token):
    """Reads UD Gender morph feature off a parsed token dict (see
    parse_spacy_doc's "gender" key) into this project's shared friendly
    vocabulary ("feminine"/"masculine"/None). Relocated here from
    mutation_engine.py -- reading a generic UD morph feature off a spaCy
    token isn't mutation-specific logic, and prep_engine.py/
    plural_engine.py need the same capability without importing
    mutation_engine.py's internals just to get it. mutation_engine.py
    still uses this under the same name -- re-exported there via a plain
    import, so every existing call site keeps working unchanged."""
    if not spacy_token:
        return None
    g = spacy_token.get("gender")
    if g == "Fem":
        return "feminine"
    elif g == "Masc":
        return "masculine"
    return None


def extract_number_from_spacy(spacy_token):
    """Reads UD Number morph feature off a parsed token dict into this
    project's shared friendly vocabulary ("plural"/"singular"/None). Same
    relocation rationale as extract_gender_from_spacy above -- this is
    plural_engine.py's primary signal (see that module's docstring)."""
    if not spacy_token:
        return None
    n = (spacy_token.get("morph") or {}).get("Number")
    if n == "Plur":
        return "plural"
    elif n == "Sing":
        return "singular"
    return None


def parse_spacy_doc(text):
    if SPACY_NLP is None:
        return None
    try:
        doc = SPACY_NLP(text)
        tokens = []
        for t in doc:
            morph = dict(t.morph.to_dict()) if t.morph else {}
            head_morph = dict(t.head.morph.to_dict()) if t.head.morph else {}
            tokens.append({
                "text":     t.text,
                "lemma":    t.lemma_,
                "dep":      t.dep_,
                "head":     t.head.text,
                "head_dep": t.head.dep_,
                "head_pos": t.head.pos_,
                # PATCH: needed to distinguish a finite verb from a verb-noun
                # governing this token -- POS alone (VERB/NOUN) is NOT
                # reliable for this in cy_ud_cy_ccg: verb-nouns can surface
                # as either POS tag, distinguished only by this feature.
                # Per UD_Welsh-CCG docs, VerbForm takes Fin/FinRel/Vnoun and
                # occurs on NOUN, VERB, and AUX tokens alike.
                "head_verbform": head_morph.get("VerbForm"),
                "pos":      t.pos_,
                "morph":    morph,
                "mutation": morph.get("Mutation"),
                "gender":   morph.get("Gender"),
                "is_punct": t.is_punct or t.is_space or t.pos_ in ("PUNCT", "SPACE"),
            })
        return tokens
    except Exception as e:
        tqdm.write(f" ⚠️ spaCy parse error: {e}")
        return None