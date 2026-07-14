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