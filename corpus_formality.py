"""
corpus_formality.py
====================
Computes a grounded, continuous, per-video formality score from data the
pipeline already collects -- a post-hoc analysis pass over already-written
CSVs under runs/, not a hand-assigned label collected at ingest time (see
corpus_io.py's CURATED_CHANNELS for why that approach was replaced).

Replaces channel_register (formal/informal/casual, fixed once per channel,
assigned by the researcher, with no way to check whether it was actually
right) with:

  - Heylighen & Dewaele's F-score (1999, "Formality of language:
    definition, measurement and behavioral determinants"): a validated,
    citable measure of formality from the balance of "context-independent"
    word classes (noun/adjective/preposition/article -- carry meaning
    without needing surrounding context) against "context-dependent" ones
    (pronoun/verb/adverb/interjection -- lean on situational/discourse
    context, characteristic of spontaneous speech). Computed straight from
    POS tag counts already sitting in pos_*.csv -- no new tagging work,
    no re-transcription, applies retroactively to every video already
    processed.
  - Filler-word rate (WELSH_FILLERS, already in mutation_tables.py) --
    disfluency density, a classic spontaneous-speech marker.
  - Lexical diversity (type-token ratio) -- repetition typical of
    unplanned speech.
  - Mean words per segment -- utterance planning/fragmentation proxy.
  - Code-switch rate -- reported as ITS OWN column, deliberately NOT
    folded into f_score. It's a real, independent contact-intensity
    signal, but mixing it into the formality composite would muddy
    whether an erosion-vs-formality correlation reflects "formality" or
    just "how much English is in the room" -- keep the two separable so
    each can be examined on its own.

Circularity guard: nothing here reads or derives from mutation/erosion
status -- mutations_*.csv is never opened by this module. The formality
score is computed entirely independently of the phenomenon it will later
be correlated against; that correlation happens downstream in
corpus_analyzer.py, not here.

Usage:
    python corpus_formality.py
        -- scans every video already under runs/, computes all of the
           above per video, writes video_formality.csv to OUT_DIR. Safe
           to re-run any time (e.g. after a new batch) -- always
           recomputed fresh from whatever's on disk, nothing incremental
           to go stale.
"""
import sys

import pandas as pd

from corpus_io import MUT_DIR, OUT_DIR
from mutation_tables import WELSH_FILLERS


# ========================= HEYLIGHEN-DEWAELE F-SCORE =========================
# "Context-independent" word classes -- carry their meaning without
# leaning on the surrounding situation/discourse, characteristic of
# planned/formal language.
CONTEXT_INDEPENDENT_POS = {"NOUN", "PROPN", "ADJ", "ADP", "DET"}
# "Context-dependent" word classes -- lean on situational/discourse
# context to be interpretable, characteristic of spontaneous speech.
CONTEXT_DEPENDENT_POS   = {"PRON", "VERB", "AUX", "ADV", "INTJ"}


def _f_score_from_pos_counts(pos_counts, total):
    """
    Heylighen & Dewaele (1999): F = 50 + (sum(context-independent %) -
    sum(context-dependent %)) / 2. Returns None if there's nothing to
    compute from (total == 0) rather than dividing by zero -- an absent
    score is more honest than a fabricated 50 (the formula's own
    "perfectly balanced" midpoint, which would misleadingly look like a
    real measurement of a genuinely balanced video rather than "no data").
    """
    if not total:
        return None
    indep_pct = sum(pos_counts.get(p, 0) for p in CONTEXT_INDEPENDENT_POS) / total * 100
    dep_pct   = sum(pos_counts.get(p, 0) for p in CONTEXT_DEPENDENT_POS) / total * 100
    return 50 + (indep_pct - dep_pct) / 2


def _discover_video_dirs():
    """
    Every video folder under runs/<stamp>/<slug>/ that has at least a
    pos_*.csv -- mirrors corpus_analyzer.load_and_merge_mutations()'s
    discovery pattern, but keyed by video FOLDER (this module needs
    segments/words/lemmas/pos together per video, not just mutations).
    """
    pos_files = [p for p in MUT_DIR.rglob("pos_*.csv") if "_deleted" not in p.parts]
    return sorted({p.parent for p in pos_files})


def _find_sibling_csv(video_dir, prefix, pos_csv_stem):
    """pos_<stamp>_<slug>.csv -> <prefix>_<stamp>_<slug>.csv, same folder
    -- every per-video output file shares the same "<stamp>_<slug>"
    folder_name (see corpus_io._video_slug())."""
    folder_name = pos_csv_stem[len("pos_"):]
    candidate = video_dir / f"{prefix}_{folder_name}.csv"
    return candidate if candidate.exists() else None


def compute_video_formality(video_dir):
    """
    Returns one dict of formality metrics for this video, or None if the
    required files aren't there (e.g. an incomplete/orphaned folder --
    see corpus_io.cleanup_incomplete_video_dirs for why those can exist).
    """
    pos_csvs = sorted(video_dir.glob("pos_*.csv"))
    if not pos_csvs:
        return None
    pos_csv = pos_csvs[0]

    try:
        pos_df = pd.read_csv(pos_csv, encoding="utf-8-sig")
    except Exception as e:
        print(f"  ⚠️ Couldn't read {pos_csv.name}: {e}")
        return None
    if pos_df.empty or "video_url" not in pos_df.columns:
        return None

    video_url   = pos_df["video_url"].iloc[0]
    video_title = pos_df["video_title"].iloc[0] if "video_title" in pos_df.columns else video_dir.name
    source      = pos_df["source"].iloc[0] if "source" in pos_df.columns else None

    # ---- F-score: POS counts already in pos_*.csv ----
    # spaCy primary (spacy_pos, the raw UD tag -- has the DET/INTJ
    # granularity Cysill's own tagset doesn't distinguish; cysill_coarse_pos
    # folds interjections/particles both into "PART"), falling back to
    # cysill_coarse_pos only for rows spaCy never tagged at all. A row
    # that falls back this way and happens to be a genuine interjection
    # will be undercounted on the context-dependent side -- a known,
    # minor limitation of Cysill's tagset, not a bug in this scoring.
    if "spacy_pos" in pos_df.columns:
        valid_spacy = pos_df["spacy_pos"].notna() & (pos_df["spacy_pos"] != "none")
        pos_col = pos_df["spacy_pos"].where(valid_spacy, pos_df.get("cysill_coarse_pos"))
    else:
        pos_col = pos_df.get("cysill_coarse_pos")
    pos_counts = pos_col.value_counts().to_dict() if pos_col is not None else {}
    total_tagged = int(sum(pos_counts.values()))
    f_score = _f_score_from_pos_counts(pos_counts, total_tagged)

    # ---- Filler-word rate ----
    words_csv = _find_sibling_csv(video_dir, "words", pos_csv.stem)
    filler_rate = None
    if words_csv is not None:
        try:
            words_df = pd.read_csv(words_csv, encoding="utf-8-sig")
            if "word" in words_df.columns and len(words_df):
                normed = words_df["word"].astype(str).str.lower().str.strip(".,!?;:'\"()[]")
                filler_rate = normed.isin(WELSH_FILLERS).sum() / len(words_df)
        except Exception as e:
            print(f"  ⚠️ Couldn't read {words_csv.name}: {e}")

    # ---- Code-switch rate + video word count: already on segments_*.csv,
    # stamped once per video onto every row -- no recomputation needed.
    # Reported as its OWN column, never folded into f_score (see module
    # docstring). ----
    segments_csv = _find_sibling_csv(video_dir, "segments", pos_csv.stem)
    video_word_count = None
    codeswitch_rate = None
    mean_words_per_segment = None
    if segments_csv is not None:
        try:
            seg_df = pd.read_csv(segments_csv, encoding="utf-8-sig")
            if len(seg_df):
                if "video_word_count" in seg_df.columns:
                    video_word_count = seg_df["video_word_count"].iloc[0]
                if "video_codeswitch_word_count" in seg_df.columns and video_word_count:
                    codeswitch_rate = seg_df["video_codeswitch_word_count"].iloc[0] / video_word_count
                if video_word_count:
                    mean_words_per_segment = video_word_count / len(seg_df)
        except Exception as e:
            print(f"  ⚠️ Couldn't read {segments_csv.name}: {e}")

    # ---- Lexical diversity (type-token ratio) ----
    lemmas_csv = _find_sibling_csv(video_dir, "lemmas", pos_csv.stem)
    ttr = None
    if lemmas_csv is not None:
        try:
            lemma_df = pd.read_csv(lemmas_csv, encoding="utf-8-sig")
            if len(lemma_df) and "lemma" in lemma_df.columns:
                fallback_col = lemma_df["normalized_word"] if "normalized_word" in lemma_df.columns else lemma_df["word"]
                lemma_col = lemma_df["lemma"].fillna(fallback_col)
                ttr = lemma_col.nunique() / len(lemma_df)
        except Exception as e:
            print(f"  ⚠️ Couldn't read {lemmas_csv.name}: {e}")

    return {
        "video_url":              video_url,
        "video_title":             video_title,
        "source":                  source,
        "f_score":                 round(f_score, 2) if f_score is not None else None,
        "pos_tagged_word_count":   total_tagged,
        "filler_rate":             round(filler_rate, 4) if filler_rate is not None else None,
        "lexical_diversity_ttr":   round(ttr, 4) if ttr is not None else None,
        "mean_words_per_segment":  round(mean_words_per_segment, 2) if mean_words_per_segment is not None else None,
        # Reported separately -- see module docstring. NOT an input to f_score.
        "codeswitch_rate":         round(codeswitch_rate, 4) if codeswitch_rate is not None else None,
        "video_word_count":        video_word_count,
    }


def build_video_formality_table():
    video_dirs = _discover_video_dirs()
    if not video_dirs:
        print(f"No pos_*.csv files found under {MUT_DIR}. Run the pipeline first.")
        return pd.DataFrame()

    rows = []
    for video_dir in video_dirs:
        row = compute_video_formality(video_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No videos with usable pos_*.csv data found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # A video processed more than once (multiple runs/<stamp>/ for the
    # same video_url) will have appeared once per run -- keep the most
    # recently written one so this table has exactly one row per video,
    # matching how corpus_analyzer.py's own merged corpus dedupes.
    df = df.drop_duplicates(subset="video_url", keep="last")
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing per-video formality scores...")
    df = build_video_formality_table()
    if df.empty:
        sys.exit(1)

    out_path = OUT_DIR / "video_formality.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n{len(df)} video(s) scored. Written to: {out_path}")

    print("\nF-score distribution (higher = more formal):")
    print(df["f_score"].describe().round(2).to_string())

    if "source" in df.columns:
        print("\nMean F-score by source (sanity check -- compare against your own")
        print("prior expectations for these channels, now that channel_register")
        print("no longer hand-asserts an answer):")
        by_source = df.dropna(subset=["f_score"]).groupby("source")["f_score"].agg(["mean", "count"])
        for src, r in by_source.sort_values("mean", ascending=False).iterrows():
            print(f"  {src:<60} mean={r['mean']:.1f}  n={int(r['count'])}")


if __name__ == "__main__":
    main()
