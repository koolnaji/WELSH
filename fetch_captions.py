r"""
fetch_captions.py -- pulls Welsh YouTube captions (manual or auto-generated)
for a video and parses them into timestamped segments, for corroborating
against Whisper's own transcript. NOT part of the automated pipeline yet.
Same standalone convention as the other scan_*.py / alaw.py scripts --
duplicated constants, no dependency on the rest of the pipeline.

Why this matters (per the mae/gan investigation): Whisper's transcript
tracks what was actually SPOKEN (including genuine erosion), while
captions track what someone believed was SUPPOSED to be said (often
normalized back to standard written Welsh, which can silently erase real
colloquial mutation slippage). Neither is ground truth on its own -- but
where they DISAGREE, that disagreement itself is a useful signal: either
Whisper mis-heard something, or the speaker genuinely deviated from what
the (human) captioner expected. This script only fetches + parses; it
doesn't decide which side is "right" for any given mismatch -- that's a
job for a future diff/alignment pass once both sides are in comparable
form.

IMPORTANT: manual (human-authored) vs automatic (YouTube's own ASR)
captions are NOT the same thing and this script tells you which you got.
Manual captions are still a human's *belief* about what was said, prone to
normalization. Automatic captions are a second, independent ASR guess --
useful as a cross-check against Whisper's specific failure modes, but not
ground truth either.

Usage:
    python fetch_captions.py <youtube_url_or_id>
        -- lists available Welsh subtitle tracks (manual + automatic),
           downloads whichever exists (manual preferred if both exist),
           and writes a parsed segments CSV to the captions/ folder.

    python fetch_captions.py <youtube_url_or_id> --prefer-auto
        -- downloads automatic captions even if manual ones exist too
           (useful for deliberately comparing both against each other).

    python fetch_captions.py <youtube_url_or_id> --list-only
        -- just reports what's available, downloads nothing.
"""
import argparse
import difflib
import re
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import yt_dlp

# PATCH: previously duplicated these here (Path.home() / "Documents" /
# "welsh_analysis") per the old standalone-script pattern -- silently
# diverged from mutation_engine.py's actual BASE_DIR (WELSH_ANALYSIS_DIR,
# falling back to Path.home() / "welsh_analysis", no "Documents"), so
# this looked for/wrote transcripts and captions in a different folder
# than the one welsh_pipeline.py actually uses. Importing instead of
# duplicating keeps every file in the project pointed at the same place.
from mutation_engine import BASE_DIR, CAPTIONS_DIR, TRANS_DIR

WELSH_LANG_CODES = ["cy", "cy-GB", "cy-orig"]

# Matches yt-dlp's yield in captions/*.vtt: a timestamp line followed by
# one or more text lines, blocks separated by blank lines.
VTT_CUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})"
)

# Matches mutations_*.csv's "12.3s - 15.1s" timestamp format -- same
# pattern scan_dom_verbnoun_impact.py uses.
TIMESTAMP_RE = re.compile(r"([\d.]+)s\s*-\s*([\d.]+)s")

# ===================== duplicated from mutation_engine.py =====================
# Kept manually in sync rather than imported -- same standalone convention
# as every other scan_*.py script. Only the pieces actually needed here
# (word normalization + the three mutation tables, for Rule 3's candidate-
# set gating) are duplicated, not the whole trigger/detection machinery.
DIGRAPHS = ["ngh", "mh", "nh", "dd", "ff", "ll", "ph", "rh", "th",
            "ch", "ng", "ts", "j"]
SOFT_MUTATION = {
    "p": "b", "t": "d", "c": "g", "b": "f", "d": "dd", "g": "",
    "ll": "l", "rh": "r", "m": "f"
}
NASAL_MUTATION    = {"p": "mh", "t": "nh", "c": "ngh", "b": "m", "d": "n", "g": "ng"}
ASPIRATE_MUTATION = {"p": "ph", "t": "th", "c": "ch"}


def normalize_word(word):
    if not word:
        return ""
    word = str(word).replace("\u2018", "'").replace("\u2019", "'") \
                     .replace("\u201c", '"').replace("\u201d", '"')
    w = word.lower().strip(".,!?;:'\"()[]")
    w = w.replace("'r", "").replace("'n", "yn")
    return w


def initial_cluster(word):
    w = normalize_word(word)
    for d in sorted(DIGRAPHS, key=len, reverse=True):
        if w.startswith(d):
            return d
    return w[0] if w else ""


# ===================== Rule 1: segment-level mismatch gate =====================
# "Substantial mismatch": a caption segment that overlaps a Whisper segment
# in time but diverges from it by more than this fraction of their combined
# word content -- as a RATIO, not a raw phoneme/word count, so short and
# long segments are held to the same standard rather than short segments
# tripping the threshold trivially. Below this ratio, Whisper is trusted by
# default for that segment (either because the caption genuinely diverges
# a lot, or because -- separately tracked -- there was no caption there at
# all to compare against).
MISMATCH_RATIO_THRESHOLD = 0.5


def segment_match_ratio(text_a, text_b):
    import difflib
    return difflib.SequenceMatcher(None, str(text_a).split(), str(text_b).split()).ratio()


# ===================== Rule 2: POS-struggle proxy =====================
# "Functions weirdly" operationalized: a well-formed sentence gets tagged
# without unresolved POS -- spaCy's "X" tag is its own admission that a
# token didn't fit any recognized category. A spike in X-tag rate is what
# garbled/ungrammatical ASR output (e.g. "praed popeth sydd i angen am
# datlu ar da") looks like to the tagger. This only uses spaCy, not a
# Cysill lookup-failure signal too -- kept intentionally lightweight so
# this pass doesn't hit the rate-limited Cysill API per row; that's a
# reasonable future addition if this proxy turns out too noisy on its own.
POS_STRUGGLE_X_RATE = 0.15
POS_STRUGGLE_MARGIN = 0.10


def pos_struggle_score(nlp, text):
    text = str(text or "").strip()
    if not text:
        return 1.0
    doc = nlp(text)
    tokens = [t for t in doc if not (t.is_punct or t.is_space)]
    if not tokens:
        return 1.0
    unresolved = sum(1 for t in tokens if t.pos_ == "X")
    return unresolved / len(tokens)


# ===================== Rule 3: candidate-set gating =====================
def generate_mutation_candidates(lemma):
    """All linguistically valid surface forms of `lemma` (radical + soft/
    nasal/aspirate, whichever apply to its initial cluster), mapped to
    which mutation type each one is. A caption word that ISN'T one of
    these (e.g. caption 'chath' vs a hypothetical Whisper 'dath') isn't a
    mutation-related mismatch at all -- it's unrelated ASR/caption noise,
    and gets excluded from consideration entirely rather than treated as
    an erosion candidate."""
    lemma = normalize_word(lemma)
    if not lemma:
        return {}
    ic = initial_cluster(lemma)
    rest = lemma[len(ic):]
    candidates = {lemma: "radical"}
    for table, mtype in ((SOFT_MUTATION, "soft"), (NASAL_MUTATION, "nasal"),
                         (ASPIRATE_MUTATION, "aspirate")):
        mutated_initial = table.get(ic)
        if mutated_initial is not None:
            candidates[mutated_initial + rest] = mtype
    return candidates


def differs_only_in_initial(word_a, word_b):
    """True when two words differ ONLY in their leading consonant/digraph
    -- i.e. exactly the position mutation happens at. This is deliberately
    narrow: mutation only ever changes the single leading cluster, never
    anything else in the word, so a mismatch anywhere else isn't a
    mutation-site disagreement at all."""
    def strip_initial(w):
        ic = initial_cluster(w)
        return w[len(ic):]
    return word_a != word_b and strip_initial(word_a) == strip_initial(word_b)


def align_target_word(whisper_tokens, caption_tokens, target_index):
    """Finds the caption token aligned to whisper_tokens[target_index]
    via word-level sequence alignment (not straight positional indexing,
    since the two transcripts can differ in length before/after the
    target word). Returns (caption_word_or_None, alignment_kind)."""
    import difflib
    sm = difflib.SequenceMatcher(None, whisper_tokens, caption_tokens, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= target_index < i2:
            if tag == "equal":
                return caption_tokens[j1 + (target_index - i1)], "equal"
            if tag == "replace":
                span_c = j2 - j1
                if span_c == 0:
                    return None, "deleted_in_caption"
                offset = min(target_index - i1, span_c - 1)
                return caption_tokens[j1 + offset], "replace"
            if tag == "delete":
                return None, "deleted_in_caption"
    return None, "not_found"


def _vtt_ts_to_seconds(ts):
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def list_available_tracks(url):
    """Returns (manual_langs, auto_langs) -- lists of Welsh-ish language
    codes found in each category, without downloading anything."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    manual = info.get("subtitles") or {}
    auto   = info.get("automatic_captions") or {}

    manual_cy = [lang for lang in manual if lang in WELSH_LANG_CODES or lang.startswith("cy")]
    auto_cy   = [lang for lang in auto if lang in WELSH_LANG_CODES or lang.startswith("cy")]

    return manual_cy, auto_cy, info.get("title", url)


def download_captions(url, prefer_auto=False, out_dir=CAPTIONS_DIR):
    """Downloads whichever Welsh caption track is available (manual
    preferred unless prefer_auto=True) as a .vtt file. Returns the path,
    the language code used, and whether it was 'manual' or 'automatic'."""
    manual_cy, auto_cy, title = list_available_tracks(url)

    if not manual_cy and not auto_cy:
        return None, None, None, title

    if prefer_auto and auto_cy:
        lang, kind = auto_cy[0], "automatic"
    elif manual_cy:
        lang, kind = manual_cy[0], "manual"
    elif auto_cy:
        lang, kind = auto_cy[0], "automatic"
    else:
        return None, None, None, title

    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "skip_download": True,
        "writesubtitles": kind == "manual",
        "writeautomaticsub": kind == "automatic",
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info.get("id")
    # yt-dlp names the file <id>.<lang>.vtt
    candidates = list(out_dir.glob(f"{video_id}.{lang}.vtt"))
    if not candidates:
        candidates = list(out_dir.glob(f"{video_id}*.vtt"))
    if not candidates:
        return None, lang, kind, title

    return candidates[0], lang, kind, title


def parse_vtt(vtt_path):
    """Parses a .vtt file into a list of dicts: start, end, text.
    Same shape as segment_start/segment_end/segment_text in your existing
    segments_*.csv files, so this lines up for a direct comparison/merge."""
    raw = Path(vtt_path).read_text(encoding="utf-8", errors="replace")
    blocks = raw.split("\n\n")
    segments = []
    for block in blocks:
        m = VTT_CUE_RE.search(block)
        if not m:
            continue
        start = _vtt_ts_to_seconds(m.group(1))
        end   = _vtt_ts_to_seconds(m.group(2))
        text_lines = block[m.end():].strip().splitlines()
        # Strip inline VTT positioning/styling tags (<c>, timestamps
        # embedded in auto-captions for word-level highlighting, etc.)
        text = " ".join(re.sub(r"<[^>]+>", "", line).strip() for line in text_lines)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            segments.append({"segment_start": start, "segment_end": end,
                              "segment_text": text})
    # yt-dlp/YouTube's auto-caption VTTs often repeat overlapping cues
    # (rolling word-by-word display) -- collapse consecutive duplicate
    # text before returning, keeping the first occurrence's timing.
    deduped = []
    for seg in segments:
        if deduped and deduped[-1]["segment_text"] == seg["segment_text"]:
            continue
        deduped.append(seg)
    return deduped


def find_segments_csv(mutations_csv_path):
    """Mirrors manual_editing.py's / scan_dom_verbnoun_impact.py's
    find_segments_csv -- same layout, same fallback search.

    # PATCH: fast-path candidate updated for _video_slug()'s restored
    # run-then-video nesting (mutations/<stamp>/<slug>/, matching
    # transcriptions/<stamp>/<slug>/ symmetrically) -- mutations_csv_path
    # now has TWO parent levels to read off (slug, then stamp), not one.
    # Also switched the rglob fallback's search root from a fixed
    # mutations_csv_path.parents[N] hop count to BASE_DIR directly -- that
    # fixed-depth assumption broke the first time folder nesting changed
    # (this is the second time), since it silently resolves to the wrong
    # directory instead of erroring when depth shifts.
    """
    stem = mutations_csv_path.stem
    if not stem.startswith("mutations_"):
        return None
    folder_name = stem[len("mutations_"):]

    slug  = mutations_csv_path.parent.name
    stamp = mutations_csv_path.parent.parent.name
    candidate = TRANS_DIR / stamp / slug / f"segments_{folder_name}.csv"
    if candidate.exists():
        return candidate

    target_name = f"segments_{folder_name}.csv"
    matches = list(BASE_DIR.rglob(target_name))
    return matches[0] if matches else None


def find_words_csv(mutations_csv_path):
    """Mirrors find_segments_csv -- same layout, same fallback search --
    but for words_*.csv. Needed here (and not by find_segments_csv's
    caller previously) for its per-word word_start/word_end timestamps,
    which segments_*.csv doesn't have at word granularity."""
    mutations_csv_path = Path(mutations_csv_path)
    stem = mutations_csv_path.stem
    if not stem.startswith("mutations_"):
        return None
    folder_name = stem[len("mutations_"):]

    slug  = mutations_csv_path.parent.name
    stamp = mutations_csv_path.parent.parent.name
    candidate = TRANS_DIR / stamp / slug / f"words_{folder_name}.csv"
    if candidate.exists():
        return candidate

    target_name = f"words_{folder_name}.csv"
    matches = list(BASE_DIR.rglob(target_name))
    return matches[0] if matches else None


# ===================== Whole-video word-level alignment =====================
# PATCH: replaces the previous segment-pooling approach entirely
# (covering_segment / covering_segments_text / MISMATCH_RATIO_THRESHOLD /
# segment_match_ratio-as-a-gate / align_target_word run per-row on a local
# window). That approach compared TEXT WINDOWS bounded by an arbitrary
# time margin, which reduced every case to "is this window's word-overlap
# ratio above some threshold" -- a threshold with no principled value that
# needed constant retuning, and which discarded real signal (e.g. manual
# captions genuinely normalizing colloquial forms toward standard written
# Welsh -- see the module docstring) by lumping it in with pure alignment
# noise whenever the ratio came out low for either reason.
#
# This aligns the WHOLE video's Whisper word stream against the WHOLE
# video's caption word stream ONCE (not per mutation row), via
# difflib.SequenceMatcher. SequenceMatcher finds the longest matching
# blocks first and recursively resolves what's left before/after each one
# -- so a single deleted or substituted word doesn't cascade into
# permanent misalignment for the rest of the video, unlike a fixed-offset
# or purely positional walk would.
#
# The one real risk with pure content-based whole-video alignment:
# mutation trigger words are a small closed set of very short, very
# common words (yn, y, i, a...) that repeat dozens of times per video, so
# the algorithm could occasionally anchor a match to the WRONG occurrence
# of a common word far away in the video. Two safeguards against that:
#   1. Timing plausibility check -- every proposed word-to-word pairing is
#      checked against the two words' actual (approximate, for captions --
#      VTT only has segment-level timing) timestamps. A pairing whose
#      timestamps are wildly apart despite matching in content is almost
#      certainly the "wrong occurrence" failure mode, not a real
#      alignment, and gets flagged rather than trusted.
#   2. Neighbor corroboration -- an isolated one-word "equal" match
#      surrounded by replace/delete on both sides is weaker evidence than
#      a match sitting inside a longer run of consecutive agreement.
#      Recorded as a confidence signal (run_len) rather than treated as
#      binary match/no-match.

# Generous because caption timing is segment-level (a whole sentence/line
# of dialogue), not per-word -- every word in a caption segment inherits
# that segment's start time as an approximation, so a genuinely correct
# match can easily be several seconds off from the Whisper word's precise
# timestamp just from where in its segment that word happens to sit. This
# is a sanity check for "wildly implausible" (the wrong occurrence of a
# common word, potentially minutes away), not a precision filter.
ALIGNMENT_TIMING_TOLERANCE = 15.0

# Minimum consecutive-equal run length for a match to count as
# well-corroborated by its neighbors rather than an isolated coincidence
# -- see align_whole_video's docstring, safeguard 2.
NEIGHBOR_CORROBORATION_MIN_RUN = 3


def build_whisper_token_stream(words_df):
    """Whole-video, chronologically-ordered (word, timestamp) stream from
    words_*.csv. Uses each word's own word_end (real per-word timing from
    faster-whisper) -- matches what the mutations CSV's timestamp field
    records for a target word (see run_corroboration's target-word lookup)."""
    df = words_df.sort_values("word_start").reset_index(drop=True)
    tokens = [normalize_word(w) for w in df["word"]]
    times  = list(df["word_end"])
    return tokens, times, df


def build_caption_token_stream(caption_df):
    """Whole-video, chronologically-ordered (word, approx_timestamp)
    stream from a parsed captions CSV. VTT captions only carry per-SEGMENT
    timing, not per-word, so every word in a caption segment inherits
    that segment's segment_start as an approximation."""
    df = caption_df.sort_values("segment_start").reset_index(drop=True)
    tokens, times = [], []
    for _, row in df.iterrows():
        for w in str(row["segment_text"]).split():
            tokens.append(normalize_word(w))
            times.append(row["segment_start"])
    return tokens, times


def align_whole_video(whisper_tokens, whisper_times, caption_tokens, caption_times):
    """Aligns two full, chronologically-ordered token streams for one
    video via difflib.SequenceMatcher, and returns a dict mapping each
    whisper token index to its corresponding caption evidence:

        {
          whisper_idx: {
              "caption_word": str or None,
              "tag": "equal" | "replace" | "delete" | "replace_unmatched_length",
              "timing_ok": bool or None,   # None when caption_word is None
              "run_len": int,               # length of the opcode block this index sits in
          },
          ...
        }

    "delete" means the caption genuinely has nothing corresponding to
    this whisper word -- SequenceMatcher's recursive block-matching has
    already resolved everything it could around this point before
    concluding there's no caption counterpart, so this isn't a
    segmentation artifact the way the old approach's gaps could be.
    """
    sm = difflib.SequenceMatcher(None, whisper_tokens, caption_tokens, autojunk=False)
    mapping = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        run_len = i2 - i1
        if tag == "equal":
            for offset in range(i2 - i1):
                w_idx, c_idx = i1 + offset, j1 + offset
                dt = abs(whisper_times[w_idx] - caption_times[c_idx])
                mapping[w_idx] = {"caption_word": caption_tokens[c_idx], "tag": "equal",
                                   "timing_ok": dt <= ALIGNMENT_TIMING_TOLERANCE, "run_len": run_len}
        elif tag == "replace":
            span_w, span_c = i2 - i1, j2 - j1
            # Best-effort positional pairing within the replaced block --
            # a genuine word-for-word substitution (e.g. colloquial ->
            # formal normalization of the same word count) lines up
            # cleanly this way. Extra whisper words beyond the shorter
            # side of an uneven block have no clean partner and are left
            # unmapped rather than forced onto a guess.
            for offset in range(span_w):
                w_idx = i1 + offset
                if offset < span_c:
                    c_idx = j1 + offset
                    dt = abs(whisper_times[w_idx] - caption_times[c_idx])
                    mapping[w_idx] = {"caption_word": caption_tokens[c_idx], "tag": "replace",
                                       "timing_ok": dt <= ALIGNMENT_TIMING_TOLERANCE, "run_len": run_len}
                else:
                    mapping[w_idx] = {"caption_word": None, "tag": "replace_unmatched_length",
                                       "timing_ok": None, "run_len": run_len}
        elif tag == "delete":
            for offset in range(i2 - i1):
                w_idx = i1 + offset
                mapping[w_idx] = {"caption_word": None, "tag": "delete",
                                   "timing_ok": None, "run_len": run_len}
        # "insert" blocks have no whisper-side indices to attach evidence
        # to -- caption has an extra word Whisper never produced at all.
    return mapping


def find_whisper_index(words_df_sorted, target_norm, t_end, tolerance=0.05):
    """Locates the whisper-stream position (row position in the SAME sort
    order build_whisper_token_stream used) of the specific word a
    mutation row is about, by matching on its own recorded end-timestamp
    (precise to the millisecond in words_*.csv) plus the word text
    itself -- not by assuming any fixed relationship between the
    mutations CSV's row order and the words CSV's row order."""
    matches = words_df_sorted.index[
        (abs(words_df_sorted["word_end"] - t_end) <= tolerance) &
        (words_df_sorted["word"].apply(lambda w: normalize_word(w) == target_norm))
    ]
    return int(matches[0]) if len(matches) else None


def local_window_text(tokens, center_idx, window=6):
    """Small text window around a token index, for the POS-struggle proxy
    (rule 2) -- that check just needs "does the local grammar look
    sensible", not a precise segment boundary."""
    if center_idx is None:
        return ""
    lo, hi = max(0, center_idx - window), min(len(tokens), center_idx + window + 1)
    return " ".join(tokens[lo:hi])


def run_corroboration(mutations_csv_path, captions_csv_path, nlp=None):
    """The actual reinforcement pass: applies rules 1-3 to every row of a
    mutations_*.csv against the matching captions CSV (from a prior
    fetch_captions.py run) and the video's own words_*.csv, and writes
    the result back with new columns added.

    Deliberately additive-only: NEVER touches status/is_erosion/
    mutation_found/expected_mutation -- those are the pipeline's actual
    classification and shouldn't be silently rewritten by a second-opinion
    pass. Rows worth a second look get flagged=True (the existing
    pipeline mechanism) so they route into manual_editing.py's queue for
    a human decision, same as everything else that needs one.
    """
def run_corroboration(mutations_csv_path, captions_csv_path, nlp=None, cap_kind=None):
    """The actual reinforcement pass: applies rules 1-3 to every row of a
    mutations_*.csv against the matching captions CSV (from a prior
    fetch_captions.py run) and the video's own words_*.csv, and writes
    the result back with new columns added.

    Deliberately additive-only: NEVER touches status/is_erosion/
    mutation_found/expected_mutation -- those are the pipeline's actual
    classification and shouldn't be silently rewritten by a second-opinion
    pass. Rows worth a second look get flagged=True (the existing
    pipeline mechanism) so they route into manual_editing.py's queue for
    a human decision, same as everything else that needs one.

    cap_kind: "manual" or "automatic" if known (see download_captions's
    return value) -- persisted as its own column so downstream analysis
    can weigh a caption disagreement differently depending on whether it
    came from a human caption author (who may deliberately normalize
    colloquial speech toward standard written Welsh -- see module
    docstring) or YouTube's automatic ASR (a second machine guess, not an
    editorial choice). This function doesn't act on that distinction
    itself; it just makes sure it isn't lost by the time anyone wants to.
    """
    mutations_csv_path = Path(mutations_csv_path)
    captions_csv_path  = Path(captions_csv_path)

    words_path = find_words_csv(mutations_csv_path)
    if words_path is None:
        print(f"Couldn't find a matching words_*.csv for {mutations_csv_path.name} "
              f"-- can't corroborate without per-word Whisper timestamps.")
        sys.exit(1)

    if nlp is None:
        print("Loading Welsh spaCy model (cy_ud_cy_ccg)...")
        import spacy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = spacy.load("cy_ud_cy_ccg")

    words_df   = pd.read_csv(words_path, encoding="utf-8-sig")
    caption_df = pd.read_csv(captions_csv_path, encoding="utf-8-sig")
    mut_df     = pd.read_csv(mutations_csv_path, encoding="utf-8-sig")

    new_cols = {c: [] for c in (
        "caption_available", "caption_word", "caption_alignment_tag",
        "caption_alignment_confidence", "pos_struggle_whisper",
        "pos_struggle_caption", "caption_corroboration", "caption_flag_reason",
    )}
    counts = Counter()

    if words_df.empty or caption_df.empty:
        for _ in range(len(mut_df)):
            for c in new_cols:
                new_cols[c].append(None)
            counts["no_caption"] += 1
        for c, vals in new_cols.items():
            mut_df[c] = vals
        mut_df["caption_kind"] = cap_kind
        mut_df.to_csv(mutations_csv_path, index=False, encoding="utf-8-sig")
        print("No usable words/captions data -- nothing corroborated.")
        return

    # Build both whole-video token streams and align them ONCE -- not
    # per mutation row. See align_whole_video's docstring for the full
    # reasoning behind this replacing the old per-row segment-window
    # comparison.
    whisper_tokens, whisper_times, words_df_sorted = build_whisper_token_stream(words_df)
    caption_tokens, caption_times = build_caption_token_stream(caption_df)
    alignment_map = align_whole_video(whisper_tokens, whisper_times, caption_tokens, caption_times)

    for _, row in mut_df.iterrows():
        m = TIMESTAMP_RE.search(str(row.get("timestamp", "")))
        if not m:
            for c in new_cols:
                new_cols[c].append(None)
            new_cols["caption_corroboration"][-1] = "no_timestamp"
            counts["no_timestamp"] += 1
            continue
        t_start, t_end = float(m.group(1)), float(m.group(2))
        target_norm = normalize_word(row.get("following_word", ""))

        # PATCH: previously found the target word by re-tokenizing a
        # local text window and calling .index() on it -- fragile if the
        # word repeats in that window. Now locates it directly in
        # words_*.csv by its own recorded end-timestamp (millisecond
        # precision) plus the word text, which is unambiguous.
        w_idx = find_whisper_index(words_df_sorted, target_norm, t_end)
        if w_idx is None:
            for c in new_cols:
                new_cols[c].append(None)
            new_cols["caption_corroboration"][-1] = "no_alignment"
            counts["no_alignment"] += 1
            continue

        evidence     = alignment_map.get(w_idx)
        caption_word = evidence["caption_word"] if evidence else None
        align_tag    = evidence["tag"] if evidence else "not_found"
        timing_ok    = evidence["timing_ok"] if evidence else None
        run_len      = evidence["run_len"] if evidence else 0
        # Confidence: a match sitting inside a longer run of consecutive
        # agreement (run_len >= NEIGHBOR_CORROBORATION_MIN_RUN) is
        # trusted more than an isolated one-word coincidence -- see
        # module-level note on the "wrong occurrence of a common word"
        # risk this guards against.
        if caption_word is None:
            confidence = None
        elif timing_ok is False:
            confidence = "rejected"
        elif run_len >= NEIGHBOR_CORROBORATION_MIN_RUN:
            confidence = "high"
        else:
            confidence = "low"

        # Rule 2: local grammaticality proxy on a small word window
        # around the target, instead of a whole (arbitrarily-bounded)
        # segment -- doesn't need precise boundaries, just "does the
        # neighborhood parse sensibly".
        whisper_local = local_window_text(whisper_tokens, w_idx)
        struggle_w = pos_struggle_score(nlp, whisper_local)
        struggle_c = None
        caption_local = ""
        if caption_word is not None:
            same_word_positions = [i for i, t in enumerate(caption_tokens) if t == caption_word]
            if same_word_positions:
                c_idx = min(same_word_positions, key=lambda i: abs(caption_times[i] - t_end))
                caption_local = local_window_text(caption_tokens, c_idx)
                struggle_c = pos_struggle_score(nlp, caption_local)

        flag_reasons = []
        if struggle_c is not None and struggle_w > POS_STRUGGLE_X_RATE and (struggle_w - struggle_c) >= POS_STRUGGLE_MARGIN:
            flag_reasons.append(
                f"Whisper local POS-struggle={struggle_w:.2f} vs caption={struggle_c:.2f} "
                f"-- caption parses more cleanly, worth a manual reparse"
            )

        def _record(corroboration, extra_flags=None):
            counts[corroboration] += 1
            new_cols["caption_available"].append(caption_word is not None)
            new_cols["caption_word"].append(caption_word)
            new_cols["caption_alignment_tag"].append(align_tag)
            new_cols["caption_alignment_confidence"].append(confidence)
            new_cols["pos_struggle_whisper"].append(round(struggle_w, 3))
            new_cols["pos_struggle_caption"].append(round(struggle_c, 3) if struggle_c is not None else None)
            new_cols["caption_corroboration"].append(corroboration)
            all_flags = flag_reasons + (extra_flags or [])
            new_cols["caption_flag_reason"].append("; ".join(all_flags) if all_flags else None)

        # Rule 1a: no caption content aligned to this word at all --
        # "delete" means SequenceMatcher concluded, after resolving
        # everything around it, that the caption genuinely has nothing
        # here (your "Cymru -> (Nothing)" case). Distinct from simply
        # never having found the word at all.
        if caption_word is None:
            _record("caption_omits_word" if align_tag == "delete" else "no_alignment")
            continue

        # Rule 1b: content matched somewhere, but the timing sanity check
        # says it's implausibly far from this word's real timestamp --
        # most likely the algorithm anchored to the WRONG occurrence of a
        # common word elsewhere in the video, not a real alignment. Not
        # trusted as evidence either way.
        if timing_ok is False:
            _record("alignment_timing_implausible", [
                f"Caption word '{caption_word}' matched by content but its timing is "
                f"implausibly far from this word's timestamp -- likely aligned to the "
                f"wrong occurrence of a common word, not trusted."
            ])
            continue

        # Rule 3: word-level agreement + candidate-set gating at the
        # specific target word this row is about.
        if caption_word == target_norm:
            _record("confirmed_match")
            continue

        lemma = row.get("lemma") or target_norm
        candidates = generate_mutation_candidates(lemma)
        if caption_word not in candidates:
            # Not a valid mutated/radical form of this lemma at all
            # (e.g. caption 'dath' vs lemma 'cath') -- ASR/caption noise
            # unrelated to mutation, not evidence either way.
            _record("caption_word_not_lemma_candidate")
            continue

        if differs_only_in_initial(target_norm, caption_word):
            # Exactly the mutation-site position. Don't let the caption
            # overrule Whisper here -- this IS the phenomenon being
            # measured. But "don't overrule" isn't "don't distinguish" --
            # the four caption/Whisper combinations at this site carry
            # different evidential weight and get their own labels rather
            # than being collapsed into one bucket.
            caption_mut_type   = candidates[caption_word]
            caption_is_mutated = caption_mut_type != "radical"
            whisper_mutation_found = str(row.get("mutation_found", "none"))
            whisper_is_mutated = whisper_mutation_found != "none"

            if caption_is_mutated and not whisper_is_mutated:
                _record("confirms_erosion")
            elif not caption_is_mutated and whisper_is_mutated:
                _record("caption_disputes_mutation", [
                    f"Caption shows radical '{caption_word}' where Whisper "
                    f"found a {whisper_mutation_found} mutation -- Whisper's "
                    f"mutation call may be spurious, worth a listen."
                ])
            elif caption_is_mutated and whisper_is_mutated:
                _record("confirms_mutation_present")
            else:
                _record("confirms_no_mutation")
        else:
            # Mismatch OUTSIDE the mutation site. For manual captions
            # this could be a genuine editorial rewording (see module
            # docstring); for automatic captions it's most likely just a
            # second ASR system's own mishearing. Either way it's a real
            # phoneme-level disagreement, not the mutation phenomenon
            # itself -- flagged rather than silently applied.
            _record("caption_disagrees_non_initial", [
                f"Caption word '{caption_word}' disagrees with Whisper's "
                f"'{target_norm}' beyond just the mutation site -- likely an ASR "
                f"mishearing, or (if manual captions) a genuine editorial "
                f"rewording -- worth a listen."
            ])

    for c, vals in new_cols.items():
        mut_df[c] = vals
    mut_df["caption_kind"] = cap_kind

    if "flagged" not in mut_df.columns:
        mut_df["flagged"] = False
    needs_flag = mut_df["caption_flag_reason"].notna()
    mut_df.loc[needs_flag, "flagged"] = mut_df.loc[needs_flag, "flagged"].fillna(False) | True

    # Back up the pre-corroboration file ONCE -- if this is re-run later
    # (e.g. after re-fetching better captions), the backup still holds the
    # true original, not an already-corroborated intermediate version.
    backup_path = mutations_csv_path.with_name(mutations_csv_path.stem + "_precaption_backup.csv")
    if not backup_path.exists():
        shutil.copy2(mutations_csv_path, backup_path)

    mut_df.to_csv(mutations_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*70}")
    print(f"Corroboration complete: {mutations_csv_path.name}")
    print(f"{'='*70}")
    for label, n in counts.most_common():
        print(f"  {label:32s} {n:6d}")
    flagged_n = int(needs_flag.sum())
    print(f"\n  Rows newly flagged for manual review: {flagged_n}")
    print(f"  Original backed up to: {backup_path.name}")
    print(f"  Updated in place: {mutations_csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", nargs="?", help="YouTube URL or video ID (fetch mode)")
    ap.add_argument("--prefer-auto", action="store_true",
                     help="use automatic captions even if manual ones exist")
    ap.add_argument("--list-only", action="store_true",
                     help="just report what's available, download nothing")
    ap.add_argument("--corroborate", metavar="MUTATIONS_CSV",
                     help="run the caption-corroboration pass (rules 1-3) against this "
                          "mutations_*.csv instead of fetching. Requires --captions-csv.")
    ap.add_argument("--captions-csv", metavar="PATH",
                     help="the parsed captions CSV to corroborate against (the .csv "
                          "written next to a .vtt by a prior fetch run). Required with "
                          "--corroborate.")
    ap.add_argument("--caption-kind", choices=["manual", "automatic"], default=None,
                     help="whether --captions-csv came from a manual or automatic track "
                          "(printed by a prior fetch run) -- optional, persisted as its "
                          "own column if given.")
    args = ap.parse_args()

    if args.corroborate:
        if not args.captions_csv:
            ap.error("--corroborate requires --captions-csv PATH (from a prior fetch run)")
        run_corroboration(Path(args.corroborate), Path(args.captions_csv), cap_kind=args.caption_kind)
        return

    if not args.url:
        ap.error("either a YouTube URL, or --corroborate MUTATIONS_CSV --captions-csv PATH, is required")
    print(f"Checking available subtitle tracks...")
    manual_cy, auto_cy, title = list_available_tracks(args.url)
    print(f"Video: {title}")
    print(f"  Manual Welsh captions:    {manual_cy or 'none'}")
    print(f"  Automatic Welsh captions: {auto_cy or 'none'}")

    if not manual_cy and not auto_cy:
        print("\nNo Welsh captions of either kind found for this video.")
        sys.exit(1)

    if args.list_only:
        return

    print(f"\nDownloading ({'automatic' if args.prefer_auto else 'manual-preferred'})...")
    vtt_path, lang, kind, title = download_captions(args.url, prefer_auto=args.prefer_auto)
    if vtt_path is None:
        print("Download failed -- no file produced.")
        sys.exit(1)

    print(f"Saved {kind} captions ({lang}): {vtt_path}")
    if kind == "manual":
        print("NOTE: manual captions reflect what a human believed was said --")
        print("      may be normalized toward standard written Welsh, which can")
        print("      silently smooth over genuine colloquial mutation erosion.")
    else:
        print("NOTE: automatic captions are a SECOND, independent ASR guess --")
        print("      useful as a cross-check against Whisper's specific failure")
        print("      modes, but not ground truth either.")

    segments = parse_vtt(vtt_path)
    print(f"\nParsed {len(segments)} caption segment(s).")

    out_csv = vtt_path.with_suffix(".csv")
    import csv as csv_module
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv_module.DictWriter(f, fieldnames=["segment_start", "segment_end", "segment_text"])
        writer.writeheader()
        writer.writerows(segments)
    print(f"Wrote parsed segments to {out_csv}")
    print(f"\nThis is in the same segment_start/segment_end/segment_text shape as")
    print(f"your segments_*.csv files -- ready to align against the Whisper")
    print(f"transcript for the same video once you're ready to diff them.")


if __name__ == "__main__":
    main()