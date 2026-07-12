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
import re
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import yt_dlp

# Same location convention as the rest of the pipeline -- duplicated here
# rather than imported, per the standalone-script pattern.
BASE_DIR      = Path.home() / "Documents" / "welsh_analysis"
CAPTIONS_DIR  = BASE_DIR / "captions"
TRANS_DIR     = BASE_DIR / "transcriptions"

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
    find_segments_csv -- same layout, same fallback search."""
    stem = mutations_csv_path.stem
    if not stem.startswith("mutations_"):
        return None
    folder_name = stem[len("mutations_"):]
    stamp = mutations_csv_path.parent.name

    candidate = TRANS_DIR / stamp / folder_name / f"segments_{folder_name}.csv"
    if candidate.exists():
        return candidate

    target_name = f"segments_{folder_name}.csv"
    search_root = mutations_csv_path.parents[2] if len(mutations_csv_path.parents) >= 3 else BASE_DIR
    matches = list(search_root.rglob(target_name))
    return matches[0] if matches else None


def covering_segment(df, t_start, t_end):
    """Returns the row in df whose segment_start/segment_end overlaps
    [t_start, t_end], or None. Same overlap logic as
    scan_dom_verbnoun_impact.py's get_sentence_text."""
    if df is None or "segment_start" not in df.columns or "segment_end" not in df.columns:
        return None
    covering = df[(df["segment_start"] <= t_end) & (df["segment_end"] >= t_start)]
    if covering.empty:
        return None
    return covering.iloc[0]


def run_corroboration(mutations_csv_path, captions_csv_path, nlp=None):
    """The actual reinforcement pass: applies rules 1-3 to every row of a
    mutations_*.csv against the matching captions CSV (from a prior
    fetch_captions.py run) and the video's own segments_*.csv, and writes
    the result back with new columns added.

    Deliberately additive-only: NEVER touches status/is_erosion/
    mutation_found/expected_mutation -- those are the pipeline's actual
    classification and shouldn't be silently rewritten by a second-opinion
    pass. Rows worth a second look get flagged=True (the existing
    pipeline mechanism) so they route into manual_editing.py's queue for
    a human decision, same as everything else that needs one.
    """
    mutations_csv_path = Path(mutations_csv_path)
    captions_csv_path  = Path(captions_csv_path)

    seg_path = find_segments_csv(mutations_csv_path)
    if seg_path is None:
        print(f"Couldn't find a matching segments_*.csv for {mutations_csv_path.name} "
              f"-- can't corroborate without the Whisper segment text.")
        sys.exit(1)

    if nlp is None:
        print("Loading Welsh spaCy model (cy_ud_cy_ccg)...")
        import spacy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nlp = spacy.load("cy_ud_cy_ccg")

    whisper_df = pd.read_csv(seg_path, encoding="utf-8-sig").sort_values("segment_start").reset_index(drop=True)
    caption_df = pd.read_csv(captions_csv_path, encoding="utf-8-sig").sort_values("segment_start").reset_index(drop=True)
    mut_df     = pd.read_csv(mutations_csv_path, encoding="utf-8-sig")

    new_cols = {c: [] for c in (
        "caption_available", "caption_segment_match_ratio", "caption_word",
        "caption_word_alignment", "pos_struggle_whisper", "pos_struggle_caption",
        "caption_corroboration", "caption_flag_reason",
    )}
    counts = Counter()

    for _, row in mut_df.iterrows():
        m = TIMESTAMP_RE.search(str(row.get("timestamp", "")))
        if not m:
            for c in new_cols:
                new_cols[c].append(None)
            counts["no_timestamp"] += 1
            continue
        t_start, t_end = float(m.group(1)), float(m.group(2))

        w_seg = covering_segment(whisper_df, t_start, t_end)
        c_seg = covering_segment(caption_df, t_start, t_end)

        # Rule 1, part a: no caption at all for this stretch -- Whisper is
        # the only source, not "the winner" of a comparison that never
        # happened. Tracked separately from an active mismatch below.
        if w_seg is None or c_seg is None:
            new_cols["caption_available"].append(False)
            new_cols["caption_segment_match_ratio"].append(None)
            new_cols["caption_word"].append(None)
            new_cols["caption_word_alignment"].append(None)
            new_cols["pos_struggle_whisper"].append(None)
            new_cols["pos_struggle_caption"].append(None)
            new_cols["caption_corroboration"].append("no_caption")
            new_cols["caption_flag_reason"].append(None)
            counts["no_caption"] += 1
            continue

        whisper_text = str(w_seg["segment_text"])
        caption_text = str(c_seg["segment_text"])
        ratio = segment_match_ratio(whisper_text, caption_text)

        # Rule 1, part b: caption present but substantially different --
        # Whisper wins by default, but this is recorded as an active
        # disagreement, not an absence, since the two are NOT the same
        # thing for auditing corpus reliability later.
        if ratio < MISMATCH_RATIO_THRESHOLD:
            new_cols["caption_available"].append(True)
            new_cols["caption_segment_match_ratio"].append(round(ratio, 3))
            new_cols["caption_word"].append(None)
            new_cols["caption_word_alignment"].append(None)
            new_cols["pos_struggle_whisper"].append(None)
            new_cols["pos_struggle_caption"].append(None)
            new_cols["caption_corroboration"].append("segment_mismatch_too_large")
            new_cols["caption_flag_reason"].append(None)
            counts["segment_mismatch_too_large"] += 1
            continue

        # Rule 2: does Whisper's own text parse worse than the caption's?
        struggle_w = pos_struggle_score(nlp, whisper_text)
        struggle_c = pos_struggle_score(nlp, caption_text)
        flag_reasons = []
        if struggle_w > POS_STRUGGLE_X_RATE and (struggle_w - struggle_c) >= POS_STRUGGLE_MARGIN:
            flag_reasons.append(
                f"Whisper segment POS-struggle={struggle_w:.2f} vs caption={struggle_c:.2f} "
                f"-- caption parses more cleanly, worth a manual reparse"
            )

        # Rule 3: word-level alignment + candidate-set gating at the
        # specific target word this row is about.
        whisper_tokens = [normalize_word(w) for w in whisper_text.split()]
        caption_tokens = [normalize_word(w) for w in caption_text.split()]
        target_norm = normalize_word(row.get("following_word", ""))

        caption_word, alignment = None, None
        try:
            target_index = whisper_tokens.index(target_norm)
        except ValueError:
            target_index = None

        if target_index is None:
            corroboration = "no_alignment"
        else:
            caption_word, alignment = align_target_word(whisper_tokens, caption_tokens, target_index)
            if caption_word is None:
                corroboration = "no_alignment"
            elif caption_word == target_norm:
                corroboration = "confirmed_match"
            else:
                lemma = row.get("lemma") or target_norm
                candidates = generate_mutation_candidates(lemma)
                if caption_word not in candidates:
                    # Not a valid mutated/radical form of this lemma at
                    # all (e.g. caption 'dath' vs lemma 'cath') -- ASR/
                    # caption noise unrelated to mutation, not evidence
                    # either way.
                    corroboration = "caption_word_not_lemma_candidate"
                elif differs_only_in_initial(target_norm, caption_word):
                    # Exactly the mutation-site position. Per the
                    # discussion: don't let the caption overrule Whisper
                    # here -- this IS the phenomenon being measured. But
                    # "don't overrule" isn't the same as "don't
                    # distinguish" -- the four caption/Whisper
                    # combinations at this site carry different evidential
                    # weight and get their own labels rather than being
                    # collapsed into one bucket.
                    caption_mut_type = candidates[caption_word]
                    caption_is_mutated = caption_mut_type != "radical"
                    whisper_mutation_found = str(row.get("mutation_found", "none"))
                    whisper_is_mutated = whisper_mutation_found != "none"

                    if caption_is_mutated and not whisper_is_mutated:
                        # Caption shows the mutated form where Whisper
                        # heard radical -- caption agrees the mutated
                        # form was expected, so Whisper's radical is more
                        # likely genuine erosion, not a mishearing.
                        corroboration = "confirms_erosion"
                    elif not caption_is_mutated and whisper_is_mutated:
                        # Mirror case: caption shows radical where
                        # Whisper found a mutation. Captions normally
                        # normalize TOWARD the mutated form in an
                        # obligatory-mutation environment, so a caption
                        # showing radical here runs against that drift --
                        # worth flagging that Whisper's mutation call may
                        # be spurious, not just noting it and moving on.
                        corroboration = "caption_disputes_mutation"
                        flag_reasons.append(
                            f"Caption shows radical '{caption_word}' where Whisper "
                            f"found a {whisper_mutation_found} mutation -- Whisper's "
                            f"mutation call may be spurious, worth a listen."
                        )
                    elif caption_is_mutated and whisper_is_mutated:
                        # Agreement: both show a mutated form. Not
                        # erosion-relevant (there's no erosion here to
                        # confirm or dispute), but real corroboration --
                        # kept distinct from the radical/radical case
                        # below rather than merged into one "agrees" bucket.
                        corroboration = "confirms_mutation_present"
                    else:
                        # Agreement: both show radical. Genuine support
                        # for Whisper's "no mutation" call, not just an
                        # unexamined default.
                        corroboration = "confirms_no_mutation"
                else:
                    # Mismatch OUTSIDE the mutation site -- a genuine
                    # phoneme-level ASR mishearing, not the thing under
                    # study. Caption takes priority per the rule as
                    # discussed; flagged rather than silently applied.
                    corroboration = "caption_disagrees_non_initial"
                    flag_reasons.append(
                        f"Caption word '{caption_word}' disagrees with Whisper's "
                        f"'{target_norm}' beyond just the mutation site -- likely "
                        f"an ASR mishearing elsewhere in the word, worth a listen."
                    )

        counts[corroboration] += 1
        new_cols["caption_available"].append(True)
        new_cols["caption_segment_match_ratio"].append(round(ratio, 3))
        new_cols["caption_word"].append(caption_word)
        new_cols["caption_word_alignment"].append(alignment)
        new_cols["pos_struggle_whisper"].append(round(struggle_w, 3))
        new_cols["pos_struggle_caption"].append(round(struggle_c, 3))
        new_cols["caption_corroboration"].append(corroboration)
        new_cols["caption_flag_reason"].append("; ".join(flag_reasons) if flag_reasons else None)

    for c, vals in new_cols.items():
        mut_df[c] = vals

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
    args = ap.parse_args()

    if args.corroborate:
        if not args.captions_csv:
            ap.error("--corroborate requires --captions-csv PATH (from a prior fetch run)")
        run_corroboration(Path(args.corroborate), Path(args.captions_csv))
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