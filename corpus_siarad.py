"""
corpus_siarad.py
=================
Runs the Bangor Siarad corpus (CHAT .cha transcripts of informal Welsh-English
conversation, recorded 2005-08 at speakers' homes/workplaces with no researcher
present) through the same tagging and detection as the YouTube pipeline --
but on the HUMAN transcript, so no Whisper errors -- via
corpus_ops.analyze_segments().

Usage:
    python corpus_siarad.py <file.cha | folder of .cha files>

Output, per conversation, in runs/<stamp>/Siarad_<file>/:
  - the same segments/words/lemmas/pos/mutations/prep/plural/numeral CSVs a
    video produces, every row with an added "speaker" column;
  - speakers_*.csv: code, age, sex, role, the transcript's per-speaker
    @Comment notes (e.g. where they grew up), and word / English-tagged
    word counts;
  - utterances_*.csv: every utterance's speaker, times, raw main tier,
    cleaned text, and %gls/%eng tiers, kept for later gloss-based checks.

Conventions (confirmed against a real Siarad file, davies1.cha):
  - "*SPK:" main tier, ending in a terminator and a media bullet
    \\x15start_end\\x15 in ms; lines starting with a tab continue the tier;
  - "(y)n", "(gy)da", "o(eddw)n": parentheses mark unpronounced material --
    the FULL word is used, which also settles "o'n" = oeddwn vs "o yn";
  - word@s:eng = English, @s:cym&eng = in both dictionaries (left to the
    usual heuristic), @s:eng+cym / @s:cym+eng = mixed morphology (Welsh);
    untagged = Welsh, except in an utterance marked [- eng];
  - dropped: pauses, retraced material (<...> [/], [//]), events and
    fragments (&=laugh, &m), unintelligible xxx/yyy/www, 0-prefixed omitted
    words, linkers/terminators, bracket codes ([?], [!], [= ...]).

Word timing: CHAT times whole utterances. Word times are spread evenly across
each utterance's recorded span -- interpolated, not measured -- and kept
unique, so every detection row maps back to exactly one speaker.

Licence: Siarad is GPLv3; cite Deuchar, Webb-Davies & Donnelly (2009), The
Bangor Siarad Corpus, doi:10.21415/T5088V.
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from corpus_io import (
    ensure_dirs, run_stamp, _video_slug, append_output_csv,
    cleanup_incomplete_video_dirs, cleanup_empty_session_dir,
)
from corpus_ops import analyze_segments
from cysill_client import cysill_status_line
from mutation_engine import (
    load_spacy, load_lemma_cache, save_lemma_cache, load_bangor_lexicon,
    reset_cysill_circuit_breaker,
)

OUTPUT_KEYS = ["segments", "words", "lemmas", "pos", "mutations",
               "prep_mutations", "plural_mutations", "numeral_mutations"]
DETECTION_KEYS = ("mutations", "prep_mutations", "plural_mutations", "numeral_mutations")

BULLET_RE    = re.compile(r"\x15(\d+)_(\d+)\x15")
LANG_TAG_RE  = re.compile(r"@s:([a-z&+]+)$")
PAUSE_RE     = re.compile(r"\(\.+\)|\(\d+[:.]?\d*\.?\d*\)")
RETRACE_RE   = re.compile(r"(<[^<>]*>|\S+)\s*\[/[/?-]*\]")
REPLACE_RE   = re.compile(r"(<[^<>]*>|\S+)\s*\[:\s*([^\]]*)\]")
# every bracket code except retracing [/...] and replacement [: ...]
OTHER_CODE_RE = re.compile(r"\[(?![/:])[^\]]*\]")
# a <...> group NOT tied to a retrace/replacement code -- unwrapped first,
# so nested groups ("<<o(eddw)n i> [?] just fel> [//]") leave a flat group
# the retrace pattern can remove whole
UNWRAP_RE    = re.compile(r"<([^<>]*)>(?!\s*\[[/:])")
DROP_TOKENS  = {".", "?", "!", ",", "xxx", "yyy", "www", "xx", "yy"}
PROSODY_CHARS = "ˈˌ:^↑↓≈‡„"


def _parse_age(field):
    m = re.match(r"\s*(\d+);", field or "")
    return int(m.group(1)) if m else None


def _parse_duration(value):
    m = re.match(r"\s*(\d+):(\d+):(\d+)", value or "")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def read_chat(path):
    """Returns (header, utterances). header["participants"] maps speaker code
    -> {speaker, age, sex, role, notes}; each utterance is {speaker, main,
    gls, eng}."""
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.startswith("\t") and lines:
            lines[-1] += " " + raw.strip()
        else:
            lines.append(raw)

    header = {"participants": {}, "comments": [], "situation": None,
              "date": None, "duration_seconds": None}
    utterances, current = [], None
    for line in lines:
        if line.startswith("@ID:"):
            fields = line.split("\t", 1)[-1].split("|")
            if len(fields) > 4:
                code = fields[2].strip()
                header["participants"][code] = {
                    "speaker": code, "age": _parse_age(fields[3]),
                    "sex": fields[4].strip() or None,
                    "role": fields[7].strip() if len(fields) > 7 and fields[7].strip() else None,
                    "notes": [],
                }
        elif line.startswith("@Comment:"):
            header["comments"].append(line.split("\t", 1)[-1].strip())
        elif line.startswith("@Situation:"):
            header["situation"] = line.split("\t", 1)[-1].strip()
        elif line.startswith("@Date:"):
            header["date"] = line.split("\t", 1)[-1].strip()
        elif line.startswith("@Time Duration:"):
            header["duration_seconds"] = _parse_duration(line.split("\t", 1)[-1])
        elif line.startswith("*") and ":" in line:
            spk, body = line[1:].split(":", 1)
            current = {"speaker": spk.strip(), "main": body.strip(), "gls": None, "eng": None}
            utterances.append(current)
        elif line.startswith("%") and ":" in line and current is not None:
            tier, body = line[1:].split(":", 1)
            if tier in ("gls", "eng"):
                current[tier] = body.strip()

    for comment in header["comments"]:
        first = comment.split()[0] if comment.split() else ""
        if first in header["participants"]:
            header["participants"][first]["notes"].append(comment)
    return header, utterances


def clean_main_tier(main):
    """Returns (start_s, end_s, words) where words is a list of (text, lang).
    A "," in the transcript is attached to the preceding word so the
    pipeline's clause-boundary logic sees it."""
    bullet = BULLET_RE.search(main)
    start = int(bullet.group(1)) / 1000 if bullet else None
    end = int(bullet.group(2)) / 1000 if bullet else None

    s = BULLET_RE.sub(" ", main)
    utterance_english = "[- eng]" in s
    s = OTHER_CODE_RE.sub(" ", s)
    while True:
        unwrapped = UNWRAP_RE.sub(r" \1 ", s)
        if unwrapped == s:
            break
        s = unwrapped
    s = RETRACE_RE.sub(" ", s)
    s = REPLACE_RE.sub(lambda m: f" {m.group(2)} ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = PAUSE_RE.sub(" ", s)
    s = s.replace("<", " ").replace(">", " ")

    words = []
    for tok in s.split():
        if tok == "," and words:
            words[-1] = (words[-1][0] + ",", words[-1][1])
            continue
        if tok in DROP_TOKENS or tok.startswith(("&", "+", "#", "0")):
            continue
        lang = "eng" if utterance_english else "cym"
        m = LANG_TAG_RE.search(tok)
        if m:
            tag = m.group(1)
            lang = "eng" if tag == "eng" else "cym" if tag == "cym" else \
                   "und" if "&" in tag else "mixed"
            tok = tok[:m.start()]
        tok = tok.split("@")[0].strip("“”\"")
        tok = tok.replace("(", "").replace(")", "").replace("+", "")
        for ch in PROSODY_CHARS:
            tok = tok.replace(ch, "")
        for part in tok.split("_"):
            if part:
                words.append((part, lang))
    return start, end, words


def build_segments(utterances):
    """Segment/word objects shaped like faster-whisper's, one segment per
    non-empty utterance, plus {word start -> speaker} for row attribution."""
    segments, seg_speakers, start_to_speaker, used = [], [], {}, set()
    prev_end = 0.0
    for utt in utterances:
        start, end, words = clean_main_tier(utt["main"])
        utt["clean"] = " ".join(w for w, _ in words)
        if not words:
            continue
        if start is None:
            start = end = prev_end
        span = max(end - start, 0.001 * len(words))
        word_objs = []
        for i, (text, lang) in enumerate(words):
            w_start = round(start + span * i / len(words), 3)
            while w_start in used:
                w_start = round(w_start + 0.001, 3)
            used.add(w_start)
            w_end = round(max(start + span * (i + 1) / len(words), w_start + 0.001), 3)
            if i == len(words) - 1 and not text.endswith(","):
                text += "."
            word_objs.append(SimpleNamespace(word=text, start=w_start, end=w_end,
                                             probability=1.0, lang=lang))
            start_to_speaker[w_start] = utt["speaker"]
        segments.append(SimpleNamespace(start=start, end=start + span, text=utt["clean"],
                                        words=word_objs, no_speech_prob=0.0, avg_logprob=0.0))
        seg_speakers.append(utt["speaker"])
        prev_end = start + span
    return segments, seg_speakers, start_to_speaker


def _row_start(row):
    """Detection rows carry "<start>s - <end>s"; word/lemma/pos rows carry
    word_start. Either way it's the start time of a word this module made
    unique, so it identifies the utterance -- and speaker -- exactly."""
    try:
        if row.get("timestamp"):
            return round(float(str(row["timestamp"]).split("s - ")[0]), 3)
        if row.get("word_start") is not None:
            return round(float(row["word_start"]), 3)
    except ValueError:
        pass
    return None


def process_file(path, stamp):
    path = Path(path)
    header, utterances = read_chat(path)
    segments, seg_speakers, start_to_speaker = build_segments(utterances)
    meta = {"id": f"siarad_{path.stem}", "title": f"Siarad {path.stem}",
            "url": f"siarad:{path.stem}", "source": "siarad"}
    duration = header["duration_seconds"] or (segments[-1].end if segments else 0.0)

    vpaths = None
    try:
        results = analyze_segments(
            segments, meta, video_duration_seconds=duration, language="cy",
            language_probability=1.0, checkpoint_key=meta["url"])
        outputs = dict(zip(OUTPUT_KEYS, results))

        for row, spk in zip(outputs["segments"], seg_speakers):
            row["speaker"] = spk
        unmapped = 0
        for key in OUTPUT_KEYS[1:]:
            for row in outputs[key]:
                row["speaker"] = start_to_speaker.get(_row_start(row))
                if row["speaker"] is None and key in DETECTION_KEYS:
                    unmapped += 1
        if unmapped:
            print(f"  ⚠️  {unmapped} detection row(s) could not be mapped to a speaker")

        vpaths = _video_slug(meta, stamp)
        header_flags = [True] * len(OUTPUT_KEYS)
        for idx, key in enumerate(OUTPUT_KEYS):
            if outputs[key]:
                append_output_csv(pd.DataFrame(outputs[key]), vpaths[key], header_flags, idx)

        folder_name = vpaths["segments"].name[len("segments_"):-len(".csv")]
        out_dir = vpaths["segments"].parent
        speaker_rows = []
        for code, info in header["participants"].items():
            words = [w for u in utterances if u["speaker"] == code
                     for w in (u.get("clean") or "").split()]
            n_eng = sum(1 for u in utterances if u["speaker"] == code
                        for _, lang in clean_main_tier(u["main"])[2] if lang == "eng")
            speaker_rows.append({
                "conversation": path.stem, "speaker": code, "age": info["age"],
                "sex": info["sex"], "role": info["role"],
                "notes": " | ".join(info["notes"]) or None,
                "word_count": len(words), "english_tagged_words": n_eng,
                "situation": header["situation"], "date": header["date"],
            })
        pd.DataFrame(speaker_rows).to_csv(out_dir / f"speakers_{folder_name}.csv",
                                          index=False, encoding="utf-8-sig", quoting=1)
        pd.DataFrame([{"conversation": path.stem, "speaker": u["speaker"],
                       "main_tier": u["main"], "clean_text": u.get("clean"),
                       "gls": u["gls"], "eng": u["eng"]} for u in utterances]
                     ).to_csv(out_dir / f"utterances_{folder_name}.csv",
                              index=False, encoding="utf-8-sig", quoting=1)

        counts = ", ".join(f"{k.replace('_mutations', '')}={len(outputs[k])}" for k in DETECTION_KEYS)
        print(f"  {path.name}: {len(segments)} utterances, {len(outputs['words'])} words -- {counts}")
    except Exception as e:
        print(f"  💥 {path.name}: {e}")
        cleanup_incomplete_video_dirs(vpaths, video_label=path.name)


def main(argv):
    if len(argv) != 1:
        print("Usage: python corpus_siarad.py <file.cha | folder of .cha files>")
        return 1
    target = Path(argv[0])
    files = sorted(target.glob("*.cha")) if target.is_dir() else [target]
    if not files or not all(f.exists() for f in files):
        print(f"No .cha files found at {target}")
        return 1

    ensure_dirs()
    load_lemma_cache()
    print("Loading Welsh dependency parser...")
    load_spacy()
    load_bangor_lexicon()
    print(cysill_status_line())
    reset_cysill_circuit_breaker()

    stamp = run_stamp()
    print(f"Processing {len(files)} Siarad file(s) into runs/{stamp}/")
    try:
        for f in files:
            process_file(f, stamp)
    finally:
        save_lemma_cache()
        cleanup_empty_session_dir(stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
