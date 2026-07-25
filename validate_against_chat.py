r"""
validate_against_chat.py -- compares Whisper's own transcript for a Bangor
Siarad corpus recording against that recording's original CHAT-format
(.cha) transcript, using the exact same whole-video word-alignment engine
fetch_captions.py already uses for YouTube caption corroboration (see
align_whole_video()'s docstring there for the full alignment rationale) --
just pointed at a different independent transcript source.

Why this exists: unlike the podcast sources, the Bangor Siarad corpus
comes with a human-made, word-level ground-truth transcript for every
recording. That makes it the one place in this project where you can
actually MEASURE how well Whisper handles genuinely spontaneous,
overlapping, multi-speaker Welsh, instead of assuming it degrades
gracefully the way it does on scripted/produced audio. Run this on a
handful of recordings BEFORE trusting bulk erosion numbers from any
casual-register source -- if Whisper's word-level agreement with the
human transcript is much worse here than you'd expect from the rest of
your corpus, that's a real signal worth understanding before it quietly
distorts your casual-register erosion rate.

NOT part of the automated pipeline. Standalone, same convention as
fetch_captions.py / manual_editing.py -- but DOES import a few pieces
directly from fetch_captions.py (normalize_word, the whole-video
alignment engine) rather than duplicating them, since this is genuinely
the same job -- "align an independent transcript against Whisper's
word stream" -- just fed CHAT text instead of VTT captions.

*** IMPORTANT CAVEAT ***
The CHAT-cleaning pass below (clean_chat_utterance) handles the commonly
-documented CLAN/CHAT annotation codes (retracing/error brackets, pause
markers, paralinguistic events, code-switch @-suffixes, unintelligible-
speech placeholders, TalkBank sound-linked timing bullets). It has NOT
been validated against an actual downloaded Bangor Siarad .cha file --
this corpus may use annotation conventions not covered here. Run with
--dump-clean on one or two files FIRST and eyeball the cleaned output
before trusting the alignment numbers at scale.

Usage:
    # Step 1 -- ALWAYS do this first on a new file:
    python validate_against_chat.py recording.cha --dump-clean

    # Step 2 -- once the cleaned text looks right, compare against your
    # pipeline's own output for the same recording (the words_*.csv from
    # running that file through option 1):
    python validate_against_chat.py recording.cha \
        WELSH_ANALYSIS_DIR/transcriptions/<stamp>/<slug>/words_<stamp>_<slug>.csv
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from fetch_captions import (
    normalize_word, build_whisper_token_stream, align_whole_video,
)

# ===================== CHAT annotation cleanup =====================
# TalkBank sound-linked timing bullet: a control character (0x15) wraps
# "start_end" in milliseconds at the point in the line it corresponds to.
# Some renderings/exports use a printable bullet symbol instead of the
# raw control char -- both are matched here. Adjust this regex first if
# --dump-clean shows raw bullets surviving into the cleaned text.
CHAT_BULLET_RE = re.compile(r'[\x15\u2022]?(\d+)_(\d+)[\x15\u2022]?')

BRACKET_ANNOTATION_RE = re.compile(r'\[[^\]]*\]')   # retraces, errors, overlaps, glosses
PARALING_EVENT_RE     = re.compile(r'&=\S+')        # &=laughs, &=coughs, etc -- non-verbal
FRAGMENT_MARKER_RE    = re.compile(r'&-(\S+)')      # &-um -- false-start fragment, keep the sound
SPECIAL_FORM_SUFFIX_RE= re.compile(r"(\w[\w'-]*)@[\w:]+")  # word@s:eng -> word
TRAILING_OFF_RE       = re.compile(r'\+[./?]+')     # +... +//. trailing-off / interruption markers
UNINTELLIGIBLE_RE     = re.compile(r'\b(?:xxx|yyy|www)\b')
PAREN_PAUSE_RE        = re.compile(r'\(\.+\)')      # (.) (..) (...) standalone pause markers
PAREN_ELISION_RE      = re.compile(r'[()]')         # (h)ave -- drop parens, keep the letters


def clean_chat_utterance(text):
    """
    Strips CHAT/CLAN annotation codes from a main-tier utterance line,
    leaving (approximately) the words actually spoken. See the module
    docstring's caveat -- validate against real files before trusting
    this at scale on a corpus-specific annotation style.
    """
    text = BRACKET_ANNOTATION_RE.sub(' ', text)
    text = PARALING_EVENT_RE.sub(' ', text)
    text = FRAGMENT_MARKER_RE.sub(r'\1', text)
    text = SPECIAL_FORM_SUFFIX_RE.sub(r'\1', text)
    text = TRAILING_OFF_RE.sub(' ', text)
    text = UNINTELLIGIBLE_RE.sub(' ', text)
    text = PAREN_PAUSE_RE.sub(' ', text)
    text = PAREN_ELISION_RE.sub('', text)
    return text


def parse_cha(path):
    """
    Parses a CHAT (.cha) transcript into a list of
    {speaker, text, start, end} dicts -- start/end in seconds, taken from
    TalkBank's sound-linked timing bullets when present, else None.

    Only main tier lines (start with '*SPEAKER:') carry spoken text.
    Dependent tiers (%mor, %gra, %com, ...) and header/metadata lines
    (@Begin, @Participants, ...) are skipped entirely. A soft-wrapped
    continuation line (doesn't start with *, %, or @) is joined onto the
    preceding tier line, since CHAT allows a single utterance to wrap
    across multiple physical lines.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    joined = []
    for line in lines:
        if line.startswith(("*", "%", "@")):
            joined.append(line)
        elif joined:
            joined[-1] += " " + line.strip()
        # else: stray line before any tier has started -- ignore

    utterances = []
    for line in joined:
        if not line.startswith("*") or ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        speaker = speaker[1:].strip()  # drop leading '*'

        start = end = None
        m = CHAT_BULLET_RE.search(text)
        if m:
            start = int(m.group(1)) / 1000.0
            end   = int(m.group(2)) / 1000.0
            text  = CHAT_BULLET_RE.sub('', text)

        text = clean_chat_utterance(text).strip()
        if text:
            utterances.append({"speaker": speaker, "text": text,
                                "start": start, "end": end})
    return utterances


def build_chat_token_stream(utterances):
    """
    Whole-transcript, chronologically-ordered (word, approx_timestamp)
    stream from parsed CHAT utterances -- same shape as
    fetch_captions.build_caption_token_stream(), so it plugs straight
    into align_whole_video(). Utterances without their own timing bullet
    inherit the nearest PRECEDING timed utterance's end time as an
    approximation, rather than None -- leaving timestamps as None would
    disable the timing-plausibility safeguard in align_whole_video()
    entirely for any stretch of untimed lines.
    """
    tokens, times = [], []
    last_time = 0.0
    for u in utterances:
        t = u["start"] if u["start"] is not None else last_time
        for w in u["text"].split():
            tokens.append(normalize_word(w))
            times.append(t)
        if u["end"] is not None:
            last_time = u["end"]
    return tokens, times


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cha_path", help="path to the .cha transcript")
    ap.add_argument("words_csv", nargs="?",
                     help="path to the matching words_*.csv from your pipeline "
                          "(option 1 output for this same recording)")
    ap.add_argument("--dump-clean", action="store_true",
                     help="just print the cleaned CHAT utterances and exit -- "
                          "ALWAYS run this first on a new file to sanity check "
                          "the parser before trusting the alignment numbers")
    args = ap.parse_args()

    utterances = parse_cha(args.cha_path)
    if not utterances:
        print("No main-tier utterances parsed -- check that lines start with "
              "'*SPEAKER:' as CHAT expects, and that the file read correctly.")
        sys.exit(1)

    if args.dump_clean:
        for u in utterances[:200]:
            t = f"[{u['start']:.1f}s]" if u['start'] is not None else "[no timing]"
            print(f"{t} *{u['speaker']}: {u['text']}")
        print(f"\n{len(utterances)} utterance(s) total "
              f"(showing up to 200 above). Check for leftover CHAT codes "
              f"(brackets, @ suffixes, bullets) before trusting this parser.")
        return

    if not args.words_csv:
        ap.error("words_csv is required unless --dump-clean is given")

    words_df = pd.read_csv(args.words_csv, encoding="utf-8-sig")
    whisper_tokens, whisper_times, _ = build_whisper_token_stream(words_df)
    chat_tokens, chat_times = build_chat_token_stream(utterances)

    if not whisper_tokens or not chat_tokens:
        print("Empty token stream on one side -- nothing to compare.")
        sys.exit(1)

    mapping = align_whole_video(whisper_tokens, whisper_times, chat_tokens, chat_times)

    n = len(whisper_tokens)
    tag_counts = Counter(v["tag"] for v in mapping.values())
    equal_and_timed_ok = sum(
        1 for v in mapping.values()
        if v["tag"] == "equal" and v["timing_ok"] is not False
    )

    print(f"\nWhisper words: {n}   CHAT words: {len(chat_tokens)}")
    print(f"Word-level agreement (exact match, timing-plausible): "
          f"{equal_and_timed_ok}/{n} ({equal_and_timed_ok / n:.1%})")
    print("\nAlignment breakdown:")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:<28} {count:>6}  ({count / n:.1%})")

    print("\nSample disagreements (first 20):")
    shown = 0
    for w_idx in sorted(mapping.keys()):
        if shown >= 20:
            break
        evidence = mapping[w_idx]
        if evidence["tag"] == "equal" and evidence["timing_ok"] is not False:
            continue
        w_word = whisper_tokens[w_idx]
        c_word = evidence.get("caption_word")
        print(f"  whisper[{w_idx}]={w_word!r} (t={whisper_times[w_idx]:.1f}s)  "
              f"<-> chat={c_word!r}  tag={evidence['tag']}")
        shown += 1

    if shown == 0:
        print("  (none in the first pass through the alignment -- looks clean)")


if __name__ == "__main__":
    main()
