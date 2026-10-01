"""
corpus_siarad.py
=================
Runs the Bangor Siarad corpus (CHAT .cha transcripts of informal Welsh-English
conversation, recorded 2005-08 at speakers' homes/workplaces with no researcher
present) through the same tagging and detection as the YouTube pipeline --
but on the HUMAN transcript, so no Whisper errors -- via
corpus_ops.analyze_segments().

Usage:
    python corpus_siarad.py <file.cha | folder of .cha files> [--limit N] [--redo]
                            [--corpus siarad|patagonia]
    (a folder is searched recursively; conversations already done on the
    current pipeline version, with Cysill, are skipped unless --redo is given,
    so a stopped run -- or one after a detection fix -- is simply restarted;
    --limit N processes only the next N not yet done; --corpus defaults to
    "patagonia" when the path contains "patagonia", else "siarad")

Also reads the Bangor Patagonia corpus (Welsh-Spanish, Argentina, 2009; same
team and CHAT conventions, GPLv3; cite Deuchar & Webb-Davies (2011), The
Bangor Patagonia Corpus, doi:10.21415/T55K5C). The only difference that
matters here is the second language: each file's @Languages header names
them, the FIRST is the default for untagged words (Patagonia documentation;
some files may be Spanish-default), and a bare "@s" means "the file's other
language" -- Spanish there, English in Siarad.

Output, per conversation, in runs/<stamp>/Siarad_<file>/ (Patagonia_<file>/):
  - the same segments/words/lemmas/pos/mutations/prep/plural/numeral/quantifier CSVs a
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
    a bare word@s (the form the TalkBank release uses) = the utterance's
    other language, i.e. English in a Welsh utterance;
    untagged = decided by spelling (the pipeline's code-switch heuristic),
    except in an utterance marked [- eng] (English);
  - dropped: pauses, retraced material (<...> [/], [//]), events and
    fragments (&=laugh, &m), unintelligible xxx/yyy/www, 0-prefixed omitted
    words, linkers/terminators, bracket codes ([!], [= ...]);
  - "[?]" (transcriber unsure): the words are kept but get probability 0.5,
    so no detection branch scores them (see UNCERTAIN_RE).

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
    RUNS_DIR, ensure_dirs, run_stamp, _video_slug, append_output_csv,
    cleanup_incomplete_video_dirs, cleanup_empty_session_dir, pipeline_version,
    is_current_version,
    set_session_label, session_dir,
)
from corpus_ops import analyze_segments
from corpus_io import has_lexicon_features  # unversioned
from output_merge import merge_with_previous
from cysill_client import TECHIAITH_API_KEY, cysill_status_line, is_cysill_disabled
from mutation_engine import (
    load_spacy, load_lemma_cache, save_lemma_cache, load_bangor_lexicon,
    reset_cysill_circuit_breaker,
)

OUTPUT_KEYS = ["segments", "words", "lemmas", "pos", "mutations",
               "prep_mutations", "plural_mutations", "numeral_mutations",
               "quantifier_mutations"]
DETECTION_KEYS = ("mutations", "prep_mutations", "plural_mutations", "numeral_mutations",
                  "quantifier_mutations")

CORPORA = ("siarad", "patagonia")
# Used when a file has no @Languages header: Siarad's pair.
DEFAULT_LANGUAGES = ("cym", "eng")
# "[- spa]" / "[- eng]" / "[- cym]": the whole utterance is in that language.
UTTERANCE_LANG_RE = re.compile(r"\[-\s*([a-z]{3})\]")

BULLET_RE    = re.compile(r"\x15(\d+)_(\d+)\x15")
# "@s:eng" / "@s:cym&eng" (explicit), or a BARE "@s" = "the other language of
# this utterance" -- which is what the TalkBank release actually uses
# ("beans@s", "ting@s"). Only the explicit form was recognised at first
# (built against the GitHub beta), so nearly every English word in the real
# files was read as Welsh: davies1.cha came out with 20 English-tagged words
# where Siarad averages ~4% English (2026-09-26).
LANG_TAG_RE  = re.compile(r"@s(?::([a-z&+]+))?$")
PAUSE_RE     = re.compile(r"\(\.+\)|\(\d+[:.]?\d*\.?\d*\)")
RETRACE_RE   = re.compile(r"(<[^<>]*>|\S+)\s*\[/[/?-]*\]")
REPLACE_RE   = re.compile(r"(<[^<>]*>|\S+)\s*\[:\s*([^\]]*)\]")
# every bracket code except retracing [/...] and replacement [: ...]
OTHER_CODE_RE = re.compile(r"\[(?![/:])[^\]]*\]")
# a <...> group NOT tied to a retrace/replacement code -- unwrapped first,
# so nested groups ("<<o(eddw)n i> [?] just fel> [//]") leave a flat group
# the retrace pattern can remove whole
UNWRAP_RE    = re.compile(r"<([^<>]*)>(?!\s*\[[/:])")
# "word [?]" / "<words> [?]": the transcriber wasn't sure what was said
# ("<(y)na rai te> [?]" in fusser19.cha produced a rhai erosion, 2026-09-28).
# Those words get probability UNCERTAIN_PROBABILITY -- below the 0.65 floor
# every branch applies to triggers and targets, so they are never scored,
# mutated or not, but still count as words.
UNCERTAIN_RE = re.compile(r"(<[^<>]*>|[^\s<>]+)\s*\[\?\]")
UNCERTAIN_MARK = "‡"
UNCERTAIN_PROBABILITY = 0.5
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

    # "languages": @Languages, first = default. "pair": the conversation's two
    # languages from the @ID lines ("cym, spa"). The TalkBank Patagonia release
    # lists three in @Languages ("cym, eng, spa"), but its bare "@s" words are
    # Spanish ("no@s", "treinta@s"), so the "other language" comes from here.
    header = {"participants": {}, "comments": [], "situation": None,
              "date": None, "duration_seconds": None, "languages": DEFAULT_LANGUAGES,
              "pair": None}
    utterances, current = [], None
    for line in lines:
        if line.startswith("@Languages:"):
            langs = tuple(l.strip() for l in line[len("@Languages:"):].split(",") if l.strip())
            if langs:
                header["languages"] = langs
        elif line.startswith("@ID:"):
            fields = line.split("\t", 1)[-1].split("|")
            id_langs = tuple(l.strip() for l in fields[0].split(",") if l.strip())
            if header["pair"] is None and len(id_langs) >= 2:
                header["pair"] = id_langs[:2]
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


def clean_main_tier(main, languages=DEFAULT_LANGUAGES, pair=None):
    """Returns (start_s, end_s, words) where words is a list of (text, lang,
    uncertain, name_part). A "," in the transcript is attached to the preceding word so
    the pipeline's clause-boundary logic sees it. `languages` is the file's
    @Languages list (the first is the default for untagged words); `pair` is
    the conversation's two languages from @ID, if known."""
    bullet = BULLET_RE.search(main)
    start = int(bullet.group(1)) / 1000 if bullet else None
    end = int(bullet.group(2)) / 1000 if bullet else None

    s = BULLET_RE.sub(" ", main)
    precode = UTTERANCE_LANG_RE.search(s)
    utterance_lang = precode.group(1) if precode else languages[0]
    # the "other language" a bare "@s" switches to
    pair = pair or (tuple(languages) + DEFAULT_LANGUAGES)[:2]
    other_lang = pair[1] if utterance_lang == pair[0] else pair[0]
    s = UNCERTAIN_RE.sub(
        lambda m: " ".join(t + UNCERTAIN_MARK for t in m.group(1).strip("<>").split()), s)
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
        uncertain = UNCERTAIN_MARK in tok
        tok = tok.replace(UNCERTAIN_MARK, "")
        if tok == "," and words:
            words[-1] = (words[-1][0] + ",",) + words[-1][1:]
            continue
        if tok in DROP_TOKENS or tok.startswith(("&", "+", "#", "0")):
            continue
        # Untagged words are NOT assumed Welsh: transcribers leave some
        # English-spelled words untagged ("boy", "condoms" in davies13.cha,
        # whose transcriber tags ~350 other English words), and forcing them
        # to Welsh made them numeral/rhai/mutation targets. None = decided by
        # spelling, the same heuristic as YouTube data -- the orthography
        # criterion (decision 2026-09-26). Explicit tags still win. In a
        # non-Welsh utterance (Spanish-default Patagonia file, or "[- eng]")
        # untagged words are that language.
        lang = None if utterance_lang == "cym" else utterance_lang
        m = LANG_TAG_RE.search(tok)
        if m:
            tag = m.group(1)
            if tag is None:     # bare "@s": switch to the utterance's other language
                lang = other_lang
            elif "&" in tag:    # in both dictionaries ("cym&eng", "cym&spa")
                lang = "und"
            elif "+" in tag:    # mixed morphology, treated as Welsh
                lang = "mixed"
            else:               # "cym", "eng", "spa", ...
                lang = tag
            tok = tok[:m.start()]
        tok = tok.split("@")[0].strip("“”\"")
        tok = tok.replace("(", "").replace(")", "").replace("+", "")
        for ch in PROSODY_CHARS:
            tok = tok.replace(ch, "")
        # In a mixed-morphology word the underscore joins a stem to a Welsh
        # ending (hammer_o@s:eng+cym = "hammero"); splitting it left a fake
        # preposition "o" that the prep branch scored as "o fo" (audit
        # 2026-09-29). Elsewhere it joins the words of a name or compound.
        if lang == "mixed":
            tok = tok.replace("_", "")
        parts = [p for p in tok.split("_") if p]
        # A capitalised underscore word is a multi-word name (Pen_y_Bont_Fawr,
        # Pen_y_groes); every part is flagged, since a lower-case part after
        # "y" ("groes") would otherwise pass as a common noun -- the mutation
        # branch doesn't measure place names (decision 2026-09-29).
        name_part = len(parts) > 1 and any(p[:1].isupper() for p in parts)
        for part in parts:
            words.append((part, lang, uncertain, name_part))
    return start, end, words


def build_segments(utterances, languages=DEFAULT_LANGUAGES, pair=None):
    """Segment/word objects shaped like faster-whisper's, one segment per
    non-empty utterance, plus {word start -> speaker} for row attribution."""
    segments, seg_speakers, start_to_speaker, used = [], [], {}, set()
    prev_end = 0.0
    for utt in utterances:
        start, end, words = clean_main_tier(utt["main"], languages, pair)
        utt["clean"] = " ".join(w[0] for w in words)
        if not words:
            continue
        if start is None:
            start = end = prev_end
        span = max(end - start, 0.001 * len(words))
        word_objs = []
        for i, (text, lang, uncertain, name_part) in enumerate(words):
            w_start = round(start + span * i / len(words), 3)
            while w_start in used:
                w_start = round(w_start + 0.001, 3)
            used.add(w_start)
            w_end = round(max(start + span * (i + 1) / len(words), w_start + 0.001), 3)
            if i == len(words) - 1 and not text.endswith(","):
                text += "."
            word_objs.append(SimpleNamespace(
                word=text, start=w_start, end=w_end, lang=lang, name_part=name_part,
                probability=UNCERTAIN_PROBABILITY if uncertain else 1.0))
            start_to_speaker[w_start] = utt["speaker"]
        segments.append(SimpleNamespace(start=start, end=start + span, text=utt["clean"],
                                        words=word_objs, no_speech_prob=0.0, avg_logprob=0.0,
                                        speaker=utt["speaker"]))
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


def process_file(path, stamp, corpus="siarad"):
    path = Path(path)
    header, utterances = read_chat(path)
    languages, pair = header["languages"], header["pair"]
    segments, seg_speakers, start_to_speaker = build_segments(utterances, languages, pair)
    meta = {"id": f"{corpus}_{path.stem}", "title": f"{corpus.capitalize()} {path.stem}",
            "url": f"{corpus}:{path.stem}", "source": corpus}
    duration = header["duration_seconds"] or (segments[-1].end if segments else 0.0)
    meta["_coverage_window"] = [0.0, float(duration or 0.0)]   # the whole conversation

    vpaths = None
    try:
        vpaths = _video_slug(meta, stamp)
        results = analyze_segments(
            segments, meta, video_duration_seconds=duration, language="cy",
            language_probability=1.0, checkpoint_key=meta["url"],
            tagged_cache_path=vpaths["tagged"])
        # Cysill's hourly limit tripped somewhere in this file (tagging OR the
        # lemma lookups during detection): its rows would be tagged differently
        # from every other file's, so it isn't saved at all -- see main().
        if TECHIAITH_API_KEY and is_cysill_disabled():
            print(f"  ⏸ {path.name}: Cysill hit its hourly limit during this file -- not saved.")
            cleanup_incomplete_video_dirs(vpaths, video_label=path.name)
            return False
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
            word_langs = [lang for u in utterances if u["speaker"] == code
                          for _, lang, _, _ in clean_main_tier(u["main"], languages, pair)[2]]
            speaker_rows.append({
                "conversation": path.stem, "speaker": code, "age": info["age"],
                "sex": info["sex"], "role": info["role"],
                "notes": " | ".join(info["notes"]) or None,
                "word_count": len(words),
                "english_tagged_words": sum(1 for l in word_langs if l == "eng"),
                # every word tagged as a non-Welsh language (Spanish in Patagonia)
                "other_language_tagged_words": sum(
                    1 for l in word_langs if l not in (None, "cym", "mixed", "und")),
                "languages": ",".join(languages),
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
        # One folder per conversation: an earlier run's folder is absorbed and
        # moved to runs/_deleted/ (output_merge.py). The whole conversation is
        # covered, so nothing of the earlier run is kept.
        try:
            merge_with_previous(vpaths, meta["url"], meta["_coverage_window"])
        except Exception as e:
            print(f"  ⚠️ Merge with earlier output failed ({e}) -- this run's output is "
                  f"saved on its own; earlier folders untouched.")
    except Exception as e:
        print(f"  💥 {path.name}: {e}")
        cleanup_incomplete_video_dirs(vpaths, video_label=path.name)
    return True


def _saved_run_quality(pos_csv):
    """(pipeline_version, share of words with a Cysill POS tag) of a saved run."""
    import csv
    tagged = total = 0
    version = None
    with open(pos_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            tagged += bool(row.get("cysill_pos"))
            version = version or row.get("pipeline_version")
    return version, (tagged / total if total else 0.0)


def _already_done(path, corpus="siarad"):
    """True if some earlier run already wrote this conversation's output ON
    THE CURRENT PIPELINE VERSION and, when a key is set, WITH Cysill tags.
    The whole corpus takes several sittings, so a plain rerun picks up where
    it stopped -- and after a detection fix, the same plain rerun redoes
    everything still on an older version (no --redo needed, which would
    restart from file 1 each time the hourly limit stops a run). A
    conversation saved without Cysill (fusser15-18, 2026-09-27) isn't done."""
    for segments_csv in RUNS_DIR.glob(f"*/{corpus.capitalize()}_{path.stem}/segments_*.csv"):
        pos_csv = next(segments_csv.parent.glob("pos_*.csv"), None)
        if pos_csv is None:
            continue
        version, cysill_share = _saved_run_quality(pos_csv)
        if not has_lexicon_features(pos_csv):  # unversioned
            continue  # unversioned
        if is_current_version(version) and (not TECHIAITH_API_KEY or cysill_share >= 0.9):
            return True
    return False


def main(argv):
    redo = "--redo" in argv
    argv = [a for a in argv if a != "--redo"]
    corpus = None
    if "--corpus" in argv:
        at = argv.index("--corpus")
        corpus = argv[at + 1].lower() if at + 1 < len(argv) else None
        if corpus not in CORPORA:
            print(f"--corpus needs one of: {', '.join(CORPORA)}")
            return 1
        argv = argv[:at] + argv[at + 2:]
    limit = None
    if "--limit" in argv:
        at = argv.index("--limit")
        try:
            limit = int(argv[at + 1])
        except (IndexError, ValueError):
            print("--limit needs a number, e.g. --limit 5")
            return 1
        argv = argv[:at] + argv[at + 2:]
    if len(argv) != 1:
        print("Usage: python corpus_siarad.py <file.cha | folder of .cha files> "
              "[--limit N] [--redo] [--corpus siarad|patagonia]")
        return 1
    target = Path(argv[0])
    if corpus is None:
        corpus = "patagonia" if "patagonia" in str(target).lower() else "siarad"
    print(f"Corpus: {corpus}")
    # rglob: the TalkBank zip unpacks into a Siarad/ subfolder
    files = sorted(target.rglob("*.cha")) if target.is_dir() else [target]
    if not files or not all(f.exists() for f in files):
        print(f"No .cha files found at {target}")
        return 1
    if not redo:
        done = [f for f in files if _already_done(f, corpus)]
        if done:
            print(f"Skipping {len(done)} conversation(s) already done on this pipeline "
                  f"version with Cysill (add --redo to process them again).")
        files = [f for f in files if f not in done]
        if not files:
            print("Nothing left to process.")
            return 0
    if limit is not None:
        files = files[:limit]

    ensure_dirs()
    load_lemma_cache()
    print("Loading Welsh dependency parser...")
    load_spacy()
    # Detection depends on the lexicon, so a run without it isn't comparable
    # with the others -- stop rather than write degraded rows (2026-09-30).
    if not load_bangor_lexicon():
        raise SystemExit("Stopped: the Bangor lexicon is required (see the message above).")
    print(cysill_status_line())
    print(f"Pipeline version: {pipeline_version()}")
    reset_cysill_circuit_breaker()

    stamp = run_stamp()
    set_session_label(stamp, label=f"{corpus}-{len(files)}")
    print(f"Processing {len(files)} {corpus.capitalize()} file(s) into runs/{session_dir(stamp).name}/")
    try:
        for n, f in enumerate(files):
            if not process_file(f, stamp, corpus):
                print(f"Stopped: Cysill's hourly limit was reached. {len(files) - n} "
                      f"conversation(s) not processed -- run the same command again "
                      f"in about an hour; finished ones are skipped.")
                break
    finally:
        save_lemma_cache()
        cleanup_empty_session_dir(stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
