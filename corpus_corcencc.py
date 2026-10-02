"""
corpus_corcencc.py
===================
Runs the SPOKEN part of CorCenCC (Corpws Cenedlaethol Cymraeg Cyfoes, the
National Corpus of Contemporary Welsh, v1.0.0, Cardiff University; CC-BY-SA;
cite Knight et al. (2020), doi:10.17035/d.2020.0119878310) through the same
tagging and detection as Siarad and the YouTube videos, via
corpus_ops.analyze_segments() -- the human transcript, retagged with Cysill
like every other source, so all corpora are tagged the same way. CorCenCC's
own CyTag tags (POS, lemma, mutation) are only used to recognise speaker
codes, foreign words and proper nouns in the transcript.

Usage:
    python corpus_corcencc.py <CorCenCC_corpus folder> [--limit N] [--redo]
    (the folder holding corpus_data.txt and the .tsv metadata tables; it is
    searched recursively, so the unzipped download folder works too.
    Recordings already done on the current pipeline version, with Cysill,
    are skipped unless --redo is given, so a run stopped by Cysill's hourly
    limit is simply restarted.)

Input, as checked against the v1.0.0 download (2026-10-02):
  - corpus_data.txt: one token per line, tab-separated -- file, token,
    sentence no., position, lemma, basic POS, enriched POS, mutation,
    semantic tag. Spoken recordings are the "lla_" files (1,331);
  - a sentence starts with its speaker code ("S1", basic POS "Anon") when
    the speaker changes; sentences without one continue the last speaker;
  - transcription marks, all dropped: [saib] (pause), [aneglur]
    (unclear), [-] (cut off), "+" (interruption), "< pesychu >" (events,
    with everything between the angle brackets), and [=] ... [/=] (a
    repeated stretch -- the words inside are dropped, the way Siarad's
    retraced material is);
  - enriched POS "Gwest" (gair estron, foreign word) -> English; everything
    else is left to the spelling heuristic, as for untagged Siarad words;
  - clitics are their own tokens ("ti 'n siarad", "'di", "'r"): written
    out as yn / wedi / yr (Siarad's spelling) when CyTag tags them as the
    particle / article, otherwise glued back on ("ro" + "'n" = ro'n);
  - anonymisation placeholders (basic POS "Anon": lleoliad, enwb1, cyfenw2)
    are kept as capitalised name parts, which no branch measures;
  - the first word of every sentence is capitalised: lower-cased unless
    CyTag calls it a proper noun (enriched POS "Ep..."), since the mutation
    branch treats a capitalised word as a name.

Speakers and metadata: CorCenCC anonymises speakers as S1, S2... per
recording, and lists the recording's contributors (contrib_links.tsv) in no
particular order, so a speaker code can't be matched to a contributor. Their
background is therefore recorded PER RECORDING in speakers_*.csv:
learner_status = all_l1 / all_learner / mixed / unknown (contributor.tsv
"learner": 1 = yes, 2 = no), the contributors' birth years and self-rated
ability, plus genre/context (categorisation.tsv -> taxonomy.tsv), scripted
(spoken.tsv) and place. Only all_l1 vs all_learner recordings can be
compared on learner status.

Word timing: CorCenCC has no audio times. Each word gets 0.4 s on a running
clock and each sentence is one segment -- positions, not measurements, kept
unique so every detection row maps back to exactly one speaker.

Output, per recording, in runs/<stamp>/CorCenCC_<file>/: the same CSVs a
Siarad conversation produces (speakers_*.csv and utterances_*.csv included).

Not in _DETECTION_SOURCES (like news_text.py): after a change here that
alters the word stream, bump READER_MARKER and every recording is redone.
"""
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from corpus_io import (
    RUNS_DIR, ensure_dirs, run_stamp, _video_slug, append_output_csv,
    cleanup_incomplete_video_dirs, cleanup_empty_session_dir, pipeline_version,
    is_current_version, set_session_label, session_dir, has_lexicon_features,
)
from corpus_ops import analyze_segments
from corpus_siarad import OUTPUT_KEYS, DETECTION_KEYS, _row_start, _saved_run_quality
from output_merge import merge_with_previous
from cysill_client import TECHIAITH_API_KEY, cysill_status_line, is_cysill_disabled
from mutation_engine import (
    load_spacy, load_lemma_cache, save_lemma_cache, load_bangor_lexicon,
    reset_cysill_circuit_breaker,
)

CORPUS = "corcencc"
SPOKEN_PREFIX = "lla_"
SPEAKER_RE = re.compile(r"^S\d+$")
SENTENCE_END = {".", "?", "!"}
DROP_MARKS = {"'", '"', "-", "(", ")", "…", "...", ":", ";", "/"}
REPEAT_OPEN, REPEAT_CLOSE = "[=]", "[/=]"
# A pause is dropped without a trace, as Siarad's "(.)" is. Every other
# bracket mark -- [aneglur] (unclear), [-] (cut off) -- and "+" (interrupted)
# or an <event> is a gap: the words either side were not said together, so
# the word before it gets a "," and no branch pairs across it ("i [aneglur]
# plant" is not "i plant").
TRANSPARENT_MARKS = {"[saib]"}
GAP_MARKS = {"+"}
# Split-off clitics, written out only when CyTag tags them as the particle /
# preposition / article / perfect marker. Otherwise they are glued back onto
# the word before: CorCenCC splits "ro'n i" (I was) into ro + 'n + i with 'n
# tagged as a pronoun ending (Rha), and writing it out as "yn" produced a
# fake "yn i" / "yn nhw" -- conjugated-preposition "erosions" (2026-10-02).
CLITICS = {"'n": ("yn", {"U", "Ar"}), "'r": ("yr", {"YFB"}), "'di": ("wedi", {"U"})}
FOREIGN_POS = "Gwest"
# CyTag basic POS of speaker codes AND of the anonymisation placeholders:
# lleoliad (place), enwb / enwg (female / male name), cyfenw (surname)...
# The transcribers write a placeholder unmutated whatever was said, so "i
# lleoliad" (originally e.g. "i Gaerdydd") looked like erosion: 1,363
# mutation rows in 349 recordings (2026-10-02). A placeholder is kept as a
# capitalised name part -- a blocker that no branch measures (names are out:
# decisions 2026-09-29 / 10-01).
ANON_POS = "Anon"
WORD_SECONDS = 0.4
# Written into a recording's folder by this reader; bump the name whenever a
# change here alters the word stream (see _already_done).
READER_MARKER = "corcencc_reader_v2.txt"


# ========================= METADATA =========================
def _read_tsv(folder, name):
    path = next(Path(folder).rglob(name), None)
    if path is None:
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_metadata(folder):
    """{recording id -> dict of recording-level metadata}."""
    taxonomy = {r["id"]: (r["category"], r["tag"]) for r in _read_tsv(folder, "taxonomy.tsv")}
    tags = {}
    for r in _read_tsv(folder, "categorisation.tsv"):
        cat, tag = taxonomy.get(r["taxonomy_id"], (None, None))
        if cat:
            tags.setdefault(r["contribution_id"], {}).setdefault(cat, []).append(tag)
    spoken = {r["contribution_id"]: r for r in _read_tsv(folder, "spoken.tsv")}
    people = {r["contributor_id"]: r for r in _read_tsv(folder, "contributor.tsv")}
    links = {}
    for r in _read_tsv(folder, "contrib_links.tsv"):
        links.setdefault(r["contribution_id"], []).append(r["contributor_id"])

    meta = {}
    for cid in set(spoken) | {c for c in links if c.startswith(SPOKEN_PREFIX)}:
        individuals = [people[p] for p in links.get(cid, [])
                       if p in people and people[p].get("contributor_type") == "1"]
        learner = {p.get("learner") for p in individuals}
        if not individuals or learner - {"1", "2"}:
            status = "unknown"
        elif learner == {"2"}:
            status = "all_l1"
        elif learner == {"1"}:
            status = "all_learner"
        else:
            status = "mixed"
        years = sorted(int(p["contributor_year_of_birth"]) for p in individuals
                       if p.get("contributor_year_of_birth", "").isdigit()
                       and int(p["contributor_year_of_birth"]) > 1900)
        t = tags.get(cid, {})
        meta[cid] = {
            "genre": "; ".join(t.get("genre", [])) or None,
            "context": "; ".join(t.get("context", [])) or None,
            "topic": "; ".join(t.get("topic", [])) or None,
            "scripted": spoken.get(cid, {}).get("scripted"),
            "place": spoken.get(cid, {}).get("place") or None,
            "contributors": len(individuals),
            "learner_status": status,
            "learners": sum(p.get("learner") == "1" for p in individuals),
            "l1_speakers": sum(p.get("learner") == "2" for p in individuals),
            "birth_year_min": years[0] if years else None,
            "birth_year_max": years[-1] if years else None,
            "abilities": ",".join(sorted(p.get("contributor_ability", "") for p in individuals)) or None,
        }
    return meta


# ========================= TRANSCRIPT =========================
def iter_recordings(corpus_data):
    """Yields (recording id, [token rows]) for every spoken recording, in
    file order. Each token row: (token, sentence, basic POS, enriched POS)."""
    current, rows, seen = None, [], set()
    with open(corpus_data, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith(SPOKEN_PREFIX):
                if current is not None:
                    yield current, rows
                    seen.add(current)
                    current, rows = None, []
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            rid = parts[0][:-4] if parts[0].endswith(".txt") else parts[0]
            if rid != current:
                if current is not None:
                    yield current, rows
                    seen.add(current)
                if rid in seen:
                    print(f"  ⚠️ {rid} appears twice in corpus_data.txt -- second part skipped")
                    current, rows = None, []
                    continue
                current, rows = rid, []
            rows.append((parts[1], parts[2], parts[5], parts[6]))
    if current is not None:
        yield current, rows


def build_sentences(rows):
    """[(speaker, sentence no., [(word, lang, is_proper_noun, name_part)])]
    with the transcription marks removed (see the module docstring and the
    constants above)."""
    sentences, speaker = [], None
    by_sentence = {}
    order = []
    for token, sent, basic, enriched in rows:
        if sent not in by_sentence:
            by_sentence[sent] = []
            order.append(sent)
        by_sentence[sent].append((token, basic, enriched))

    def gap(words):
        if words and not words[-1][0].endswith(","):
            words[-1] = (words[-1][0] + ",",) + words[-1][1:]

    for sent in order:
        words, in_event, in_repeat = [], False, False
        for position, (token, basic, enriched) in enumerate(by_sentence[sent]):
            if basic == ANON_POS and SPEAKER_RE.match(token) and position == 0:
                speaker = token
                continue
            if token == "<":
                in_event = True
                gap(words)
                continue
            if token == ">":
                in_event = False
                continue
            if token == REPEAT_OPEN:
                in_repeat = True
                continue
            if token == REPEAT_CLOSE:
                in_repeat = False
                continue
            if in_event or in_repeat or token in TRANSPARENT_MARKS:
                continue
            if token in GAP_MARKS or (token.startswith("[") and token.endswith("]")):
                gap(words)
                continue
            if token == "," and words:
                gap(words)
                continue
            if token in SENTENCE_END or token in DROP_MARKS or basic == "Atd":
                continue
            if basic == ANON_POS:      # placeholder: lleoliad, enwb1, cyfenw2...
                words.append((token.capitalize(), None, True, True))
                continue
            # enriched POS may list alternatives ("Gwest | Ep | Epb")
            alternatives = {a.strip() for a in enriched.split("|")}
            proper = any(a.startswith("Ep") for a in alternatives)
            lang = "eng" if alternatives == {FOREIGN_POS} else None
            clitic = CLITICS.get(token.lower())
            if clitic:
                full, as_word = clitic
                if {b.strip() for b in basic.split("|")} & as_word or not words:
                    token = full
                else:              # "ro" + "'n" -> "ro'n"
                    prev = words[-1]
                    words[-1] = (prev[0].rstrip(",") + token,) + prev[1:]
                    continue
            words.append((token, lang, proper, False))
        if words:
            first, lang, proper, name_part = words[0]
            if not proper and first[:1].isupper() and not first.isupper():
                words[0] = (first[:1].lower() + first[1:], lang, proper, name_part)
            sentences.append((speaker or "S?", sent, words))
    return sentences


def build_segments(sentences):
    """Segment/word objects shaped like faster-whisper's (see corpus_siarad),
    one segment per sentence, plus {word start -> speaker}."""
    segments, seg_speakers, start_to_speaker = [], [], {}
    clock = 0.0
    for speaker, _sent, words in sentences:
        start = round(clock, 3)
        word_objs = []
        for i, (text, lang, _proper, name_part) in enumerate(words):
            w_start = round(clock, 3)
            clock += WORD_SECONDS
            if i == len(words) - 1 and not text.endswith(","):
                text += "."
            word_objs.append(SimpleNamespace(
                word=text, start=w_start, end=round(clock - 0.001, 3), lang=lang,
                name_part=name_part, probability=1.0))
            start_to_speaker[w_start] = speaker
        text = " ".join(w[0] for w in words)
        segments.append(SimpleNamespace(start=start, end=round(clock, 3), text=text,
                                        words=word_objs, no_speech_prob=0.0, avg_logprob=0.0,
                                        speaker=speaker))
        seg_speakers.append(speaker)
        clock += WORD_SECONDS   # a gap between sentences
    return segments, seg_speakers, start_to_speaker


# ========================= PROCESSING =========================
def process_recording(rid, rows, stamp, meta_info):
    sentences = build_sentences(rows)
    segments, seg_speakers, start_to_speaker = build_segments(sentences)
    if not segments:
        return True
    meta = {"id": f"{CORPUS}_{rid}", "title": f"CorCenCC {rid}",
            "url": f"{CORPUS}:{rid}", "source": CORPUS}
    duration = segments[-1].end
    meta["_coverage_window"] = [0.0, float(duration)]

    vpaths = None
    try:
        vpaths = _video_slug(meta, stamp)
        results = analyze_segments(
            segments, meta, video_duration_seconds=duration, language="cy",
            language_probability=1.0, checkpoint_key=meta["url"],
            tagged_cache_path=vpaths["tagged"])
        if TECHIAITH_API_KEY and is_cysill_disabled():
            print(f"  ⏸ {rid}: Cysill hit its hourly limit during this recording -- not saved.")
            cleanup_incomplete_video_dirs(vpaths, video_label=rid)
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
        for code in dict.fromkeys(seg_speakers):
            spoken = [w for spk, _s, ws in sentences if spk == code for w in ws]
            speaker_rows.append({
                "conversation": rid, "speaker": code, "age": None, "sex": None, "role": None,
                "notes": None, "word_count": len(spoken),
                "english_tagged_words": sum(1 for w in spoken if w[1] == "eng"),
                "other_language_tagged_words": sum(1 for w in spoken if w[1] == "eng"),
                "languages": "cym,eng", "situation": meta_info.get("context"), "date": None,
                **meta_info,
            })
        pd.DataFrame(speaker_rows).to_csv(out_dir / f"speakers_{folder_name}.csv",
                                          index=False, encoding="utf-8-sig", quoting=1)
        pd.DataFrame([{"conversation": rid, "speaker": spk, "sentence": sent,
                       "main_tier": " ".join(w[0] for w in ws),
                       "clean_text": " ".join(w[0] for w in ws), "gls": None, "eng": None}
                      for spk, sent, ws in sentences]
                     ).to_csv(out_dir / f"utterances_{folder_name}.csv",
                              index=False, encoding="utf-8-sig", quoting=1)

        (out_dir / READER_MARKER).write_text(
            "Read by corpus_corcencc.py with the 2026-10-02 fixes (placeholders, "
            "clitics, gaps).\n", encoding="utf-8")
        counts = ", ".join(f"{k.replace('_mutations', '')}={len(outputs[k])}" for k in DETECTION_KEYS)
        print(f"  {rid}: {len(segments)} sentences, {len(outputs['words'])} words -- {counts}")
        try:
            merge_with_previous(vpaths, meta["url"], meta["_coverage_window"])
        except Exception as e:
            print(f"  ⚠️ Merge with earlier output failed ({e}) -- this run's output is "
                  f"saved on its own; earlier folders untouched.")
    except Exception as e:
        print(f"  💥 {rid}: {e}")
        cleanup_incomplete_video_dirs(vpaths, video_label=rid)
    return True


def _already_done(rid):
    """Same test as corpus_siarad._already_done: output on the current
    pipeline version, with the lexicon, and Cysill-tagged when a key is set
    -- plus READER_MARKER, since this file isn't in the version hash: output
    from an older reader is redone even after "Update all results" has
    re-stamped it with the current version."""
    for segments_csv in RUNS_DIR.glob(f"*/CorCenCC_{rid}/segments_*.csv"):
        if not (segments_csv.parent / READER_MARKER).exists():
            continue
        pos_csv = next(segments_csv.parent.glob("pos_*.csv"), None)
        if pos_csv is None or not has_lexicon_features(pos_csv):
            continue
        version, cysill_share = _saved_run_quality(pos_csv)
        if is_current_version(version) and (not TECHIAITH_API_KEY or cysill_share >= 0.9):
            return True
    return False


def main(argv):
    redo = "--redo" in argv
    argv = [a for a in argv if a != "--redo"]
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
        print("Usage: python corpus_corcencc.py <CorCenCC_corpus folder> [--limit N] [--redo]")
        return 1
    folder = Path(argv[0])
    corpus_data = next((p for p in folder.rglob("corpus_data.txt") if "__MACOSX" not in p.parts), None)
    if corpus_data is None:
        print(f"No corpus_data.txt under {folder} -- unzip corpus_data.txt.zip there first.")
        return 1

    print("Reading CorCenCC metadata...")
    metadata = load_metadata(corpus_data.parent)
    learner_counts = pd.Series([m["learner_status"] for m in metadata.values()]).value_counts()
    print(f"  {len(metadata)} spoken recordings; learner status: "
          + ", ".join(f"{k} {v}" for k, v in learner_counts.items()))

    done = set() if redo else {rid for rid in metadata if _already_done(rid)}
    if done:
        print(f"Skipping {len(done)} recording(s) already done on this pipeline version "
              f"with Cysill (add --redo to process them again).")
    todo = len(metadata) - len(done)
    if limit is not None:
        todo = min(todo, limit)
    if todo <= 0:
        print("Nothing left to process.")
        return 0

    ensure_dirs()
    load_lemma_cache()
    print("Loading Welsh dependency parser...")
    load_spacy()
    if not load_bangor_lexicon():
        raise SystemExit("Stopped: the Bangor lexicon is required (see the message above).")
    print(cysill_status_line())
    print(f"Pipeline version: {pipeline_version()}")
    reset_cysill_circuit_breaker()

    stamp = run_stamp()
    set_session_label(stamp, label=f"{CORPUS}-{todo}")
    print(f"Processing {todo} CorCenCC recording(s) into runs/{session_dir(stamp).name}/")
    processed = 0
    try:
        for rid, rows in iter_recordings(corpus_data):
            if rid in done:
                continue
            if processed >= todo:
                break
            if not process_recording(rid, rows, stamp, metadata.get(rid, {})):
                print(f"Stopped: Cysill's hourly limit was reached after {processed} "
                      f"recording(s) -- run the same command again in about an hour; "
                      f"finished ones are skipped.")
                break
            processed += 1
            if processed % 25 == 0:
                save_lemma_cache()   # 1,331 recordings: don't lose hours of lookups to a crash
    finally:
        save_lemma_cache()
        cleanup_empty_session_dir(stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
