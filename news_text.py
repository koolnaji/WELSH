"""
news_text.py
============
Runs the scraped Welsh news articles (Desktop\\NEWS scraper output,
news_corpus/raw/cy/*.txt) through the same tagging and detection as the
YouTube and Siarad data -- as TEXT, so no Whisper and no ASR error.

Why (2026-09-30): professionally edited Welsh from a national broadcaster is
the formal end of the scale.
  1. Noise floor: real erosion in edited narration should be close to zero,
     so what the detectors flag there is mostly detector error -- the
     baseline the spoken rates are compared against.
  2. A same-outlet formality scale: Newyddion S4C article narration ->
     quoted speech in those articles -> Newyddion S4C videos -> Siarad.

Each article becomes up to THREE documents, each its own output folder:
  - narration (source "news-narration"): everything outside quotation marks;
  - quote     (source "news-quote"): interviewees' words inside "..." / “...”
    -- reported speech, lightly edited by the journalist, so a middle point,
    not natural speech;
  - statement (source "news-statement"): quoted text attributed to a
    spokesperson / statement / organisation (STATEMENT_RE) -- written press
    copy, formal, NOT speech; kept apart so it doesn't pull the quote point
    toward narration. Apostrophes (’ ') are never quote marks: only double
    quotes open and close a quote.
Kept apart as sources (not a "speaker" column) so the per-corpus summary
splits them and the speaker-level formality analysis stays people-only.

Sources: only newyddion.s4c.cymru and ycymro.cymru articles (ALLOWED_SOURCES).
BBC articles are skipped -- BBC's terms prohibit dataset creation. Report
counts and rates only; never republish article text.

Word timing: text has none, so each word is given one synthetic second
(document "duration" = its word count). Documents already done on the
current pipeline version (with Cysill) are skipped, like corpus_siarad.py.

Where the articles are found (find_welsh_articles): NEWS_CORPUS_DIR if set,
then Desktop\\news_corpus (the scraper's fixed output folder since
2026-09-30), Desktop\\NEWS\\news_corpus (runs started inside the NEWS folder
before that), the same under OneDrive\\Desktop, and <data folder>\\news_corpus
-- every one that exists, in both the scraper's layouts (raw/cy/ and the
pre-2026-09-27 cy/raw/). An article found in two places is used once.

Usage:
    python news_text.py [corpus folder or raw/cy folder] [--limit N] [--redo]
"""
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from corpus_io import (
    BASE_DIR, CODE_DIR, RUNS_DIR, ensure_dirs, run_stamp, _video_slug, append_output_csv,
    cleanup_incomplete_video_dirs, cleanup_empty_session_dir, pipeline_version,
    is_current_version, set_session_label, session_dir, keep_awake,
    document_folders, publish_document,
)
from corpus_ops import analyze_segments
import run_progress
from output_merge import merge_with_previous, _retire
from cysill_client import TECHIAITH_API_KEY, cysill_status_line, is_cysill_disabled
from mutation_engine import (
    load_spacy, load_lemma_cache, save_lemma_cache, load_bangor_lexicon,
    reset_cysill_circuit_breaker,
)

ALLOWED_SOURCES = ("newyddion.s4c.cymru", "ycymro.cymru")
KINDS = ("narration", "quote", "statement")

# A quote attributed to a spokesperson or organisation is a written press
# statement, not speech -- as formal as the narration or more. First run
# (2026-09-30): in a typical article 300 of 479 quoted words were
# spokesperson statements ("dywedodd llefarydd ar ran Llywodraeth Cymru:"),
# which pulled the "quote" point of the formality scale toward narration.
# The narration of a paragraph decides; quote-only paragraphs that follow
# continue the same speaker (a statement quoted over several paragraphs).
STATEMENT_RE = re.compile(r"\b(llefarydd\w*|llefarwyr|datganiad\w*|ar ran|medden nhw)\b",
                          re.IGNORECASE)
# Written into every news folder. Folders without it are redone -- see
# _already_done: those from before the quote/statement split (v2), and all
# made before contracted "â'r", "gyda'r", "mae'r"... were split for tagging
# (v3, 2026-10-01: the tags after such a word were one word off, and the tagged
# file keeps that, so "Update all results" can't repair these folders).
SPLIT_MARKER = "news_tokens_v3.txt"
OUTPUT_KEYS = ["segments", "words", "lemmas", "pos", "mutations",
               "prep_mutations", "plural_mutations", "numeral_mutations",
               "quantifier_mutations"]
DETECTION_KEYS = OUTPUT_KEYS[4:]

# Sentence ends: . ! ? (optionally followed by a closing bracket), then space.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# Straight quotes toggle; curly quotes open/close. Single quotes are left
# alone -- they are apostrophes in Welsh (mae'r, i'w).
QUOTE_CHARS = {'"': None, "“": True, "”": False, "„": True}


def corpus_roots():
    """Every existing news_corpus folder, most likely first (module docstring)."""
    env = os.environ.get("NEWS_CORPUS_DIR", "").strip()
    # The Desktop that holds this code comes first: cmd may run as another
    # Windows account, whose home isn't where the corpus is (see corpus_io).
    desktops = [CODE_DIR.parent, Path.home() / "Desktop", Path.home() / "OneDrive" / "Desktop"]
    candidates = ([Path(env)] if env else []) + \
        [d / "news_corpus" for d in desktops] + \
        [d / "NEWS" / "news_corpus" for d in desktops] + \
        [BASE_DIR / "news_corpus"]
    roots = []
    for c in candidates:
        if c.is_dir() and c.resolve() not in [r.resolve() for r in roots]:
            roots.append(c)
    return roots


def find_welsh_articles(folder=None):
    """(Welsh article files, folders searched). `folder` may be a corpus root
    or a folder of .txt files; without it, every corpus root is searched.
    Both layouts are read; an article file name seen twice is kept once."""
    roots = [Path(folder)] if folder else corpus_roots()
    found = {}
    for root in roots:
        subdirs = [d for d in (root / "raw" / "cy", root / "cy" / "raw") if d.is_dir()]
        for d in subdirs or [root]:
            for p in d.glob("*.txt"):
                if "__" in p.name:          # articles are domain__slug__hash.txt
                    found.setdefault(p.name, p)
    return sorted(found.values(), key=lambda p: p.name), roots


def _article_domain(path):
    return path.name.split("__")[0]


def _doc_id(path):
    """Short, unique id: the scraper's 8-hex hash at the end of the file name."""
    return path.stem.rsplit("__", 1)[-1]


def split_quotes(paragraph):
    """[(is_quote, text), ...] for one paragraph. The state resets at every
    paragraph: a quote continued into the next paragraph opens with its own
    mark there (news convention), and an unclosed quote runs to the end of
    its paragraph."""
    spans, buf, inside = [], [], False
    for ch in paragraph:
        if ch in QUOTE_CHARS:
            if buf:
                spans.append((inside, "".join(buf)))
                buf = []
            want = QUOTE_CHARS[ch]
            inside = (not inside) if want is None else want
        else:
            buf.append(ch)
    if buf:
        spans.append((inside, "".join(buf)))
    return [(q, t.strip()) for q, t in spans if t.strip()]


def read_article(path):
    """{"narration": [...], "quote": [...], "statement": [...]} sentences for
    one article (see STATEMENT_RE for quote vs statement)."""
    out = {k: [] for k in KINDS}
    text = path.read_text(encoding="utf-8", errors="replace")
    statement_mode = False
    for paragraph in text.splitlines():
        spans = split_quotes(paragraph.strip())
        narration = " ".join(t for is_quote, t in spans if not is_quote)
        if narration:
            statement_mode = bool(STATEMENT_RE.search(narration))
        for is_quote, span in spans:
            kind = ("statement" if statement_mode else "quote") if is_quote else "narration"
            for sentence in SENTENCE_END_RE.split(span):
                if sentence.strip():
                    out[kind].append(sentence.strip())
    return out


def build_segments(sentences):
    """Whisper-shaped segments, one per sentence, one synthetic second per word."""
    segments, t = [], 0.0
    for sentence in sentences:
        tokens = sentence.split()
        if not tokens:
            continue
        words = []
        for i, tok in enumerate(tokens):
            if i == len(tokens) - 1 and not tok.endswith((".", "!", "?", ",", ":", ";")):
                tok += "."
            words.append(SimpleNamespace(word=tok, start=round(t, 3), end=round(t + 0.9, 3),
                                         probability=1.0, lang=None, name_part=False))
            t += 1.0
        segments.append(SimpleNamespace(start=words[0].start, end=words[-1].end,
                                        text=sentence, words=words,
                                        no_speech_prob=0.0, avg_logprob=0.0))
    return segments


def _meta(path, kind):
    doc = _doc_id(path)
    return {"id": f"news_{kind}_{doc}", "title": f"News {kind} {doc}",
            "url": f"news:{path.stem}#{kind}", "source": f"news-{kind}",
            "article_file": path.name, "domain": _article_domain(path)}


def _already_done(path, kind):
    """Like corpus_siarad._already_done: output with Cysill tags (when a key
    is set) -- on the CURRENT version, or on any version when the folder
    keeps its tagged words (tagged_*.json.gz): "Update all results" (4 -> y)
    re-runs detection from those without Cysill, so re-tagging such a
    document after a detection-only change wastes hours (2026-10-06: a
    3 -> n restarted after a code change would have re-tagged ~1,000
    finished documents). After a change to TAGGING, use --redo."""
    for pos_csv in (p for d in document_folders(f"News_{kind}_{_doc_id(path)}")
                    for p in d.glob("pos_*.csv")):
        if not (pos_csv.parent / SPLIT_MARKER).exists():
            continue    # made by an older news_text (see SPLIT_MARKER)
        try:
            pos = pd.read_csv(pos_csv, usecols=lambda c: c in ("cysill_pos", "pipeline_version"),
                              dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            continue
        if pos.empty or "pipeline_version" not in pos.columns:
            continue
        cysill_share = (pos["cysill_pos"] != "").mean() if "cysill_pos" in pos.columns else 0.0
        # Many news chunks are resolved locally (lexicon/English), so a lower
        # Cysill share than Siarad's is normal; 0 means Cysill never answered.
        reusable = is_current_version(pos["pipeline_version"].iloc[0]) or \
            any(pos_csv.parent.glob("tagged_*.json.gz"))
        if reusable and (not TECHIAITH_API_KEY or cysill_share > 0):
            return True
    return False


def retire_stale(path, kind):
    """Moves this article's `kind` folders to runs/_deleted/ -- for a kind the
    article no longer has. After the statement split, an article whose quotes
    were all spokesperson statements has no "quote" document, and its old
    mixed quote folder (news-927) would otherwise stay and be counted."""
    n = 0
    for folder in document_folders(f"News_{kind}_{_doc_id(path)}"):
        if folder.is_dir():
            _retire(folder, folder, f"no {kind} text in the article any more")
            n += 1
    return n


def process_document(path, kind, sentences, stamp):
    """Tags and analyses one document. False = Cysill hit its limit (stop)."""
    segments = build_segments(sentences)
    if not segments:
        return True
    meta = _meta(path, kind)
    duration = float(segments[-1].end)
    meta["_coverage_window"] = [0.0, duration]
    vpaths = None
    try:
        vpaths = _video_slug(meta, stamp)
        results = analyze_segments(
            segments, meta, video_duration_seconds=duration, language="cy",
            language_probability=1.0, checkpoint_key=meta["url"],
            step=run_progress.step,   # names the phase on the per-document bar
            tagged_cache_path=vpaths["tagged"])
        if TECHIAITH_API_KEY and is_cysill_disabled():
            print(f"  ⏸ {path.name} ({kind}): Cysill hit its limit -- not saved.")
            cleanup_incomplete_video_dirs(vpaths, video_label=path.name)
            return False
        outputs = dict(zip(OUTPUT_KEYS, results))
        header_flags = [True] * len(OUTPUT_KEYS)
        for idx, key in enumerate(OUTPUT_KEYS):
            if outputs[key]:
                append_output_csv(pd.DataFrame(outputs[key]), vpaths[key], header_flags, idx)
        (vpaths["segments"].parent / SPLIT_MARKER).write_text(
            "quotes/statements split (news_text.STATEMENT_RE); contracted words "
            "split for tagging (mutation_engine.expand_whisper_tokens)\n",
            encoding="utf-8")
        counts = ", ".join(f"{k.replace('_mutations', '')}={len(outputs[k])}" for k in DETECTION_KEYS)
        print(f"  {_doc_id(path)} {kind}: {len(outputs['words'])} words -- {counts}")
        try:
            # A text document is always replaced WHOLE: its "seconds" are word
            # positions, so an earlier, longer version (quotes before the
            # statement split) would otherwise keep its tail beyond this
            # version's last word.
            merge_with_previous(vpaths, meta["url"], [0.0, 1e9])
        except Exception as e:
            print(f"  ⚠️ Merge with earlier output failed ({e}) -- saved on its own.")
        vpaths = publish_document(vpaths, meta)   # -> runs/news/<kind>/<article>/
    except Exception as e:
        print(f"  💥 {path.name} ({kind}): {e}")
        cleanup_incomplete_video_dirs(vpaths, video_label=path.name)
    return True


def main(argv):
    redo = "--redo" in argv
    argv = [a for a in argv if a != "--redo"]
    limit = None
    if "--limit" in argv:
        at = argv.index("--limit")
        try:
            limit = int(argv[at + 1])
        except (IndexError, ValueError):
            print("--limit needs a number, e.g. --limit 20")
            return 1
        argv = argv[:at] + argv[at + 2:]
    everything, roots = find_welsh_articles(argv[0] if argv else None)
    files = [p for p in everything if _article_domain(p) in ALLOWED_SOURCES]
    skipped_other = len(everything) - len(files)
    if not files:
        where = ", ".join(str(r) for r in roots) or "no news_corpus folder exists"
        print(f"No Newyddion S4C / Y Cymro articles found (looked in: {where}). "
              f"Set NEWS_CORPUS_DIR to the scraper's news_corpus folder if it's elsewhere.")
        return 1
    print("Reading articles from: " + ", ".join(str(r) for r in roots))
    print(f"News articles: {len(files)} usable"
          + (f" ({skipped_other} from other sites skipped -- e.g. BBC, whose terms rule it out)"
             if skipped_other else ""))

    jobs, retired = [], 0
    document_folders("", refresh=True)   # index runs/ once for _already_done
    for path in run_progress.files(files, "Checking what's done", unit="articles"):
        article = read_article(path)
        for kind in KINDS:
            if not article[kind]:
                retired += retire_stale(path, kind)
            elif redo or not _already_done(path, kind):
                jobs.append((path, kind, article[kind]))
    if retired:
        print(f"Moved {retired} outdated folder(s) to runs/_deleted/ "
              f"(documents the articles no longer have).")
    if not jobs:
        print("Nothing left to process -- every article is done on this pipeline version.")
        return 0
    if limit is not None:
        jobs = jobs[:limit]

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
    set_session_label(stamp, label=f"news-{len(jobs)}")
    print(f"Processing {len(jobs)} document(s) (narration and quotes counted separately) "
          f"into runs/news/ (staged in runs/_sessions/{session_dir(stamp).name}/ "
          f"while each is worked on)")
    keep_awake()
    run_progress.start_run(len(jobs), "documents")
    try:
        for n, (path, kind, sentences) in enumerate(jobs):
            run_progress.item(f"{_doc_id(path)} {kind}")
            if not process_document(path, kind, sentences, stamp):
                print(f"Stopped: Cysill's limit was reached. {len(jobs) - n} document(s) left -- "
                      f"run this again later; finished ones are skipped.")
                break
            run_progress.item_done()
    finally:
        run_progress.end_run()
        keep_awake(False)
        save_lemma_cache()
        cleanup_empty_session_dir(stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
