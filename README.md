# Welsh mutation analysis pipeline

An interactive pipeline for transcribing Welsh audio, detecting initial
consonant mutations, checking candidate findings against YouTube captions,
reviewing results by hand, and producing corpus-wide figures.

## What this actually does, in one paragraph

Point it at Welsh-language audio (a local MP3, or a YouTube channel you've
added to a queue). It transcribes with Whisper, tags every word with two
independent part-of-speech taggers (the Cysill API and a local spaCy
parser), and uses that to work out where a mutation-triggering word (like
"yn", "ei", "mae") is followed by a word that *should* be mutated -- then
checks whether it actually was, in what the speaker said. Where YouTube
captions exist, it cross-checks its own findings against them. Everything
lands in CSVs you can review by hand, or summarize into corpus-wide
figures.

## Setup

1. Install Python 3.11 or newer and FFmpeg, ensuring both are available on
   your command line. Stick to Python 3.11-3.12 if you can -- pandas,
   matplotlib, and seaborn (and the wider scientific-Python stack under
   them) can lag behind the newest Python release by months before
   publishing prebuilt wheels for it, and installing on a Python version
   without wheels yet either fails outright or silently falls back to
   building from source, which needs a C compiler toolchain most machines
   don't have set up.
2. Create and activate a virtual environment.
3. Install the Python dependencies with `python -m pip install -r requirements.txt`.
4. Install the Welsh spaCy model `cy_ud_cy_ccg` compatible with your spaCy
   version. The pipeline continues without it, but parser-based validation
   (dependency-aware mutation rules, caption corroboration's POS check)
   will be unavailable.
5. Optionally, download Techiaith's Bangor lexicon
   (`lecsicon_cc0.zip` from
   [techiaith/lecsicon-cymraeg-bangor](https://github.com/techiaith/lecsicon-cymraeg-bangor),
   CC0), unzip it, and place `lecsicon_cc0.txt` somewhere on disk (or set
   `BANGOR_LEXICON_PATH` to point at it -- see **Environment variables**
   below). This is a local, offline lookup used to resolve lemmas and
   unambiguous POS/mutation/gender info without hitting the Cysill API,
   cutting down on the 429 rate-limiting that endpoint runs into on real
   corpus-sized runs. The pipeline continues without it, falling back to
   Cysill/spaCy/simplemma exactly as before.
6. Copy `.env.example` to your preferred environment-variable setup and
   fill in what you need (see **Environment variables** below). Only
   `WELSH_ANALYSIS_DIR` affects core functionality; the rest are optional.
7. Run `python welsh_pipeline.py`. This opens the main interactive menu
   (see **Workflow** below).

### Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `WELSH_ANALYSIS_DIR` | No | Where all output lives (audio, transcripts, mutations, summaries, queue/cache files). Defaults to `~/welsh_analysis` if unset. |
| `WELSH_LEMMATIZER` | No | API key for the Cysill (techiaith.cymru) POS/lemmatizer service. Without it, the pipeline falls back to spaCy + local heuristics only -- it still works, just with one fewer independent tagger cross-checking every word. |
| `BANGOR_LEXICON_PATH` | No | Path to the downloaded `lecsicon_cc0.txt` file (see **Setup** step 5). Without it, lemma/POS/mutation-type/gender lookups go straight to Cysill/spaCy/simplemma, same as before this existed. Defaults to `bangor_lexicon/lecsicon_cc0.txt` (relative to wherever you run the pipeline from) if unset. |
| `GMAIL_SENDER` / `GMAIL_APP_PASSWORD` / `NOTIFY_RECIPIENT` | No | Enables an HTML completion-email summary (run stats, per-video results, erosion breakdown) after Queue & Processing -> b (Process queue) or Testing -> b (Analyze local MP3 files, when saved) finish. Needs a Gmail account with 2-Step Verification and an App Password (Google Account -> Security -> 2-Step Verification -> App passwords) -- not your normal Gmail password. `NOTIFY_RECIPIENT` defaults to `GMAIL_SENDER` (i.e. emails yourself) if unset. Leave all three blank to disable notifications entirely; the pipeline runs exactly the same either way, it just skips the email at the end. |

Never commit real values for any of these -- keep them in your actual
environment/shell profile/`.env`, not in source files.

## The pieces

`welsh_pipeline.py` is the only thing you run directly. Everything else is
either a module it imports, or a standalone companion tool you can also
run on its own from the command line.

**Core pipeline (imported by `welsh_pipeline.py`):**

- `mutation_engine.py` -- the linguistic engine: takes transcribed words
  and POS tags, works out what mutation *should* apply where, and compares
  it to what actually happened. This is where "erosion" gets decided.
- `mutation_tables.py` -- every mutation rule, trigger word, and lexicon
  the engine knows about, as plain data (no logic). If you're checking or
  adding a linguistic rule, this is the file to open.
- `cysill_client.py` -- talks to the Cysill API (POS tags, lemmas), with
  retries and a circuit breaker that falls back to spaCy-only if Cysill is
  down for a whole run.
- `spacy_tagging.py` -- loads the Welsh spaCy model and turns its output
  into plain data the rest of the pipeline uses.
- `corpus_ops.py` -- file I/O: the video queue, processed/failed logs,
  audio download, the `analyze()` function that runs one video end to end,
  and the completion email.
- `bangor_lexicon.py` -- optional, local, offline lookup against
  Techiaith's own Bangor lexicon (~830k wordforms). Loaded once at
  startup if available (see **Setup** step 5); resolves most lemmas, and
  a smaller set of unambiguous POS/mutation/gender readings, without
  going through the Cysill API at all. Never populates `cysill_pos`
  itself (different tag scheme, no published mapping) -- only the
  translated `cysill_mutation_type`/`cysill_gender` fields, and lemmas.

**Standalone companions (each also runs directly; most are called
automatically at the right point in the main workflow -- exceptions
noted below):**

- `fetch_captions.py` -- downloads a video's YouTube captions and checks
  them against a mutations CSV already produced for that video. Aligns
  the whole video's Whisper word stream (from `words_*.csv`, which has
  per-word timestamps) against the whole caption track in one pass,
  rather than comparing small time windows -- more robust to Whisper's
  and the caption track's segments being chunked completely
  independently of each other.
- `manual_editing.py` -- an interactive terminal tool for reviewing
  mutation rows one at a time: confirm or overturn each finding, flag
  anything uncertain, leave notes, search, or just skim a summary.
- `corpus_analyzer.py` -- reads every mutations CSV you've ever produced,
  merges them, and generates corpus-wide figures and a text summary.
- `validate_against_chat.py` -- **not** wired into the main menu, run by
  hand. Compares Whisper's own transcript for a Bangor Siarad corpus
  recording against that recording's human-made CHAT-format (`.cha`)
  ground-truth transcript, using the same whole-video alignment engine as
  `fetch_captions.py`. Useful as a spot-check on genuinely spontaneous,
  overlapping, multi-speaker audio -- the one place in this project with
  a real ground-truth transcript to measure Whisper against, rather than
  assuming it degrades gracefully the way it does on scripted/produced
  sources. Always run with `--dump-clean` on a new file first (its CHAT
  annotation cleanup hasn't been validated against a real downloaded
  Bangor Siarad file yet -- see the script's own docstring).

## Workflow

`welsh_pipeline.py`'s menu has three categories -- pick a number, then a
letter. `q` at either level cancels back without losing anything already
done.

The usual path: **1a** (discover) -> **1b** (process queue) -> **1d**
(review by hand) -> **2a** (generate figures).

**1 -- Queue & Processing**
- **a** Discover new videos -- scans `CURATED_CHANNELS` (in `mutation_engine.py`) for anything new, adds it to `video_queue.json`. Doesn't download or transcribe.
- **b** Process queue -- transcribes, analyzes, and (YouTube sources only) caption-corroborates. Failed videos retry up to 3x (`failed_videos.json`) before being given up on. Sends a completion email if configured.
- **c** Manage queue -- view, filter, or remove queued videos.
- **d** Manually review mutations -- launches `manual_editing.py` (`--help` for its filtering options).
- **e** Re-run mutation rule(s) -- launches `rerun_rules.py` to re-evaluate already-transcribed videos after a rule change, without re-transcribing or re-hitting Cysill.

**2 -- Analysis**
- **a** Run corpus analyzer -- merges every mutations file ever produced into corpus-wide figures + a summary. Read-only; also runnable directly as `python corpus_analyzer.py`.

**3 -- Testing**
- **a** Test a Welsh phrase -- no audio, no transcription wait.
- **b** Analyze local MP3 files -- point it at a folder of MP3s (no captions to corroborate against). Asks **save** (real corpus, same as 1b) or **preview** (writes to `mp3_previews/` instead, never marked processed) -- use preview to sanity-check an unfamiliar audio source before committing it.

## Where your data ends up

Everything lives under `WELSH_ANALYSIS_DIR` (or `~/welsh_analysis` if you
didn't set that variable). The full folder layout is created up front on
every run (not lazily, one folder at a time, as each menu option first
needs it), so you'll see all of it even before you've used every menu
option:

```
WELSH_ANALYSIS_DIR/
├── audio/                                  downloaded/local MP3s
├── test_audio/                             drop local MP3s here for Testing -> b
├── captions/<stamp>/<slug>/                downloaded .vtt + parsed .csv caption files --
│                                              nested per-run/per-video, same as transcriptions/
│                                              and mutations/ below
├── transcriptions/<stamp>/<slug>/
│   ├── segments_<stamp>_<slug>.csv         Whisper's segment-level transcript
│   ├── words_<stamp>_<slug>.csv            word-level transcript + POS tags + per-word
│   │                                        timestamps (what caption corroboration aligns against)
│   ├── lemmas_<stamp>_<slug>.csv
│   └── pos_<stamp>_<slug>.csv
├── mutations/<stamp>/<slug>/
│   ├── mutations_<stamp>_<slug>.csv        the actual findings -- this is what
│   │                                        manual_editing.py and corpus_analyzer.py read
│   └── mutations_..._precaption_backup.csv only present if captions were fetched --
│                                              a one-time pre-corroboration safety copy
├── summaries/                              per-run CSV summaries written by Queue & Processing -> b
│                                              itself (research_summary/, erosion_by_trigger_type/,
│                                              erosion_by_rule/), not Analysis -> a
├── analysis/                               Analysis -> a's output: merged_mutations.csv,
│   └── figures/                              utterance_export.csv, and chart images
├── phrase_tests/                           Testing -> a's ad-hoc "test a Welsh phrase"
│                                              output -- deliberately kept outside mutations/
│                                              and transcriptions/ so it never gets swept into
│                                              the real corpus by Analysis -> a or rerun_rules.py
├── mp3_previews/                           Testing -> b's output when you choose "preview" instead
│                                              of "save" -- same quarantine idea as phrase_tests/,
│                                              never marked processed, never swept into the corpus
├── video_queue.json                        pending videos (Queue & Processing -> a adds, -> b consumes)
├── processed_videos.json                   videos already handled by Queue & Processing -> b
├── processed_local_mp3s.json               local files already saved via Testing -> b
├── failed_videos.json                      videos that errored, with retry count
└── lemma_cache.json                        word -> lemma lookups, cached across every run so
                                              repeat words (very common in Welsh function words)
                                              don't re-hit the Cysill API or simplemma every time
```

`captions/`, `transcriptions/`, and `mutations/` all share the exact same
`<stamp>/<slug>/` nesting, generated once per video by `_video_slug()` --
browsing any one of the three by run, then by video, lands you on the
same folder name in the other two. The caption `.vtt`/`.csv` filenames
themselves are still named by video ID + language (yt-dlp's own
convention, e.g. `abc123.cy.vtt`), not by stamp/slug, since that naming
comes from yt-dlp internally -- only the *directory* they land in follows
the shared per-video layout.

Running `fetch_captions.py` directly from the command line (rather than
through the main pipeline) has no run/video context to nest into, so it
still saves flat into `captions/` at the top level -- that's expected for
ad-hoc standalone use, not a bug.

If you see a `_precaption_backup.csv` next to a video's main mutations
file, that's expected and safe to leave alone -- it's a one-time snapshot
taken right before caption corroboration writes its columns into the real
file, so corroboration can never be accidentally run twice on top of
itself. `corpus_analyzer.py` and `manual_editing.py` both know to skip it
automatically.

## State and recovery

Queue, cache, and output files all live in `WELSH_ANALYSIS_DIR`. Queue
failures are recorded in `failed_videos.json` and retried up to three
times before being marked processed anyway. Successfully processed local
MP3s are recorded in `processed_local_mp3s.json`, keyed by filename and
file size -- editing or replacing a file makes it eligible for
reprocessing automatically.

Never store API keys or email passwords in source files. Use environment
variables instead.