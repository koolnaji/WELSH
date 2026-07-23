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
5. Copy `.env.example` to your preferred environment-variable setup and
   fill in what you need (see **Environment variables** below). Only
   `WELSH_ANALYSIS_DIR` affects core functionality; the rest are optional.
6. Run `python welsh_pipeline.py`. This opens the main interactive menu
   (see **Workflow** below).

### Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `WELSH_ANALYSIS_DIR` | No | Where all output lives (audio, transcripts, mutations, summaries, queue/cache files). Defaults to `~/welsh_analysis` if unset. |
| `WELSH_LEMMATIZER` | No | API key for the Cysill (techiaith.cymru) POS/lemmatizer service. Without it, the pipeline falls back to spaCy + local heuristics only -- it still works, just with one fewer independent tagger cross-checking every word. |
| `GMAIL_SENDER` / `GMAIL_APP_PASSWORD` / `NOTIFY_RECIPIENT` | No | Enables an HTML completion-email summary (run stats, per-video results, erosion breakdown) after menu options 1 and 3 finish. Needs a Gmail account with 2-Step Verification and an App Password (Google Account -> Security -> 2-Step Verification -> App passwords) -- not your normal Gmail password. `NOTIFY_RECIPIENT` defaults to `GMAIL_SENDER` (i.e. emails yourself) if unset. Leave all three blank to disable notifications entirely; the pipeline runs exactly the same either way, it just skips the email at the end. |

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

**Standalone companions (each also runs directly, and each is called
automatically at the right point in the main workflow):**

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

## Workflow

The usual path through a Welsh channel looks like: **2** (find videos) ->
**3** (transcribe + analyze + auto-corroborate) -> **7** (review by hand) ->
**6** (generate figures once you're happy with the data). Everything below
is a menu option in `welsh_pipeline.py`; entering `q` at any sub-prompt
cancels back to this menu without losing anything already done.

**1 -- Analyze local MP3 files.** Point it at a folder of Welsh MP3s (no
YouTube involved, so no captions to corroborate against). Transcribes and
analyzes each one exactly like option 3 does, then writes the same output
files. Remembers which files it's already processed (by filename + size)
in `processed_local_mp3s.json`, so re-running after adding new files to
the folder only processes the new ones -- and if a file changes size, it's
treated as new and reprocessed.

**2 -- Discover new videos.** Scans the channels configured in
`CURATED_CHANNELS` (inside `mutation_engine.py`) for videos not already in
your queue or already processed, and adds them to `video_queue.json` for
option 3 to work through later. Doesn't download or transcribe anything
itself.

**3 -- Process queue.** The main event. Pulls videos off the queue (you
choose how many) and, for each one: fetches its YouTube captions first (if
available), transcribes the audio with Whisper, runs the full mutation
analysis, saves everything to its own subfolder, then immediately
cross-checks the mutation findings against the captions it fetched at the
start. A video that fails doesn't stop the batch -- it's retried up to 3
times across future runs (`failed_videos.json`) before being given up on
and marked processed anyway, so one bad video can't block the queue
forever. Sends a completion email if you've set up notifications.

**4 -- Test a Welsh phrase.** Type or paste a short phrase directly (no
audio) and see how the engine analyzes it. Useful for checking a
linguistic rule against a specific example without waiting on
transcription.

**5 -- Manage queue.** View, reorder, or remove videos sitting in the
queue before they're processed.

**6 -- Run corpus analyzer.** Runs `corpus_analyzer.py` over every
mutations file you've produced so far (across every past run, not just
the most recent one), merges them into one dataset, and generates figures
plus a text summary. This is a read-only reporting step -- it never
changes your underlying mutation data, and you can also run
`python corpus_analyzer.py` directly from the command line for the same
result.

**7 -- Manually review mutations.** Launches `manual_editing.py`, which
walks you through mutation rows one at a time so you can confirm or
correct each `is_erosion` call by hand. At each row: `[y]es`/`[n]o` records
your decision and moves on, `[s]kip` leaves it untouched, `[f]lag` marks it
for later with an optional note, `[j]ump` goes straight to a row number,
`[/]` searches the whole queue, `[` `]` jump to the previous/next file's
first row, `[c]` clears a row's review history, `[p]revious` steps back,
`[q]uit` saves and exits. Rows you've already reviewed are skipped on the
next run by default (pass `--review-all` to see them again); flagged and
skipped rows always reappear until you decide on them. Run
`python manual_editing.py --help` to see every filtering option
(`--status`, `--trigger`, `--rule`, `--flagged-only`, `--sample N`, and
more) for narrowing the queue before you start.

## Where your data ends up

Everything lives under `WELSH_ANALYSIS_DIR` (or `~/welsh_analysis` if you
didn't set that variable). The full folder layout is created up front on
every run (not lazily, one folder at a time, as each menu option first
needs it), so you'll see all of it even before you've used every menu
option:

```
WELSH_ANALYSIS_DIR/
├── audio/                                  downloaded/local MP3s
├── test_audio/                             drop local MP3s here for option 1
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
├── summaries/                              per-run CSV summaries written by option 3 itself
│                                              (research_summary/, erosion_by_trigger_type/,
│                                              erosion_by_rule/), not option 6
├── analysis/                               option 6's output: merged_mutations.csv,
│   └── figures/                              utterance_export.csv, and chart images
├── phrase_tests/                           menu option 4's ad-hoc "test a Welsh phrase"
│                                              output -- deliberately kept outside mutations/
│                                              and transcriptions/ so it never gets swept into
│                                              the real corpus by option 6 or rerun_rules.py
├── video_queue.json                        pending videos (option 2 adds, option 3 consumes)
├── processed_videos.json                   videos already handled by option 3
├── processed_local_mp3s.json               local files already handled by option 1
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