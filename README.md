# Welsh grammatical erosion pipeline

An interactive pipeline for transcribing Welsh audio and testing whether
specific grammatical structures erode under sustained English-contact
pressure -- and whether that erosion tracks a structure's *typological
foreignness* to English specifically, not just informality in general.

## The hypothesis this pipeline tests

Dominant-language contact pressure erodes the structures the dominant
language has no equivalent machinery for *faster* than structures it does
have some counterpart for. Four grammatical phenomena are measured
against that prediction:

| Branch | Phenomenon | English counterpart? | Predicted |
|---|---|---|---|
| `mutation_*` | Initial consonant mutation (soft/nasal/aspirate) | None at all | Erodes |
| `prep_*` | Conjugated prepositions ("arna i" vs. "ar fi") | None -- English prepositions never conjugate | Erodes |
| `numeral_*` | Singular noun after a numeral ("tri chi", not "tri cŵn") | None -- English uses a plural | Erodes (toward the English plural) |
| `plural_*` | Plural noun after "rhai" ("rhai llyfrau") | Yes -- English "some books" | Resists erosion |

`numeral_*` and `plural_*` are a matched pair: both measure the same
thing -- the singular/plural tag on the noun -- so tagger errors hit both
sides equally and their rates compare directly.

**Data sources.** YouTube/podcast audio (transcribed by Whisper) and the
Bangor Siarad corpus (`corpus_siarad.py`): 40 hours of informal
conversation between 153 Welsh-English bilinguals aged roughly 18-72,
recorded 2005-08 at their homes or workplaces with no researcher present,
with human transcripts, speaker age/sex, and human code-switch tags. Every
Siarad detection row carries a `speaker` column that joins to that
conversation's `speakers_*.csv`.

A fourth module, `corpus_formality.py`, replaces a hand-assigned
formal/informal/casual channel label with a grounded, continuous,
per-video formality score computed from the transcript itself (POS-class
balance, filler rate, lexical diversity) -- so "does erosion track
formality" is an actual regression against a measured variable, not a
comparison between three researcher-picked buckets.

## What this actually does, in one paragraph

Point it at Welsh-language audio (a local MP3, or a YouTube channel
you've added to a queue). It transcribes with Whisper, tags every word
with two independent part-of-speech taggers (the Cysill API and a local
spaCy parser), and runs all four detection branches above over the same
tagged word stream -- each producing its own findings (mutation-, prep-,
numeral- and plural-specific CSVs) from the same transcript pass. Where YouTube
captions exist, mutation findings are cross-checked against them.
Everything lands in CSVs you can review by hand, or summarize into
corpus-wide figures, including an erosion-vs-formality regression.

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
   CC0), unzip it, and put `lecsicon_cc0.txt` in the pipeline folder (next
   to `bangor_lexicon.py`), or set `BANGOR_LEXICON_PATH` to its full path.
   Startup prints `✅ Bangor lexicon loaded (... wordforms) from <path>`
   when it is found. It resolves lemmas, and noun gender/number from the
   dictionary (which outrank spaCy's guess wherever the lexicon has one),
   without hitting the Cysill API. The pipeline continues without it,
   falling back to Cysill/spaCy/simplemma.
6. Set the environment variables you need (see **Environment variables**
   below) in the shell you run the pipeline from -- e.g. `export
   WELSH_LEMMATIZER=...` in Git Bash, or in `~/.bashrc` so every session
   has it. **Nothing reads `.env` automatically**; `.env.example` is only a
   list of the variables. Only `WELSH_ANALYSIS_DIR` affects core
   functionality; the rest are optional. Startup prints a `Cysill:` line
   saying whether the key was found.
7. Set up a PO token provider for yt-dlp (see **YouTube PO tokens**
   below). Without this, audio downloads may intermittently or
   persistently fail with `HTTP Error 403: Forbidden` even when
   captions/metadata calls succeed fine -- this is a separate issue from
   cookie auth (step 6) and from YouTube's rate limiting, and has become
   common enough industry-wide during 2026 that it's effectively
   required now, not just a nice-to-have.
8. Run `python welsh_pipeline.py`, optionally followed by a transcription
   preset (`fast`, `balanced`, or `accurate` -- see **Transcription
   presets** below) and/or `--sample-minutes`/`--skip-minutes` (see
   **Sampling long videos** below). This opens the main interactive menu
   (see **Workflow** below).

### Transcription presets

`python welsh_pipeline.py [fast|balanced|accurate]`. Controls Whisper's
`beam_size`/`best_of`/`temperature` decoding settings -- the knobs that
actually drive transcription time. Omitting this argument (or running
`accurate` explicitly) reproduces the pipeline's original hardcoded
behavior exactly; nothing changes unless you deliberately pick something
else.

| Preset | `beam_size` | `temperature` fallback | Use when |
|---|---|---|---|
| `fast` | 5 | none (`[0.0]` only) | You want the fastest turnaround and trust the hallucination filter to catch what fallback would have caught |
| `balanced` | 5 | one retry (`[0.0, 0.2]`) | A middle ground |
| `accurate` (default) | 7 | full fallback (`[0.0, 0.2, 0.4]`) | Original behavior; slowest, most thorough |

The `temperature` list isn't "try three temperatures and blend them" --
it's sequential fallback. Whisper decodes at the first value; only if
that decode fails quality checks does it re-decode the *entire segment
from scratch* at the next value, paying the full `beam_size`/`best_of`
cost again each time. On messy or code-switching audio this can trigger
often enough to multiply decode time substantially on the affected
segments -- likely the single biggest lever if a run feels slower than
it should. Cutting fallback (`fast`) is safe to experiment with on this
project specifically because `filter_hallucinated_segments()` already
exists downstream to catch garbled output that fallback would otherwise
have tried to fix -- you're not removing your only safety net, just the
expensive one. See `TRANSCRIBE_PRESETS` in `corpus_ops.py` for the exact
values and full reasoning.

Worth spot-checking a known video's output against a prior `accurate`
run before trusting `fast`/`balanced` for the rest of your corpus.

### Sampling long videos

`python welsh_pipeline.py [preset] --sample-minutes N [--skip-minutes M]`

For long recordings (2hr+ podcasts, in particular), transcribing the
whole file isn't necessary to get a valid measurement -- every metric
this project reports is a rate/proportion (mutation application rate,
code-switch rate, erosion rate), not a raw count, so a fixed-length
sample per video is a legitimate way to cut wall-clock time without
changing what's being measured.

| Flag | Default | Meaning |
|---|---|---|
| `--sample-minutes N` | unset (full video) | Only transcribe/analyze an `N`-minute window of each video instead of the whole file. Omit entirely for unchanged, full-video behavior. |
| `--skip-minutes M` | `5.0` | Where that window starts. Only matters if `--sample-minutes` is set. |

So `--sample-minutes 18` alone samples minutes 5-23 of every video --
skipping the first 5 minutes by default, since intros, cold opens, and
sponsor reads aren't representative of the spontaneous speech this
project is trying to measure, and sampling from 0:00 would
systematically feed the pipeline the *least* representative minutes of
every video. `--skip-minutes 0` samples from the true start if you want
that instead.

A video too short to support the requested window (i.e. `skip_minutes +
sample_minutes` exceeds the video's actual length) falls back to
sampling from 0:00 automatically, rather than seeking past the end of
the file and producing an empty clip.

**Sentence boundaries are respected, not just chopped.** A hard cut at
an arbitrary timestamp doesn't know or care about grammar, and mutation
detection needs at least sentence-level context -- a trigger/target pair
split across an artificial cut is unusable, not just noisy. So the
actual extraction pulls a slightly WIDER clip than requested (20s of
padding on each side, clamped to the real file boundaries) purely so
Whisper has complete audio for whatever sentence straddles each edge.
After transcription, any Whisper segment that only exists because of
that padding -- i.e. starts before, or ends after, the *true* requested
window -- is dropped before mutation detection ever sees it. A boundary
that happens to sit at the video's genuine 0:00 or its genuine true end
is never treated as an artificial cut, so nothing gets dropped there.
See `sample_audio_window()` and `_shift_and_trim_padded_segments()` in
`corpus_ops.py` for the full mechanism.

**Timestamps are corrected back to true-video time.** Whisper only ever
sees the extracted clip and counts from 0:00 of *that file* -- without
correction, every timestamp in the mutations/words CSVs would be off by
however much got trimmed off the front, and caption corroboration in
`mutation_captions.py` (which aligns against the real caption track's real
timestamps) would silently misalign on every sampled run. Every segment
and word timestamp is shifted back to true-video time immediately after
transcription, before anything else touches it -- so the `timestamp`
column in your CSVs, and caption corroboration, both work exactly the
same whether or not a video was sampled.

`video_duration_seconds` in the output CSVs reflects the TRUE (unpadded)
sample length when sampling is active, not the length of the padded
clip Whisper actually transcribed -- otherwise every "minutes of corpus
covered" total in `corpus_analyzer.py` would be silently inflated by
2x the padding on every sampled video.

Mechanically, the trim itself uses the audio FILE (via `ffmpeg -ss ...
-t ... -c copy`, a near-instant stream copy, not a re-encode) rather
than faster-whisper's own `clip_timestamps` parameter. That's
deliberate: per faster-whisper's own docs, passing `clip_timestamps`
makes it silently ignore `vad_filter` -- and this pipeline's VAD
settings are load-bearing for the hallucination defense
(`filter_hallucinated_segments()` downstream assumes VAD already did
its job). Trimming the file keeps VAD running exactly as it always has,
just over a shorter (padded) file.

The trimmed file is written next to the source audio as
`<name>_sample<true_start>-<true_end>s.mp3` (e.g.
`podcast_sample300-1380s.mp3` -- named for the TRUE requested window,
not the padded extraction) -- see **Where your data ends up** below.

This applies process-wide for the run, the same way a transcription
preset does -- it's not a per-video or per-menu-choice setting. Both
Queue & Processing -> b and Testing -> b respect it if set at launch.

Applies only to already-downloaded/local audio -- it doesn't reduce
what gets downloaded from YouTube first (that's still the full video).

### YouTube PO tokens

YouTube increasingly requires a PO (Proof-of-Origin) token on the actual
media (audio/video) fetch -- a cryptographic attestation mechanism,
separate from and unrelated to cookie authentication or the earlier
n-signature/JS-challenge handling (`remote_components: ["ejs:github"]`,
which needs Deno or another JS runtime installed and unblocked --
`Unblock-File` on Windows if downloaded rather than installed via a
package manager). Without a PO token provider, `download_audio()` can
fail with a persistent `HTTP Error 403: Forbidden` on the media URL
itself, even though caption/metadata calls for the same video succeed
normally -- that combination (captions fine, audio 403s) is the
signature of this specific issue rather than rate-limiting or cookie
problems.

1. Download the Rust POT provider binary (`bgutil-pot`) for your
   platform from
   [jim60105/bgutil-ytdlp-pot-provider-rs releases](https://github.com/jim60105/bgutil-ytdlp-pot-provider-rs/releases).
   On Windows, unblock the downloaded `.exe` the same way as Deno.
2. Download the matching plugin zip from the same releases page and
   extract it into a yt-dlp plugin directory (on Windows,
   `%APPDATA%\yt-dlp\plugins\`) -- you should end up with a
   `yt_dlp_plugins\extractor\` folder containing `getpot_bgutil*.py`
   files somewhere inside whatever folder you extracted.
3. Run the provider as an HTTP server: `bgutil-pot server --host
   127.0.0.1` (explicitly binding IPv4 avoids a mismatch with yt-dlp's
   default `base_url` of `http://127.0.0.1:4416` -- the binary's own
   default binds the IPv6 wildcard `[::]`, which yt-dlp's default
   `127.0.0.1` base URL won't reach). This needs to be running as a
   persistent background process for the full duration of any pipeline
   run -- it is not started automatically by `download_audio()` or
   anything else in this codebase.
4. Verify with `yt-dlp -v <any video URL>` and check for a
   `PO Token Providers: bgutil:http-...` line (not `unavailable`) in
   the debug output, and confirm the server's own log shows it
   generating tokens on request.

No `ydl_opts` changes are needed in `corpus_ops.py` for the default
setup -- yt-dlp auto-detects a correctly-installed plugin and reachable
server. If you run the server on a non-default host/port, that would
need `extractor_args: {"youtubepot-bgutilhttp": {"base_url":
"http://HOST:PORT"}}` merged into `ydl_opts` alongside
`yt_dlp_cookie_opts()`.

Providing a PO token does not *guarantee* a 403 won't happen -- per the
provider's own documentation, it may just make requests appear more
legitimate. If 403s persist after this is set up and confirmed
reachable, check for a leftover `.part` file from an earlier failed
attempt in `AUDIO_DIR` first (a resumed byte-range request against a
freshly re-signed URL can 403 independently of PO-token status --
`yt-dlp --no-continue` on the same URL is the fastest way to tell the
two failure modes apart).

If 403s persist even with a fresh (non-resumed) download and a
confirmed-reachable token server, check next whether yt-dlp is
fetching player data with one client but requesting a token for
another (visible in `-v` output as e.g. `Downloading android vr
player API JSON` followed by `Generating a gvs PO Token for web_safari
client`) -- that mismatch has been observed to 403 partway through an
otherwise-successful-looking download.

**Known issue, not yet fixed in this codebase (flagging honestly rather
than claiming otherwise):** a pinned `player_client` (e.g. forcing
`extractor_args: {"youtube": {"player_client": ["mweb"]}}` in
`download_audio()`) and a slower, audio-download-specific request
interval have both been discussed as mitigations for this and for a
separate, confirmed asymmetry -- low-view/niche-channel videos (this
project's actual corpus) 403 more readily than heavily-viewed videos
under otherwise identical conditions, plausibly because YouTube's
anti-bot heuristics weight traffic-pattern legitimacy signals that
low-view content simply doesn't have. **Neither mitigation is currently
implemented in `corpus_ops.py`/`youtube_access.py`** -- some project
documentation elsewhere describes them as already live, but a direct
check of this repo's actual `download_audio()` and
`youtube_access.call()` found no `player_client` pinning and no
audio-specific pacing parameter. Treat this as an open item, not a
solved one, until someone actually adds it here and this note is
updated.

**Unconfirmed as of 2026-08-17**: a nightly yt-dlp build was reported
to stop producing these 403s in quick manual testing. This has not
been isolated or confirmed at batch scale -- see `limitations.txt`
Section 1.4 before relying on it alone. If you do switch to nightly,
record the exact build (`yt-dlp --version`) somewhere durable, since
nightly builds aren't version-pinned and a later build could
reintroduce this behavior without warning.

### Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `WELSH_ANALYSIS_DIR` | No | Where all output lives (audio, transcripts, mutations, summaries, queue/cache files). Defaults to `~/welsh_analysis` if unset. |
| `WELSH_LEMMATIZER` | No | API key for the Cysill (techiaith.cymru) POS/lemmatizer service. Without it, the pipeline falls back to spaCy + local heuristics only -- it still works, just with one fewer independent tagger cross-checking every word. A missing key used to be completely silent; startup now prints a `Cysill:` status line, and every video prints a `Tagger coverage:` line with how many words spaCy, Cysill and the lexicon each covered. |
| `BANGOR_LEXICON_PATH` | No | Full path to `lecsicon_cc0.txt` (see **Setup** step 5). Only needed if the file is not in the pipeline folder. It is tried first; then the loader looks next to `bangor_lexicon.py`, then in a `bangor_lexicon/` subfolder there, then in `./bangor_lexicon/`. If it points at a file that doesn't exist, startup prints a warning and uses the first copy it finds in those places. |
| `YTDLP_COOKIES_FILE` / `YTDLP_COOKIES_FROM_BROWSER` | No | Authenticates yt-dlp's caption listing/download requests the same way a logged-in browser tab would. Audio downloads deliberately run without cookies and use them only as a fallback for videos that require sign-in (age gate, bot check, members-only) -- sending the cookie file on downloads produced persistent `HTTP Error 403` that went away without it (see `limitations.txt` Section 1.4). Note that yt-dlp writes cookies back into this file after every call, so point it at a copy used only by this pipeline, never your only export of a logged-in session. Channel discovery never uses cookies. YouTube rate-limits anonymous requests to its caption/timedtext endpoint hard (`HTTP Error 429: Too Many Requests`), and the resulting block has been reported to last on the order of hours -- authenticating avoids tripping it in the first place, rather than just retrying through it. `YTDLP_COOKIES_FILE` points at a `cookies.txt` (Netscape format, e.g. exported via a "Get cookies.txt LOCALLY" browser extension -- portable between machines, and the more reliable option on Windows, see below); `YTDLP_COOKIES_FROM_BROWSER` names a browser (`chrome`, `firefox`, ...) to read cookies live from instead, machine-local only. If both are set, the file wins. Leave both blank to run fully anonymous, exactly as before this existed. **Windows + Chrome-family browsers:** newer Chrome versions' "app-bound encryption" is known to break yt-dlp's live cookie decryption on Windows (see [yt-dlp#15401](https://github.com/yt-dlp/yt-dlp/issues/15401)) -- if `YTDLP_COOKIES_FROM_BROWSER=chrome` fails to decrypt, either try `firefox` instead or switch to `YTDLP_COOKIES_FILE`. |
| `GMAIL_SENDER` / `GMAIL_APP_PASSWORD` / `NOTIFY_RECIPIENT` | No | Enables an HTML completion-email summary (run stats, per-video results, erosion breakdown) after Queue & Processing -> b (Process queue) or Testing -> b (Analyze local MP3 files, when saved) finish. Needs a Gmail account with 2-Step Verification and an App Password (Google Account -> Security -> 2-Step Verification -> App passwords) -- not your normal Gmail password. `NOTIFY_RECIPIENT` defaults to `GMAIL_SENDER` (i.e. emails yourself) if unset. Leave all three blank to disable notifications entirely; the pipeline runs exactly the same either way, it just skips the email at the end. |

Never commit real values for any of these -- keep them in your actual
environment/shell profile/`.env`, not in source files.

## The pieces

`welsh_pipeline.py` is the only thing you run directly. Everything else is
either a module it imports, or a standalone companion tool you can also
run on its own from the command line.

**Naming convention:** files are prefixed by which branch owns them --
`mutation_*.py` (consonant mutation), `prep_*.py` (conjugated
prepositions), `plural_*.py` (plural marking). Unprefixed files are
shared infrastructure or cross-branch analysis, used by every branch,
owned by none of them. Each branch's `_tables.py` is pure linguistic
data (no logic); its `_engine.py`/`engine.py` is the detection logic. A
branch's engine never imports another branch's tables or engine
directly -- shared capabilities (tagging output, the consumption-tracking
guard that stops two rules double-counting the same word) live in
`spacy_tagging.py` instead, so branches stay independent of each other's
internals and a new branch can be added the same way without touching
the existing ones.

**Shared infrastructure (imported by every branch):**

- `corpus_io.py` -- the single source of truth for "where does this
  project's state/output actually live": directory layout, the
  `runs/<stamp>/<slug>/` per-video path scheme, every JSON state log
  (queue/processed/failed), lemma-cache and checkpoint persistence, and
  `CURATED_CHANNELS` (the list of channels/feeds `discover_new_videos`
  scans -- relocated here from `mutation_engine.py`, since channel
  discovery isn't mutation-specific).
- `corpus_ops.py` -- file I/O and orchestration: the video queue,
  processed/failed logs, audio download, and the `analyze()`/
  `analyze_phrase()` functions that run Whisper plus *every* detection
  branch (mutation, prep, plural) over one video or phrase end to end,
  plus the completion email. Also owns `TRANSCRIBE_PRESETS` (see
  **Transcription presets** above) and `sample_audio_window()` (see
  **Sampling long videos** above).
- `spacy_tagging.py` -- loads the Welsh spaCy model, turns its output
  into plain data the rest of the pipeline uses, and hosts a few small
  generic capabilities every branch needs: `extract_gender_from_spacy`/
  `extract_number_from_spacy` (reading UD morph features into this
  project's shared vocabulary), and `mark_consumed`/`was_consumed` (the
  guard that stops a word already scored by one rule from being
  independently re-scored by another rule walking the same word
  stream).
- `cysill_client.py` -- talks to the Cysill API (POS tags, lemmas), with
  retries and a circuit breaker that falls back to spaCy-only if Cysill is
  down for a whole run. Once tripped, later calls return instantly and
  silently rather than re-attempting or re-announcing failure -- a long
  run doesn't get slower or noisier just because Cysill went down early
  in it.
- `bangor_lexicon.py` -- optional, local, offline lookup against
  Techiaith's own Bangor lexicon (~830k wordforms). Loaded once at
  startup if available (see **Setup** step 5); resolves most lemmas, and
  a smaller set of unambiguous POS/mutation/gender readings, without
  going through the Cysill API at all. Never populates `cysill_pos`
  itself (different tag scheme, no published mapping) -- only the
  translated `cysill_mutation_type`/`cysill_gender` fields, and lemmas.
  Per word, it also gives noun gender and number from the word's noun
  readings (`lex_gender`/`lex_number` in `pos_*.csv`), kept only when all
  noun readings agree. These come before spaCy for the feminine-noun
  mutation rules and for the numeral/rhai number check (`number_source`
  in those CSVs says which one decided). Epicene nouns (`Gender=Fem,Masc`)
  come back as `epicene`, so no feminine-noun rule fires on them.
  Also recognizes English code-switch words and skips sending them to
  Cysill at all, rather than letting a Welsh-only tagger guess at them.

  **A row resolved this way (or via a recognized code-switch word)
  carries `locally_resolved=True` in the mutations CSV, and always has an
  empty `cysill_pos`** -- it never counts toward genuine Cysill
  corroboration in `tagger_agreement`/`detection_source`/
  `confidence_score`, even though `cysill_mutation_type`/`cysill_gender`
  are still populated and usable. Worth knowing if you're doing your own
  analysis on top of the mutations CSVs rather than going through
  `corpus_analyzer.py`: `cysill_pos` being empty doesn't mean "Cysill had
  nothing to say," it can also mean "this word never needed asking."

**Mutation branch (no English counterpart -- predicted to erode):**

- `mutation_engine.py` -- the linguistic engine: takes transcribed words
  and POS tags, works out what mutation *should* apply where, and compares
  it to what actually happened. This is where "erosion" gets decided.
- `mutation_tables.py` -- every mutation rule, trigger word, and lexicon
  the engine knows about, as plain data (no logic). If you're checking or
  adding a linguistic rule, this is the file to open.
- `mutation_captions.py` -- downloads a video's YouTube captions and
  checks them against a mutations CSV already produced for that video.
  Aligns the whole video's Whisper word stream (from `words_*.csv`,
  which has per-word timestamps) against the whole caption track in one
  pass, rather than comparing small time windows -- more robust to
  Whisper's and the caption track's segments being chunked completely
  independently of each other. YouTube-request retry/backoff/pacing is
  delegated entirely to `youtube_access.py`'s shared coordinator (see
  below) -- this file does not keep its own separate circuit breaker.
  Transcription/mutation output for the video itself is unaffected by a
  caption failure either way -- captions are corroboration-only.
- `mutation_manual_editing.py` -- an interactive terminal tool for
  reviewing mutation rows one at a time: confirm or overturn each
  finding, flag anything uncertain, leave notes, search, or just skim a
  summary.
- `mutation_rerun_rules.py` -- re-evaluates already-transcribed videos
  against an updated rule in `mutation_engine.py`/`mutation_tables.py`,
  without re-transcribing or re-hitting Cysill. Writes a
  `*_rerun_candidate.csv` comparison file by default; `--commit` applies
  it, never overwriting a `manual_reviewed=True` row.

**Preposition branch (no English counterpart -- predicted to erode):**

- `prep_engine.py` -- detects conjugated Welsh prepositions ("arna i")
  against the eroded, analytic pattern English already has natively
  ("ar fi" -- bare preposition + independent pronoun). The colloquial
  dropping of a conjugated form's final unstressed `-f` (e.g. spoken
  "arna" for citation-form "arnaf") is treated as ordinary phonology, not
  erosion, via a general rule applied at match time -- not something
  worth conflating with the actual phenomenon under study.
- `prep_tables.py` -- the conjugated-preposition paradigms this branch
  checks against, as plain data, cross-checked against multiple sources
  (see the file's own comments for exactly which forms came from where,
  and which cells are inferred rather than directly sourced).

**Numeral branch (no English counterpart -- predicted to erode):**

- `numeral_engine.py` -- a numeral directly followed by a noun ("tri
  chi", "dwy flynedd") should take the singular; a plural there is the
  English pattern. The partitive "tri o'r plant" is native and skipped, as
  is "un" (singular in English too). The trigger must be tagged as a
  numeral and the target as a non-code-switched noun with a known number.
- `numeral_tables.py` -- numeral forms (including mutated ones), as data.

**Plural-marking branch (has an English counterpart -- predicted to
resist erosion):**

- `plural_engine.py` -- after "rhai" ("some"), checks whether the
  following noun carries plural marking at all (not which allomorph --
  Welsh plural formation is too irregular to verify the exact form).
  Welsh and English both require a plural here, so this is the partner
  of the numeral branch. ("rhai pobl" is standard despite "pobl" being
  grammatically singular.) The original trigger, "y rhain" + noun, never
  fired on real speech -- "y rhain" is a pronoun ("these ones").
- `plural_tables.py` -- trigger forms and collective-noun exceptions.

**Human-transcribed corpus:**

- `corpus_siarad.py` -- `python corpus_siarad.py <file.cha | folder>`
  runs Bangor Siarad CHAT transcripts through the same tagging and
  detection as audio (via `corpus_ops.analyze_segments()`), writing the
  usual CSVs plus `speakers_*.csv` (age, sex, per-speaker notes, English
  word counts) and `utterances_*.csv` (raw and cleaned text, %gls/%eng
  tiers). Transcripts: TalkBank (Bangor/Siarad, the cited version) or
  bangortalk.org.uk; GPLv3, cite Deuchar, Webb-Davies & Donnelly (2009),
  doi:10.21415/T5088V. Word times are interpolated within each utterance
  (CHAT only times whole utterances). Siarad rows land under `runs/`
  alongside video rows -- filter on `source == "siarad"` to separate them.

**Cross-branch analysis:**

- `corpus_analyzer.py` -- reads every mutations CSV you've ever produced,
  merges them, and generates corpus-wide figures and a text summary,
  including the erosion-vs-formality-score regression (see below).
- `corpus_formality.py` -- computes a grounded, continuous, per-video
  formality score retroactively from transcript data already collected
  (no re-transcription needed): the Heylighen & Dewaele (1999) F-score
  from POS-class balance, filler-word rate, lexical diversity, and mean
  words per segment. Replaces a hand-assigned, per-channel
  formal/informal/casual label that had no way to be independently
  checked. Code-switch rate is computed and reported alongside this, but
  deliberately kept as its own separate figure/column, not folded into
  the formality score -- see the module's own docstring for why mixing
  the two would muddy what an erosion-formality correlation actually
  reflects.
- `youtube_access.py` -- the single shared coordinator for *every*
  yt-dlp request in a run (captions, audio downloads, channel discovery
  alike): paced inter-request timing, escalating jittered backoff on a
  429, and a rate-limit cooldown that persists to disk and survives a
  process restart, not just the rest of the current run -- since a
  YouTube 429 block has been reported to last on the order of hours,
  longer than any single run.

## Workflow

`welsh_pipeline.py`'s menu has three categories -- pick a number, then a
letter. `q` at either level cancels back without losing anything already
done.

The usual path: **1a** (discover) -> **1b** (process queue) -> **1d**
(review by hand) -> **2a** (generate figures).

**1 -- Queue & Processing**
- **a** Discover new videos -- scans `CURATED_CHANNELS` (in `corpus_io.py`) for anything new, adds it to `video_queue.json`. Doesn't download or transcribe.
- **b** Process queue -- transcribes, then runs all three detection branches (mutation, prep, plural) and (YouTube sources only) caption-corroborates the mutation findings. Failed videos retry up to 3x (`failed_videos.json`) before being given up on. Sends a completion email if configured.
- **c** Manage queue -- view, filter, or remove queued videos.
- **d** Manually review mutations -- launches `mutation_manual_editing.py` (`--help` for its filtering options).
- **e** Re-run mutation rule(s) -- launches `mutation_rerun_rules.py` to re-evaluate already-transcribed videos after a rule change, without re-transcribing or re-hitting Cysill.

**2 -- Analysis**
- **a** Run corpus analyzer -- merges every mutations file ever produced into corpus-wide figures + a summary, including the erosion-vs-formality regression (`corpus_formality.py`). Read-only; also runnable directly as `python corpus_analyzer.py`.

**3 -- Testing**
- **a** Test a Welsh phrase -- no audio, no transcription wait. Runs all three detection branches on the typed phrase.
- **b** Analyze local MP3 files -- point it at a folder of MP3s (no captions to corroborate against). Asks **save** (real corpus, same as 1b) or **preview** (writes to `mp3_previews/` instead, never marked processed) -- use preview to sanity-check an unfamiliar audio source before committing it.

## Where your data ends up

Everything lives under `WELSH_ANALYSIS_DIR` (or `~/welsh_analysis` if you
didn't set that variable). The full folder layout is created up front on
every run (not lazily, one folder at a time, as each menu option first
needs it), so you'll see all of it even before you've used every menu
option:

```
WELSH_ANALYSIS_DIR/
├── runs/<stamp>/                           one folder per pipeline run
│   ├── research_summary_<stamp>.csv        session-level summaries, written by
│   ├── erosion_by_trigger_type_<stamp>.csv   Queue & Processing -> b itself,
│   ├── erosion_by_rule_<stamp>.csv           not Analysis -> a
│   └── <slug>/                             one folder per video, generated by _video_slug()
│       ├── segments_<stamp>_<slug>.csv     Whisper's segment-level transcript
│       ├── words_<stamp>_<slug>.csv        word-level transcript + POS tags + per-word
│       │                                     timestamps (what caption corroboration aligns against)
│       ├── lemmas_<stamp>_<slug>.csv
│       ├── pos_<stamp>_<slug>.csv
│       ├── mutations_original_<stamp>_<slug>.csv       the actual findings, as first detected --
│       │                                                 this is what mutation_manual_editing.py and
│       │                                                 corpus_analyzer.py read; never modified
│       │                                                 by corroboration
│       ├── mutations_corroborated_<stamp>_<slug>.csv   only present once captions are fetched --
│       │                                                 corroboration's cross-checked result,
│       │                                                 written to this separate file
│       ├── prep_mutations_<stamp>_<slug>.csv           conjugated-preposition branch findings
│       ├── plural_mutations_<stamp>_<slug>.csv         plural-marking branch findings
│       ├── <video_id>_<title>.mp3          raw audio -- plus a `_sample<start>-<end>s.mp3`
│       │                                     variant next to it if trimmed by --sample-minutes
│       │                                     (see **Sampling long videos**)
│       ├── <video_id>_<title>_norm.mp3     normalized audio
│       └── <video_id>.<lang>.vtt / .csv    downloaded + parsed captions -- named by yt-dlp's own
│                                              video-ID + language convention, not stamp/slug
├── runs/unclassified_audio/
│   ├── original/                           audio migrate_to_new_structure.py couldn't match to a
│   └── normalized/                           video (no fetched captions to recover its ID from --
│                                               typically music/session recordings), split the same
│                                               way per-video audio is, by a `_norm` filename marker
├── runs/_deleted/                          the pipeline's own soft-delete archive
├── test_audio/                             drop local MP3s here for Testing -> b
├── analysis/                               Analysis -> a's output: merged_mutations.csv,
│                                              utterance_export.csv, video_formality.csv
│                                              (per-video formality scores, corpus_formality.py),
│                                              and asr_divergence.csv
│   └── figures/                              chart images, including erosion_vs_formality.png
│                                              and erosion_vs_codeswitch.png
├── phrase_tests/                           Testing -> a's ad-hoc "test a Welsh phrase"
│                                              output -- deliberately kept outside runs/ so it
│                                              never gets swept into the real corpus by
│                                              Analysis -> a or mutation_rerun_rules.py
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

Everything for a run and a video -- transcript, mutations, audio, and
captions alike -- lands in one shared `runs/<stamp>/<slug>/` folder,
generated once per video by `_video_slug()`. Older code that still refers
to `TRANS_DIR`, `MUT_DIR`, `CAPTIONS_DIR`, `AUDIO_DIR`, or `SUMMARY_DIR`
(`corpus_io.py`) is reading from this same tree -- those names are now
aliases for the one `runs/` directory, kept so code written for the old
separate-tree layout didn't all need renaming at once. If you're reading
older code or docs that describe `transcriptions/`, `mutations/`,
`captions/`, or `summaries/` as distinct top-level folders, that's the
pre-migration layout; `migrate_to_new_structure.py` is what moved
everything into the flat structure shown above.

Running `mutation_captions.py` directly from the command line (rather
than through the main pipeline) has no run/video context to nest into,
so it still saves flat into `runs/` at the top level -- that's expected
for ad-hoc standalone use, not a bug.

Corroboration never modifies a video's `mutations_original_*.csv` --
running it (which happens automatically whenever captions are fetched)
writes its cross-checked result to a separate `mutations_corroborated_*.csv`
instead, so the original, pre-corroboration findings are always still
there to compare against. `corpus_analyzer.py` and `mutation_manual_editing.py`
both know to prefer the corroborated file when one exists.

`prep_mutations_*.csv` and `plural_mutations_*.csv` are each a single
file per video (no original/corroborated split -- caption corroboration
is currently mutation-specific only) and are not yet merged into
`corpus_analyzer.py`'s corpus-wide figures; reading them directly, or
extending `corpus_analyzer.py` to aggregate them the same way it does
mutation findings, is the natural next step once those two branches have
been validated against real corpus audio.

## State and recovery

Queue, cache, and output files all live in `WELSH_ANALYSIS_DIR`. Queue
failures are recorded in `failed_videos.json` and retried up to three
times before being marked processed anyway. Successfully processed local
MP3s are recorded in `processed_local_mp3s.json`, keyed by filename and
file size -- editing or replacing a file makes it eligible for
reprocessing automatically.

Never store API keys or email passwords in source files. Use environment
variables instead.