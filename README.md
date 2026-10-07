# Welsh grammatical erosion pipeline

An interactive pipeline for transcribing Welsh audio and testing whether
specific grammatical structures erode under sustained English-contact
pressure -- and whether that erosion tracks a structure's *typological
foreignness* to English specifically, not just informality in general.

## The hypothesis this pipeline tests

Dominant-language contact pressure erodes the structures the dominant
language has no equivalent machinery for *faster* than structures it does
have some counterpart for. Five grammatical phenomena are measured
against that prediction:

| Branch | Phenomenon | English counterpart? | Predicted |
|---|---|---|---|
| `mutation_*` | Initial consonant mutation (soft/nasal/aspirate) | None at all | Erodes |
| `prep_*` | Conjugated prepositions ("arna i" vs. "ar fi") | None -- English prepositions never conjugate | Erodes |
| `numeral_*` | Singular noun after a numeral ("tri chi", not "tri cŵn") | None -- English uses a plural | Erodes (toward the English plural) |
| `plural_*` | Plural noun after "rhai" ("rhai llyfrau") | Yes -- English "some books" | Resists erosion |
| `quantifier_*` | Plural noun after a quantifier + "o" ("llawer o lyfrau", "lot o bethau") | Yes -- English "lots of books" | Resists erosion |

`numeral_*`, `plural_*` and `quantifier_*` all measure the same thing --
the singular/plural tag on the noun, from the lexicon -- so tagger errors
hit both sides equally and their rates compare directly.

**Why two "English agrees" branches.** "rhai" + noun is rare in speech (a
handful of contexts across 60 Siarad conversations), so on its own it
can't carry the resist-erosion prediction, so `quantifier_*` (added
2026-09-28) gives that side a second, far more frequent structure. Both
test one rule -- a count noun after a quantity word is plural, in Welsh
and in English -- so since 2026-10-02 the analyzer also pools them as
**"Controls combined"**, and the quantifier branch now includes the
partitives "un o'r", "rhai o'r", "dau o'r", "y rhan fwyaf o'r". The
English loan quantifier "lot o" is counted but flagged
(`is_loan_quantifier`), so results can be reported with and without it.

**Data sources.** The Bangor Siarad corpus (`corpus_siarad.py`) is the
main dataset: 40 hours of informal conversation between 153 Welsh-English
bilinguals aged roughly 18-72, recorded 2005-08 at their homes or
workplaces with no researcher present, with human transcripts, speaker
age/sex, and human code-switch tags. Because the transcripts are human-
made, Siarad findings carry no speech-recognition error. Every Siarad
detection row carries a `speaker` column that joins to that conversation's
`speakers_*.csv`, so erosion can be modelled against speaker age as well as
formality. The other sources, from most to least formal:

| Source | What it is | Transcript |
|---|---|---|
| News articles (`news_text.py`) | ~500 Newyddion S4C / Y Cymro articles, split into narration, quoted speech and spokesperson statements. Narration is edited text, so its "erosion" rate is the **detector's noise floor**. BBC articles are excluded. | Text, no ASR |
| YouTube / podcasts | Newyddion S4C news and interviews, Hansh (youth TV), Haclediad (casual podcast); 10-minute samples | Whisper |
| CorCenCC spoken (`corpus_corcencc.py`) | 1,331 recordings from the National Corpus of Contemporary Welsh (Knight et al. 2020, CC-BY-SA): conversations, meetings, broadcasts, with genre and learner/L1 status per recording | Human |
| Patagonia (`corpus_siarad.py`) | Bangor's 43 Welsh-Spanish conversations from Argentina, speakers aged 8-96, heritage speakers and learners included | Human |

`corpus_formality.py` replaces a hand-assigned formal/informal/casual
channel label with a grounded, continuous formality score computed from
the transcript itself (POS-class balance, filler rate, lexical diversity)
-- per video, and for Siarad also per speaker -- so "does erosion track
formality" is an actual regression against a measured variable, not a
comparison between three researcher-picked buckets.

## Current status (2026-10-04)

Data freeze **2026-10-08**; report due 2026-10-30. The numbers below come
from the 2026-10-02/03 analyzer runs and will be replaced after the freeze.

**Mutation erosion by corpus -- a formality ladder:** news narration 1.6%
(noise floor) < news statements 2.8% < news quotes 3.9% < Newyddion S4C
17.6% < CorCenCC 19.4% < Siarad 20.7% < Haclediad 25.8% (one run, raw) <
Patagonia 27.8% < Hansh 35.0%.

**By formality third** (least / middle / most formal): mutation 21.5 /
17.8 / 4.5%; prepositions 10.2 / 11.0 / 4.2%; numeral 0.9 / 2.0 / 0.7%;
controls combined (rhai + quantifier) 1.4 / 0.7 / 0%.

**Formality effect inside one corpus** (`within_corpus_slopes.csv`; odds
ratio of erosion per 10 F-points, below 1 = less erosion when more formal):
mutation in CorCenCC 0.65 (95% CI 0.59-0.71) per recording and 0.62
(0.58-0.67) per speaker; YouTube 0.27; Siarad and Patagonia not
significant (each spans only ~12-24 F-points). Prepositions in CorCenCC:
1.11, not significant.

What this supports so far:

- Structures Welsh shares with English (rhai, quantifier + o) barely erode
  (~0-1%); structures English lacks (mutation, conjugated prepositions)
  do (~6-28%).
- Numeral + singular noun does **not** erode (~1%) -- that prediction
  fails. A likely reason: speakers use "tri o blant" (numeral + o +
  plural), which the branch doesn't count.
- Mutation erosion tracks formality, also within one corpus and per
  speaker, so it isn't just a difference between corpora. Preposition
  erosion doesn't track formality (region may matter more: northern Siarad
  ~6%, mostly southern CorCenCC ~11%).
- Numeral, rhai and quantifier have too few erosions for a formality slope
  -- their rates are precise, their slopes can't be estimated.

**Validation.** Three rounds of precision audit (`audit_sample.py`, see
**The pieces**). Round 3 (2026-10-03), flagged mutation erosions that held
up: Siarad 22/25, CorCenCC 18/23, YouTube 18/25 (the YouTube losses are
Whisper errors; round 2 put the YouTube rate at ~11% corrected vs 17.6%
raw). CorCenCC rhai "erosions" were mostly false (2/17), and the
preposition/numeral samples exposed southern "ŷn ni" and decade words
("degau") -- all fixed 2026-10-03, to be re-sampled. Quantifier singulars
are never counted automatically; by hand, real slips were 1/30 (Siarad),
2/30 (CorCenCC), 1/13 (YouTube).

**Statistics.** Rates with Wilson 95% intervals. Formality curves are
logistic (binomial, weighted by contexts, band widened by quasi-binomial
dispersion because contexts from one video aren't independent), reported
as an odds ratio per 10 F-points, and fitted only with at least 10
erosions (`MIN_FIT_EVENTS`). Planned instead of a mixed-effects model: a
speaker bootstrap (resampling speakers) for per-speaker intervals.

**Audit round 4 (2026-10-05)**, flagged erosions that held up:
mutation Siarad 22/25, CorCenCC 20/24, YouTube 18/23; prepositions Siarad
19/20, CorCenCC 11/19 (southern bod forms read as prepositions), YouTube
2/8 (speech-recognition errors); numeral CorCenCC 15/20 (4 of the 5 false
ones -- Bible "dau Brenhinoedd", "ugain miloedd", a song title -- are now
excluded in code), Siarad 3/4.

**Before the freeze (Oct 8):** fresh window -> **4 -> y** (quantifier and
numeral fixes of 2026-10-05) -> **6 -> d** (quantifier census) -> **6 -> c**
-> **6 -> b**; judge the census and the redrawn numeral sample; then **5**.
Copy `lemma_cache.json` with the frozen data.

**Every output row carries `pipeline_version`** (a hash of the detection
code), so rows from different rule versions can never be silently pooled;
`corpus_analyzer.py` reports which versions it read. Code that can't change
a row -- source discovery and the completion email in `corpus_ops.py` --
sits between `# >>> NOT VERSIONED` / `# <<< NOT VERSIONED` markers (or on a
line ending `# unversioned`) and is left out of the hash, so editing it
doesn't mark the corpus as stale.

## What this actually does, in one paragraph

Point it at Welsh-language audio (a local MP3, or a YouTube channel
you've added to a queue). It transcribes with Whisper, tags every word
with two independent part-of-speech taggers (the Cysill API and a local
spaCy parser), and runs all five detection branches above over the same
tagged word stream -- each producing its own findings (mutation-, prep-,
numeral-, plural- and quantifier-specific CSVs) from the same transcript
pass. Siarad transcripts skip the Whisper step and go straight to tagging. Where YouTube
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
   without hitting the Cysill API. For a form with several possible
   lemmas, it keeps the readings that match the mutation and POS Cysill's
   POS reply gave that word in context (the method Techiaith recommended),
   so the one-word-per-request lemmatizer is only a last resort. The
   pipeline continues without it, falling back to Cysill/spaCy/simplemma.
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
menu 2 (process the queue) and More tools -> d (local MP3s) respect it if set at launch.

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
| `GMAIL_SENDER` / `GMAIL_APP_PASSWORD` / `NOTIFY_RECIPIENT` | No | Enables an HTML completion-email summary (all five branches first, then run stats, per-video results and the mutation breakdown -- see `corpus_ops.py` under **The pieces**) after menu 2 (Process the queue) or More tools -> d (Analyze local MP3 files, when saved) finish. Needs a Gmail account with 2-Step Verification and an App Password (Google Account -> Security -> 2-Step Verification -> App passwords) -- not your normal Gmail password. `NOTIFY_RECIPIENT` defaults to `GMAIL_SENDER` (i.e. emails yourself) if unset. Leave all three blank to disable notifications entirely; the pipeline runs exactly the same either way, it just skips the email at the end. |

Never commit real values for any of these -- keep them in your actual
environment/shell profile/`.env`, not in source files.

## The pieces

`welsh_pipeline.py` is the only thing you run directly. Everything else is
either a module it imports, or a standalone companion tool you can also
run on its own from the command line.

**Naming convention:** files are prefixed by which branch owns them --
`mutation_*.py` (consonant mutation), `prep_*.py` (conjugated
prepositions), `numeral_*.py` (numeral + singular noun), `plural_*.py`
("rhai" + plural noun), `quantifier_*.py` (quantifier + "o" + plural noun). Unprefixed files are
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
  discovery isn't mutation-specific). The list is curated for a wide
  spread of formality (news bulletins and lectures -> politics panel and
  interviews -> youth TV -> casual podcasts), so per-video F-scores span
  the range the erosion-vs-formality scatter needs. Each entry has a
  `speech` label (`scripted` / `mixed` / `spontaneous`), since formal
  speech is usually also scripted, and a scripted bulletin's mutations
  are an editor's, not a speaker's. A new source needs only its URL:
  `corpus_ops._detect_source_type()` routes YouTube to yt-dlp, RSS/Atom
  feeds to the direct feed parser, and Y Pod JSON caches to their
  adapter; a bare YouTube channel URL is pointed at its `/videos` tab so
  Shorts stay out.
- `corpus_ops.py` -- file I/O and orchestration: the video queue,
  processed/failed logs, audio download, and the `analyze()`/
  `analyze_phrase()` functions that run Whisper plus *every* detection
  branch (mutation, prep, numeral, plural, quantifier) over one video or phrase end to end,
  plus the completion email. The email (since 2026-10-04) opens with all
  five branches -- eroded/contexts, rate and 95% interval per branch, plus
  the combined control -- and a per-video table with one column per
  branch; the subject line reads e.g. `Mut 25.8% | Prep 2/18 | Num 0/3 |
  rhai - | Quant 0/3` (a branch under 20 contexts shows counts, not a
  percentage). `run_branch_stats()` reads them back from the run's own
  CSVs, the same way `corpus_analyzer.py` does; the older mutation
  breakdowns follow under "Mutation branch: detail". Also owns `TRANSCRIBE_PRESETS` (see
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
  in it. POS requests pack whole utterances into ~2,700-character chunks.
  The API has an hourly limit (HTTP 429, `Retry-After: 3600`); Techiaith
  raised this project's allowance on request (2026-09-27), and lemmas now
  come mostly from the POS reply rather than the one-word lemmatizer (see
  **Setup** step 5), so a full Siarad pass fits in far fewer requests.
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

**Quantifier branch (has an English counterpart -- predicted to resist
erosion):**

- `quantifier_engine.py` -- after a quantifier + "o" ("llawer o",
  "digon o", "gormod o", "mwy o", "faint o", "lot o"...), checks whether
  the noun is plural. English "lots of / enough / too many / more" needs a
  plural count noun too, so this is the second "English agrees" branch.
  Same gates and same lexicon-only number as `plural_*`/`numeral_*`. Mass
  nouns are excluded by lemma ("llawer o waith" = "a lot of work",
  singular in English too), and so are nouns the lexicon never lists in
  the plural (no singular/plural choice existed); both exclusions remove
  singular and plural outcomes alike. "llawer o bobl" (collective) counts
  as correct.
- `quantifier_tables.py` -- quantifier forms (with mutated forms), the
  mass-noun lemma list, and the collective-noun exception. The mass-noun
  list is the part most likely to need additions after the precision
  audit.

**Coding decisions (apply to every branch).** First fixed on 2026-09-26,
extended after each precision audit, and written into the tables/engines,
so every run applies them the same way:

| Question | Decision |
|---|---|
| Place names after a trigger ("i Bangor") | **Excluded** (2026-09-29) -- fluent speakers often leave them unmutated so they stay recognisable |
| Person names, and any capitalised target | **Excluded** (2026-10-01) -- edited news leaves names unmutated ("gan Deian"), so name mutation is optional even in the standard. Months, days, languages etc. stay in (`CAPITALISED_COMMON_WORDS`) |
| Fixed expressions ("wrth gwrs", "i gyd", "ei gilydd", "a ballu") | **Excluded** -- frozen forms, not live mutation (`FIXED_EXPRESSIONS`) |
| English loanwords | **Counted as Welsh once spelled with Welsh orthography**; English-spelled words are code-switches and never scored |
| Digits as triggers ("100 diwrnod") | **Excluded** -- the spoken numeral is unknown |
| `mutation_mismatch` rows | **Not counted** (outside `EVALUABLE_STATUSES`) |
| "rhai" + mass noun ("rhai amser") | **Excluded** -- English "some time" is singular too, so it isn't an "English agrees" context (`MASS_NOUNS`) |
| "y rhai" = "the ones", "rhai" + pawb/gyd/tro... | **Excluded** -- pronoun or adverbial "rhai", not a quantity word before a noun (`NOT_RHAI_NOUNS`) |
| Numeral + cant/mil/miliwn/degau | **Excluded** -- a bigger number, singular in English too (`NUMBER_WORDS`) |
| Singular after a quantifier + "o" | **Judged by hand, every one** (the quantifier census, 2026-10-05) -- most are mass or degree readings ("llawer o wahaniaeth", "gormod o babi") or "one FROM" ("un o'r ardal"); only a candidate judged a real slip ("un o'r bachgen") counts as erosion |
| Names and titles after a numeral or quantifier ("dau Brenhinoedd" = 2 Kings, "un o Sir Fôn") | **Excluded** (2026-10-05) -- not a counted noun; nationalities ("rhai o'r Cymry") stay in |
| What a Siarad / CorCenCC transcript says | **Taken as what was said** -- transcribers mark reduced articles ("(y)r"), so an unwritten article is a real omission |

Two principles run through every exclusion:

- **Skips are symmetric.** A context that isn't a real mutation/number
  environment is skipped *whatever the target's form* -- mutated or not --
  so no exclusion can push the erosion rate up or down on its own.
  Examples: tag particles ("yn te" = isn't it?), "yn bore" (= "yn y bore",
  article dropped), numeral sequences ("tri pedwar o'gloch"), letters that
  can't mutate ("y lôn"), the pronoun "i" after a first-person verb
  ("dw i meddwl"), the pronoun "o" before a verb-noun ("oedd o mynd"),
  demonstrative "yna" ("y pnawn yna"), relative "lle" ("o lle mae..."),
  and targets that are forms of "bod" or conjugated prepositions ("sy
  gynno"). "mae"/"ydy"/"oes" and "mai"/"taw" are not triggers at all:
  none of them mutates the next word.
- **Evidence is conservative.** Noun number for the numeral/rhai branches
  must come from the lexicon, not spaCy's guess; a target any tagger marks
  as already mutated is `erosion_unverified`, not erosion (unless that
  tagger misread the word's part of speech); and Siarad words the
  transcriber marked `[?]` (unsure what was said) are never scored.

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

  Resuming is automatic: a conversation counts as done only if its saved
  output was made on the **current** `pipeline_version` **with** Cysill
  coverage (>= 90% of words), so after a rule change, or a run that
  Cysill's rate limit cut short, just run the same command again. `--limit
  N` processes only the next N not yet done; `--redo` forces everything.
  If Cysill trips mid-conversation, that conversation is not saved and the
  run stops with a message, rather than writing half-tagged output.

  CHAT conventions handled: "(y)n"-style elided letters are restored; a
  bare `@s` marks the utterance's other language (English, in a Welsh
  utterance); untagged words are judged by spelling; retracings,
  fragments and `xxx` are dropped; `[?]` words are kept but never scored
  (see **Coding decisions**). The same script reads the Patagonia CHAT
  files (menu 3 -> p).
- `corpus_corcencc.py` -- the spoken part of CorCenCC (menu 3 -> c; cite
  Knight et al. 2020, doi:10.17035/d.2020.0119878310). Reads
  `corpus_data.txt` and retags the human transcript with Cysill like every
  other source; CorCenCC's own CyTag tags are only used to spot speaker
  codes, foreign words, proper nouns and elided words. Anonymisation
  placeholders (lleoliad, enwb1...) become capitalised name parts, which no
  branch measures; clitics ("ti 'n", "a 'i") are glued back on ("ti'n") so
  they split the way Whisper text does; transcription marks ([aneglur],
  [-], "+", <events>) break the word stream like a comma. Speaker codes
  can't be linked to contributors, so learner/L1 status, genre and
  "scripted" are recorded per recording. Not versioned: after a change
  that alters the word stream, bump `READER_MARKER` and every recording is
  redone.
- `news_text.py` -- the news articles as text (menu 3 -> n): each article
  becomes narration, quoted speech and spokesperson statements, three
  separate documents (sources `news-narration`, `news-quote`,
  `news-statement`). Narration is edited, so whatever it "erodes" is
  detector error -- the noise floor every other rate is read against.
  Newyddion S4C and Y Cymro only; BBC articles are skipped.

**Cross-branch analysis:**

- `corpus_analyzer.py` -- reads every mutations CSV you've ever produced,
  merges them (only the latest run per video/conversation), and generates
  corpus-wide figures and a text summary, including the
  erosion-vs-formality-score regression (see below), and all five branches
  side by side against formality, per video and (Siarad) per speaker
  (`branches_vs_formality.png`, `branches_by_formality_band.png`,
  `speaker_branches_*.png`). Since 2026-10-02 the curves are logistic, not
  straight lines (a straight line predicted -10% erosion at the formal
  end), and `within_corpus_slopes.png/.csv` (+ `speaker_` version) fits
  the formality slope inside each corpus separately, so the corpus can't
  stand in for formality -- the confound-controlled headline. A slope
  needs at least 10 erosions (`MIN_FIT_EVENTS`); below that it is printed
  as "not fitted" instead of drawn as a meaningless 0.002-3000 interval.
  Rates use Wilson 95% intervals, and "Controls combined" pools rhai and
  the quantifier branch.
- `audit_sample.py` -- the precision audit (menu 6). Draws a seeded
  (`SEED = 20261001`), stratified random sample of erosion and correct
  rows per branch and source into `analysis/audit2/<source>/`, with the
  sentence around each row. Each row gets a verdict -- ok / wrong /
  invalid / unsure -- in the audit file or in a
  `claude_verdicts_<branch>.csv` sidecar (merged on the next draw). Scoring
  gives each label's precision and an audit-corrected erosion rate
  (bootstrap interval). Redrawing keeps every verdict whose row is
  unchanged. Round 1 (2026-09-29, older code) is in `analysis/audit/`.
  `census()` (menu 6 -> d) lists every quantifier singular candidate for
  the quantifier census; `corpus_analyzer.py` counts the ones judged
  `slip` as erosion and prints how many are still unjudged, so the
  quantifier rate is measured rather than 0% by construction.
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
  reflects. For Siarad it also scores each speaker separately
  (`speaker_formality.csv`; speakers with under 100 tagged words are
  written but left out of the figures).
- `youtube_access.py` -- the single shared coordinator for *every*
  yt-dlp request in a run (captions, audio downloads, channel discovery
  alike): paced inter-request timing, escalating jittered backoff on a
  429, and a rate-limit cooldown that persists to disk and survives a
  process restart, not just the rest of the current run -- since a
  YouTube 429 block has been reported to last on the order of hours,
  longer than any single run.

## Workflow

`welsh_pipeline.py` has one menu, in the order the work is usually done.
Every action returns to it; `q` quits. The top line shows the queue size
and the pipeline version.

The usual paths:
- **New YouTube data:** 1 (find) -> 2 (process) -> 5 (numbers).
- **Siarad / Patagonia / CorCenCC / news:** 3 -> 5.
- **After changing detection code:** open a **fresh** window (the menu
  refuses to run detection when the code on disk differs from what it
  loaded), then 4 (update results) -> 5 -> 6 -> c (redraw the audit
  samples whose rows changed).

1. **Find new YouTube videos** -- scans `CURATED_CHANNELS` (in `corpus_io.py`) for anything new, adds it to `video_queue.json`. Doesn't download or transcribe. The batch is filled round-robin across sources (one item from each in turn), so no single channel fills it. Items under `MIN_EPISODE_SECONDS` (3 min: Shorts, trailers, promos, news stings) are skipped here when the source publishes a duration, and after download when it doesn't.
2. **Process the queue** -- transcribes, then runs all five detection branches (mutation, prep, numeral, plural, quantifier) and caption-corroborates the mutation findings. Failed videos retry up to 3x (`failed_videos.json`). Sends a completion email if configured. The Whisper model size is asked once per session.
3. **Process transcripts (Siarad / Patagonia / CorCenCC / news)** -- `s`/`p`: `corpus_siarad.py` on `<data folder>/siarad` or `/patagonia` (or a path you type). `c`: `corpus_corcencc.py` on the unzipped CorCenCC download in `<data folder>/corcencc` (~1,300 recordings; an overnight run). `n`: `news_text.py` on the scraped Welsh news articles (found automatically: `NEWS_CORPUS_DIR`, else `Desktop\news_corpus` -- the scraper's fixed output folder -- plus older `Desktop\NEWS\news_corpus` / data-folder copies, in both `raw/cy/` and legacy `cy/raw/` layouts; Newyddion S4C and Y Cymro only, BBC skipped) -- each article becomes two text documents, narration (source `news-narration`) and quoted speech (`news-quote`), the no-ASR formal baseline and detector noise floor. Anything already done on the current version is skipped, so after a Cysill limit stop just choose it again.
4. **Update all results after a code change** -- `mutation_rerun_rules.py` on every processed folder, all five branches, from the saved tagging: no re-transcription, no Cysill calls, manual reviews kept. `y` applies now, `p` previews only, `n` cancels. (Per-branch / per-folder / per-trigger filters: `python mutation_rerun_rules.py --help`.)
5. **Show the numbers** -- the corpus analyzer: every branch's rate per corpus, figures, the erosion-vs-formality curves and the within-corpus slopes. Reloads the analyzer code each time, so an edit to `corpus_analyzer.py` doesn't need a restart. Also `python corpus_analyzer.py`.
6. **Precision audit** -- `audit_sample.py` for Siarad, YouTube and (once processed) CorCenCC: `a` draw/refresh the samples, `b` score the verdicts, `c` redraw only the samples whose rows changed (their old verdicts are dropped), `d` the quantifier census -- every singular after a quantifier, all sources, in `analysis/quantifier_census/census_quantifier.csv` (verdicts `slip` / `no` / `unsure`; run it again to pull in verdicts from `claude_verdicts_census.csv`). Verdicts follow their row (video, timestamp, word), not their position in the sample, so a redraw can't move a verdict onto a different row.
7. **More tools**
   - **a** Manage the queue -- show (with per-channel counts), remove by number, add a URL, clear.
   - **b** Review flagged erosions by hand -- `mutation_manual_editing.py` (`--help` for its filters).
   - **c** Test a Welsh phrase -- no audio. Runs all five branches on the typed phrase; output in `phrase_tests/`. The menu loads the code once at startup -- restart it after changing any `.py` file.
   - **d** Analyze local MP3 files -- asks **save** (real corpus) or **preview** (`mp3_previews/`, never marked processed).

All three processing scripts stop at startup if the Bangor lexicon can't be loaded -- detection depends on it, and a run without it isn't comparable with the others.

## Where your data ends up

Everything lives under `WELSH_ANALYSIS_DIR` (or `~/welsh_analysis` if you
didn't set that variable). The full folder layout is created up front on
every run (not lazily, one folder at a time, as each menu option first
needs it), so you'll see all of it even before you've used every menu
option:

```
WELSH_ANALYSIS_DIR/
├── runs/<stamp>_<label>/                   one folder per pipeline run, labelled by what it
│   │                                          processed (e.g. 20260928_081602_siarad-20,
│   │                                          ..._youtube-3, ..._local-mp3)
│   ├── research_summary_<stamp>.csv        session-level summaries, written by
│   ├── erosion_by_trigger_type_<stamp>.csv   menu 2 (process the queue) itself,
│   ├── erosion_by_rule_<stamp>.csv           not menu 5 (the analyzer)
│   └── <slug>/                             one folder per video (or Siarad_<file>), from _video_slug()
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
│       ├── numeral_mutations_<stamp>_<slug>.csv        numeral + singular-noun branch findings
│       ├── plural_mutations_<stamp>_<slug>.csv         "rhai" + plural-noun branch findings
│       ├── quantifier_mutations_<stamp>_<slug>.csv     quantifier + "o" + plural-noun branch findings
│       ├── speakers_<stamp>_<slug>.csv                 Siarad only: speaker age, sex, notes, word counts
│       ├── utterances_<stamp>_<slug>.csv               Siarad only: raw + cleaned utterances, %gls/%eng
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
├── test_audio/                             drop local MP3s here for More tools -> d
├── analysis/                               menu 5's output: merged_mutations.csv,
│                                              utterance_export.csv, video_formality.csv and
│                                              speaker_formality.csv (corpus_formality.py),
│                                              within_corpus_slopes.csv (+ speaker_ version)
│                                              and asr_divergence.csv
│   ├── figures/                              chart images, including erosion_vs_formality.png,
│   │                                          within_corpus_slopes.png and erosion_vs_codeswitch.png
│   ├── audit/                                precision audit round 1 (2026-09-29, older code)
│   └── audit2/<source>/                      menu 6's samples and verdicts (siarad, youtube, corcencc)
├── siarad/, patagonia/, corcencc/          the transcript corpora menu 3 reads
├── phrase_tests/                           More tools -> c's ad-hoc "test a Welsh phrase"
│                                              output -- deliberately kept outside runs/ so it
│                                              never gets swept into the real corpus by
│                                              menu 5 or mutation_rerun_rules.py
├── mp3_previews/                           More tools -> d's output when you choose "preview" instead
│                                              of "save" -- same quarantine idea as phrase_tests/,
│                                              never marked processed, never swept into the corpus
├── video_queue.json                        pending videos (menu 1 adds, menu 2 consumes)
├── processed_videos.json                   videos already handled by menu 2
├── processed_local_mp3s.json               local files already saved via More tools -> d
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

`prep_mutations_*.csv`, `numeral_mutations_*.csv`,
`plural_mutations_*.csv` and `quantifier_mutations_*.csv` are each a
single file per video (no original/corroborated split -- caption
corroboration is mutation-specific only). `corpus_analyzer.py` reads all
five branches (latest run per video) for the cross-branch figures. Every row in every file carries
`pipeline_version`.

## State and recovery

Queue, cache, and output files all live in `WELSH_ANALYSIS_DIR`. Queue
failures are recorded in `failed_videos.json` and retried up to three
times before being marked processed anyway. Successfully processed local
MP3s are recorded in `processed_local_mp3s.json`, keyed by filename and
file size -- editing or replacing a file makes it eligible for
reprocessing automatically.

Never store API keys or email passwords in source files. Use environment
variables instead.