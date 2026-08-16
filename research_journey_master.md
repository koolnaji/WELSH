# Research Journey — Contact-Induced Language Change Project
*Merged reference document for the October final paper. Phase structure and narrative follow the more detailed account (worked with longer, in more depth); supplementary technical detail folded in where it doesn't conflict. Where the two accounts disagreed, the more detailed account's version is what's stated here.*

*Last updated: 2026-08-13, against a direct check of commit history on all three repos (WELSH, SCRAPER, TATAR).*

---

## Phase 0 — Origins (June 2026)

The project began smaller than it ended up: an Icelandic corpus-building exercise using a Playwright-based scraper against RÚV, originally intended for LancsBox analysis, built while genuinely new to Python. Stanza handled Icelandic lemmatization. Early problems here — boilerplate detection, duplicate detection by URL-derived filenames — were solved in ways simple enough to later generalize.

In parallel, the Welsh side started independently: a study of soft/nasal/aspirate consonant mutation erosion across S4C YouTube channels, contrasting formal and informal registers. The core research question was set here and never changed afterward: **does consonant mutation erode under sustained English-contact pressure, and can register be used as a proxy for that pressure?**

Initial pipeline architecture: `welsh_pipeline.py` + `corpus_ops.py` + `mutation_engine.py` + `corpus_analyzer.py`, using a three-layer confidence system from the start (Cysill API, spaCy `cy_ud_cy_ccg`, rule-based heuristics) — this three-way corroboration design is the single architectural decision that shaped almost everything later, both the good (real methodological rigor) and the bad (a lot of debugging effort spent keeping three independent, differently-reliable systems honest with each other).

An early, telling finding: after an overnight batch run and a round of bug fixes (the `yn`+verb-noun aspectual-marker exemption, `ll`/`rh` exemptions), soft mutation erosion dropped from ~61% to ~52.7%, and aspirate dropped from ~88% to ~74%. **Worth remembering for the methods section**: the earliest numbers this project ever produced were substantially wrong, and got less wrong through iterative bug-fixing — a useful, honest data point about why the later validation work mattered so much.

---

## Phase 1 — Foundational accuracy work (mid-July)

This is where the project shifted from "does it run" to "is it right." Most bugs here were the same *shape* of error recurring in different places: **two things that look comparable but structurally aren't, silently compared as if they were.**

- A clause-boundary bug affecting all 8 adjacency-based detection layers: punctuation was stripped without being recorded, collapsing clause-internal word lists into flat sequences and letting mutation triggers fire across clause boundaries that should have blocked them.
- The `tagger_agreement` column used a truthy guard that made real Cysill corroboration on erosion rows *structurally impossible* — all 68 erosion rows in one test run were stamped `heuristic_only` despite 100% real Cysill agreement underneath. This exact bug (labeling/vocabulary mismatch hiding real corroboration) resurfaced in August in a different form (Phase 4) — a recurring failure mode worth naming explicitly in the paper, not a one-off.
- `radical_candidates_for_target()` was accepting Cysill's own lemma guess without validating it forward — confirmed live turning "digalon" into a bogus "igalon."
- A spaCy DOM (direct-object-mutation) rule was firing on verb-noun heads it shouldn't have — 65.7% of one rule's rows had been evaluated against the wrong head type before the `head_verbform` fix.
- The orthographic-hallucination filter had been accidentally deleted during a monolith-to-multi-file split, and diaereses in legitimate words like *cwmnïau* were being wrongly rejected before a dedicated legal-mark-pairs table was added.

Also produced the first substantial extension of `manual_editing.py` (jump/pick navigation, review logging, search) — the human-in-the-loop spot-checking tool everything downstream ultimately has to answer to. Also fixed in this general window, on the mutation-table side: `mae'n` contraction-splitting, plural feminine noun false triggers (`extract_number_from_spacy()`), `sy`/`sydd` added to the trigger dictionary, and `un` added to the ll/rh soft-mutation exemption set — with that exemption logic wired into every relevant processing layer, not just one.

---

## Phase 2 — Infrastructure and the Tatar arm begins (mid-to-late July)

Two things happened roughly in parallel.

**Pipeline infrastructure hardened**: folder-structure consistency fixes (transcripts and mutations were nesting inconsistently — later settled on a `<stamp>/<slug>/` nesting scheme, with `phrase_tests/` and `mp3_previews/` quarantining ad-hoc output from the real corpus), a silent-empty-output gap closed (a video producing zero segments used to report success with nothing to show for it), and `corpus_ops.py`'s notification email rebuilt to actually summarize a run instead of just confirming it finished. Note: `BASE_DIR` defaults to `Path.home()/"welsh_analysis"`, which resolves to the Windows profile root rather than Documents — worth setting `WELSH_ANALYSIS_DIR` explicitly to avoid losing output.

**The Tatar arm was designed**, not yet built: a three-layer detection architecture for finite subordinate clause substitution (UD dependency parse primary, case-marking secondary, Russian-subordinator lexical scan tertiary) was worked out on paper before any code existed. `subordination_tables.py` and `subordination_engine.py` were scaffolded and smoke-tested by the end of this window. Bugs confirmed and patched in this codebase over time: Cyrillic subordinator substring overcounting (fixed via proper `str.count()` handling), a dead multi-word subordinator match in the parse-corroboration path, a match-ordering bug where `"что"` was preempting the longer `"потому что"`, and a duplicate accusative allomorph in `tables.py`.

---

## Phase 3 — The multilingual scraper matures, Cysill strain begins (late July–early August)

The Icelandic-only scraper from Phase 0 was generalized into a real multilingual pipeline (Icelandic, English, Welsh, Norwegian, Swedish, Danish, German, French, Spanish), consolidated from a sprawling set of files down to six: `icelandic_text_extractor.py` (main), `language_detection.py` (merged judge + voice logic), `boilerplate.py` (merged detector + review logic), `boilerplate_patterns.py`, `gemini_retry.py`, `term_ui.py`, `inspect_selectors.py`.

The language-identification system is genuinely well-designed: a weighted-vote panel — **GlotLID (1.0), OpenLID-v3 (1.0), lingua (0.6, covering Dutch and Irish)** — with CLD3/gcld3 removed entirely, a `site_hint` signal from URL/domain patterns that can corroborate but never single-handedly override a real judge, and an abstain-first design (≥55% weighted-confidence share from ≥2 corroborating judges to confirm; otherwise routed to `disputed/` if judges disagree or `unknown/` if none has an opinion) distinguishing "nobody voted" from "the voters disagree." Gemini was demoted to an optional tiebreak on disputed articles only (`--detect-language-llm`), never called per-article by default — partly because Gemini API rate limits are per-project-per-model, not per-key, so key rotation only helps across separate Google Cloud projects. This is the piece of infrastructure that later made the loanword-integration branch (Phase 6) possible without starting from scratch.

`SITE_OVERRIDES` (extraction data only, kept separate from `LANGUAGE_OVERRIDES`'s `expected_language`/`language_lock`/`listing_urls`) currently locks 14 domains: `ruv.is` (is), `bbc.com` incl. Cymru Fyw (en, cy), `apnews.com` (en), `theguardian.com` (en, excluding `/live/*`), `tagesschau.de` (de), `dw.com` (de), `nrk.no` (nb), `nos.nl` (nl), `ansa.it` (it), `rtve.es` (es), `rtp.pt` (pt), `svt.se` (sv), `lefigaro.fr` (fr), `france24.com` (fr) — with `reuters.com` and `lemonde.fr` deliberately excluded as paywalled/bot-gated, and `independent.co.uk` currently missing despite earlier confirmation it should be there (needs re-verification before the paper cites domain coverage). Boilerplate detection uses an `is_suspicious()` filter combining an edge-specific short-paragraph check with a whole-document ratio check, batched at 15 articles per Gemini call. GlotLID/OpenLID-v3 (~3GB combined) need HuggingFace access and run better on the GPU PCbang machine than the Windows PC. A per-domain "cold-start trust heuristic" and confidence-score auto-classify system were explicitly proposed and declined — a deliberate scope boundary, not an oversight.

The boilerplate-candidate review loop was also hardened with an explicit human-confirmation gate: `offer_end_of_run_boilerplate_review()` in `boilerplate.py` runs automatically at the end of any scrape with `--detect-boilerplate` on, and offers exactly three choices — review candidates now (routes into `run_review()`), save them for a later interactive pass, or discard this run's unreviewed candidates (which marks them `reviewed+discarded` rather than deleting them, preserving an audit trail of what the LLM flagged and a human declined to even look at). Nothing reaches `boilerplate_patterns.py` without an explicit `y` inside `run_review()`. Non-interactive/scripted runs (stdin not a TTY) default safely to "save for later" rather than hanging on a prompt. This closes the loop that was, earlier in the project, a named risk: an LLM-detected pattern silently mutating the boilerplate filter without a human ever seeing it.

The `inspect_selectors.py` diagnostic tool matured alongside this: auto-recommends selectors, checks suspect inner elements per candidate, prefers a clean tighter container if it retains ≥70% of the largest candidate's character count (`CLEAN_ALTERNATIVE_MIN_RATIO`), and `common_class_token()` collapses junk class-name variants to a shared token. Two bugs fixed: SVG elements crashing `inner_text()` (fixed with a `text_content()` fallback) and nested elements being double-counted across nesting levels (fixed with `dedupe_nested_matches()`). Architecture discipline maintained throughout this whole arc: every selector/pattern addition requires a confirmed real example — never guessed.

This is also, in retrospect, the window where the Cysill API problems that dominated August actually began — even though they weren't diagnosed until later. A code comment added on August 3rd documents, after the fact, that the project's own API key had been handed a full-hour `Retry-After: 3600` lockout by Cysill at some point in this stretch. **This matters for the paper's limitations section**: the instrumentation that would have caught this in real time didn't exist yet when it happened.

---

## Phase 4 — The corroboration-integrity fixes (early August)

A cluster of bugs were found and fixed here that materially change how much you can trust any erosion-rate number computed *before* this point:

- `LEMMA_CACHE` had accumulated a large fraction of null entries (877 of 1,796 — 49%) from transient failures being cached as if confirmed empty answers. A `CYSILL_CALL_FAILED` sentinel was introduced to distinguish "we asked and got nothing" from "we asked and confirmed there's nothing," and a one-off purge script cleaned the poisoned cache.
- English code-switch words were being sent to Cysill *before* the code-switch check ran, confirmed live turning "gosh" into "cosh." Fixed by moving the check earlier in the pipeline.
- The denominator used for the headline erosion-rate figure was found to silently include phantom-mutation rows, understating erosion by roughly 18 percentage points. `EVALUABLE_STATUSES` was centralized into one place (`mutation_tables.py`) specifically so this kind of drift couldn't happen twice. **Any batch processed before this fix is not directly comparable to any batch processed after it** — state this explicitly wherever the paper reports a trend across the project's timeline.
- The offline Bangor lexicon (Techiaith's own CC0 wordform data, ~497k entries, toggled via `BANGOR_LEXICON_PATH`) was integrated as a pre-Cysill resolution step — both for lemma lookup (96.3% unambiguous coverage) and, more conservatively, for whole-chunk POS resolution when every word in a chunk resolves without ambiguity.

Also around this window: transcription presets (`fast`/`balanced`/`accurate`) were added to control Whisper `beam_size` and temperature fallback depth, trading speed for accuracy as needed per batch.

---

## Phase 5 — The August "mayhem": Cysill and YouTube instability (early-to-mid August)

This is the stretch that ate the most real debugging time, and it's worth reconstructing carefully for the paper because the *diagnostic process itself* — not just the eventual fixes — says something about the reliability of automated corpus pipelines built on third-party, unpaid-maintainer services.

**What actually happened, in order:**

1. A batch run tripped Cysill's circuit breaker on its very first video — 5 consecutive chunk failures (15 total failed HTTP attempts), all clean 15-second read timeouts with no 429, no 5xx, no variance at all.
2. Because `reset_cysill_circuit_breaker()` was only called once per menu choice (not per video), that single early trip silently blinded the *entire rest of the batch* to real Cysill corroboration — nine videos ran heuristic-only without anyone intending that.
3. Cross-referencing this against Techiaith's own per-API-key usage dashboard showed something more specific than "the API is down": 15 real, key-authenticated `OK` calls *did* go through before the failures started, and the dashboard showed **zero** across OK/Blocked/Would-block for the following days — meaning the failures weren't a quota rejection (which would show as `Blocked`), they were the service going silently unresponsive mid-session.
4. Separately, the offline Bangor lexicon's whole-chunk skip logic (Phase 4) had already reduced *legitimate, working* Cysill call volume by close to 100% starting right around when this same instability began — two independent, coincidentally-overlapping causes for "why did Cysill traffic drop," only one of which was actually a problem.
5. A parallel YouTube-side rate limit appeared on the caption-fetch endpoint (HTTP 429), traced via commit history to caption fetching having no retry/backoff at all, combined with `list_available_tracks()` being called redundantly up to three times per video with zero pacing between requests — exactly the request pattern that trips a rate limiter.

**What this phase actually built, methodologically:**

- Cookie-based yt-dlp authentication (`YTDLP_COOKIES_FROM_BROWSER` / `YTDLP_COOKIES_FILE`), since anonymous requests hit YouTube's rate limit far sooner than authenticated ones.
- `youtube_access.py`: a single shared coordinator for *every* yt-dlp request in a run — captions and audio downloads alike — with real inter-request pacing, escalating jittered backoff, and a rate-limit cooldown that **persists to disk and survives a process restart**, not just the rest of the current run. This superseded a first attempt at a caption-only circuit breaker, which was built, found redundant with the shared coordinator, and deliberately removed once the shared version existed — normal engineering iteration, worth mentioning as such rather than hiding the false start. (A separate, earlier merge mishap is worth flagging for your own records even if it doesn't belong in the paper: two local working directories got edited independently during this circuit-breaker work and one silently overwrote the other's changes before the correct merged version was re-delivered.)
- `download_audio()` changed to key files by video ID rather than title (avoiding collisions) and to skip re-downloading already-present files. A DNS-failure backoff (5s/15s/30s schedule, separate from the generic retry) was also drafted for `download_audio()` after a confirmed transient DNS failure on `media24.fireside.fm` — **this patch was not confirmed pushed to the repo as of last check and should be verified before assuming it's live.**
- A Windows/OneDrive file-locking issue (`WinError 5` on the atomic checkpoint rename) was diagnosed as a sync-client race condition, not a code bug — the working directory sitting inside a OneDrive-synced folder means the sync client can transiently lock a freshly-written file before the rename completes.
- A `UnicodeDecodeError: cp949` crash was separately diagnosed as a Windows Korean-locale issue in yt-dlp's ffmpeg subprocess, fixed via `PYTHONUTF8=1`.
- On the audio-sampling side: `sample_audio_window()` was refactored to extract the true sample window plus 20s padding on each side, returning `(path, bounds)`, with `_shift_and_trim_padded_segments()` shifting Whisper's clip-relative timestamps back to true-video time and trimming edge segments correctly — fixing a `video_duration_seconds` bug in `analyze()` that had been using padded clip length instead of true sample window length. Net effect: sentence-boundary context is now preserved correctly at chunk edges, which matters for mutation-trigger detection since triggers can span clause boundaries.
- A separate transcription-model finding, methodologically significant on its own: `techiaith/whisper-large-v3-ft-verbatim-cy-en-ct2`, forced into `language='cy'`, produced a **stable, repeatable fluent-English hallucination** across independent smoke tests on genuine Welsh + fragmented-shouting audio — distinct from ordinary noise-driven hallucination, since it's a reliable misfire that would cause the `lang_recognition` filter to silently drop genuine Welsh audio (misclassified as low-recognition because it hallucinated fluent English). Resolved by switching to `faster-whisper large-v3-turbo` as the working transcription model. Worth a limitations note: a naive pipeline could have systematically under-sampled exactly the kind of informal/shouted speech most relevant to erosion research.

Corpus sources were also expanded in this general window: **Haclediad** (casual tech/culture Welsh podcast, CC BY-NC-SA, RSS-ingested via Fireside), **Beti a'i Phobol** (BBC Radio Cymru interview-register podcast), and the **Siarad corpus** (Bangor University CHAT-format fieldwork corpus, ingested as pre-transcribed text, bypassing Whisper entirely — a distinct, non-ASR data path worth flagging in the methods section).

`rerun_rules.py` was added to let mutation detection re-run on already-transcribed videos without re-transcribing or re-hitting Cysill; it defaults to writing a `*_rerun_candidate.csv` safety file, with `--commit` merging into the real CSV while never overwriting `manual_reviewed=True` rows — a deliberate human-in-the-loop safeguard.

---

## Phase 6 — Reframing: from two case studies to one theory (mid-August)

This is the point where the project's *argument*, not just its pipeline, matured. Up to this point, the Welsh and Tatar arms were two parallel case studies connected mainly by shared method. The unifying hypothesis crystallized here:

> Under sustained contact pressure, a minority language first loses the specific structures its dominant contact language has no equivalent machinery for — not structure in general, but the structures that are typologically foreign to the dominant language.

This reframes Welsh mutation erosion (English has no mutation system at all — no scaffolding) and Tatar finite-subordination substitution (Russian *does* have working finite-subordination machinery to substitute in) as two directional predictions from **one** theory, rather than two unrelated findings. It also motivated a third, lighter-weight branch: **loanword integration**, using the already-built multilingual scraper, asking a synchronic version of the same question — when a new word enters from the dominant language, does it get folded into the typologically-foreign machinery (mutation, vowel harmony) or does it route around it? This is also where **Tatar vowel harmony was deliberately repositioned** — dropped as a standalone flagship phenomenon (vowel harmony loss in loanwords is typologically universal, not Russian-contact-specific, so it can't carry the argument alone) and reintroduced instead as a metric *within* the loanword branch, where that same universality stops being a liability and becomes exactly what's being measured.

Literature grounding was assembled for all three arms in this phase — Bob Morris Jones as the closest Welsh methodological precedent, White & Roberts on speaker expectations of mutation, the Nanai-Russian contact paper as a near-exact template for the Tatar mechanism in a different language pair, and the Uralic finiteness/word-order paper for theoretical scaffolding on *why* contact should push toward finite subordination in the first place.

---

## Phase 7 — Where things stand now (as of mid-August)

- **Welsh arm**: most mature. Pipeline architecturally stable, three known bugs queued for a fix pass (the `all_three`/`parser+cysill+heuristic` tagger-agreement label mismatch, the checkpoint file-lock issue, the circuit-breaker reset scope). Erosion data exists but needs reprocessing/re-validation now that the Cysill reliability issues are understood and partially fixed.
- **Tatar arm**: architecture built (`subordination_tables.py`/`subordination_engine.py`, smoke-tested), not yet run at corpus scale. Whisper's actual reliability on Tatar speech is unconfirmed — flagged as the single biggest risk to this arm's timeline. (Separately, code size for comparison: `engine.py`+`tables.py`+`subordination_engine.py`+`subordination_tables.py`+`tagging.py` runs to roughly 1,300 lines total, versus ~2,500 in the Welsh mutation engine alone — a reasonable reflection of vowel harmony being a local, lemma-level operation versus mutation's syntax-dependent, trigger-site-based one. Production infrastructure that exists on the Welsh side — API client, circuit breaker, manual-review CLI, checkpointing, corpus normalization — is still absent here and would need building out if the Tatar arm is run at real corpus scale before October.)
- **Loanword arm**: designed, not yet built. Welsh side is the smaller lift (repurpose the existing code-switch detector rather than discard its output); Tatar side needs a new Russian-loanword detector from scratch.
- **`limitations.txt`** exists and is version-controlled in the repo — a running, evidence-linked account of exactly what the data can and can't support, written specifically so it can be lifted into the paper's limitations section rather than reconstructed from memory later.
- Repo (`github.com/koolnaji/WELSH`) is in a clean, self-consistent state as of the last verified pull — no dead code contradicting live code, test suite passing. (Standard caveat given how many times both repos have been restructured: re-clone fresh before pulling exact file/line detail into the paper, rather than trusting either summary's file layout as current.)

---

## Before October — open items worth tracking

- Re-run/re-validate Welsh erosion figures under the fixed corroboration logic before treating any pre-fix batch as final.
- Resolve the `disputed`-classification question (Section 2.4 of `limitations.txt`) — decide whether to report erosion rates with and without disputed rows as a sensitivity check, and do it before the numbers are locked in.
- Confirm Whisper's actual Tatar ASR quality before committing further build time to that arm.
- Build the Tatar Russian-loanword detector, if the loanword branch is going in the final paper at all — worth deciding explicitly rather than defaulting into it by momentum.
- Decide how much of the August diagnostic saga belongs in the methods section versus an appendix — genuinely good material for demonstrating rigor, but also a lot of material; the paper needs a clear line between "here's how I validated my instrument" and "here's a debugging diary."
- Verify two loose ends before citing them as fact: the DNS backoff patch's actual repo status, and whether `independent.co.uk` has been re-added to `SITE_OVERRIDES`.
- Decide whether the language-ID judge panel needs its own accuracy benchmark against ground-truth data before the paper cites it, or whether the panel's design rationale (weighted voting, abstain-first routing) is the intended evidence on its own — no benchmark currently exists in the repo either way.
- Write a real `README.md` for the TATAR repo (currently a placeholder) before October, given how much argumentative weight the subordination arm carries in the Phase 6 reframing.

---

## Phase 8 — Repo check-in: 2026-08-13

*Verified directly against commit history on `github.com/koolnaji/WELSH`, `SCRAPER`, and `TATAR` as of this date. Dates below are commit dates, not conversation dates.*

**WELSH — active, several commits Aug 11–13:**
- **2026-08-11**: `fetch_captions.py` gained retry-with-backoff on YouTube 429s and its own circuit breaker (mirroring `cysill_client.py`'s pattern) that disables captions for the rest of a run after a few consecutive whole-video caption failures — framed explicitly as "captions are corroboration-only," so a tripped caption breaker never affects transcription/mutation output itself.
- **2026-08-12**: `youtube_access.py` added — the shared, persistent coordinator for *every* yt-dlp request (captions and audio alike) described in Phase 5, plus `test_youtube_access.py`. **`limitations.txt` was committed this same day** (241 lines, five numbered sections — see below, this is a major addition). `download_audio()` was also given the same cookie-auth fix (`yt_dlp_cookie_opts()`) that captions already had, closing a gap where audio downloads were still going out fully anonymous and hitting the same YouTube rate-limit family. `.env.example` documented the cookie auth variables and README gained a full explanation of the caption circuit-breaker behavior and cookie setup, including a noted Windows/Chrome-specific yt-dlp cookie-decryption failure (yt-dlp#15401) with `firefox` or `YTDLP_COOKIES_FILE` as the workarounds.
- **2026-08-13, 09:08**: a commit **reverted** `corpus_ops.py`, `mutation_engine.py`, and `welsh_pipeline.py` away from the `youtube_access.py` shared-coordinator design back to an older, file-local implementation (title-only file keying instead of video-ID keying, a manual inline retry loop instead of `youtube_call()`, and removal of the caption-circuit-breaker reset call in `welsh_pipeline.py`) — this appears to be the local-working-directory mixup already on record.
- **2026-08-13, 16:50**: a same-day follow-up commit **restored** the shared-coordinator design in all three files. Current `HEAD` confirmed clean: `fetch_captions.py` now delegates entirely to `youtube_access`'s shared coordinator rather than keeping its own per-file circuit breaker (that per-file breaker was deliberately removed as redundant, matching the Phase 5 account), and `download_audio()` correctly keys files by video ID with `youtube_call()`-managed retries.
- **`limitations.txt` (new, 2026-08-12, ~241 lines)** — this is the single most citable new asset for the paper. Five sections: (1) software/dependency drift — notably the Welsh dependency parser (`cy_ud_cy_ccg`) was last validated against spaCy 3.5.0 and is a confirmed, unmaintained ceiling against the pipeline's spaCy 3.8.14; (2) corpus_analyzer.py erosion-rate computation limitations — including the specific, quantified finding that dual-tagger coverage was ~100% for Hansh vs. ~7% for Haclediad in one batch, traced directly to a circuit-breaker trip early in that run, and that the dominant detection rule (`word_trigger`, >1,200 rows in one casual-register batch) has the *weakest* cross-tagger agreement of any rule (56.3%) despite supplying the bulk of all mutation observations; (3) ASR limitations, including that Whisper's Welsh accuracy has only been spot-checked, not validated at corpus scale, against genuine Bangor Siarad recordings; (4) filesystem/operational reliability (OneDrive locking, order-dependent data loss under rate-limiting); (5) an explicit scope note that the Tatar arm will need its own parallel limitations review before its results are reported alongside Welsh findings. Section 2.4 directly confirms the earlier open item: disputed rows (>1/3 of evaluable rows in one batch) are currently counted on equal footing with fully-corroborated rows in the erosion-rate denominator, with no sensitivity analysis yet run.

**SCRAPER — no activity since 2026-08-03.** Last commits on that date were a cleanup: several Welsh-pipeline files (`welsh_pipeline.py`, `mutation_engine.py`, `mutation_tables.py`, `corpus_ops.py`, `fetch_captions.py`, `cysill_client.py`, `manual_editing.py`, `spacy_tagging.py`, `rerun_rules.py`, `validate_against_chat.py`) that had apparently been mistakenly uploaded into this repo were deleted. Current file set (7 files: `icelandic_text_extractor.py`, `language_detection.py`, `boilerplate.py`, `boilerplate_patterns.py`, `gemini_retry.py`, `term_ui.py`, `inspect_selectors.py`) matches the Phase 3 description exactly, with nothing added toward the loanword branch yet.

A few smaller things worth logging from a direct read of the current file contents, beyond what commit-history alone shows:
- **The three BBC-specific patterns flagged earlier as possibly missing (reporter-credit line, iPlayer promo, "get in touch" CTA) are confirmed present** in the current `boilerplate_patterns.py`, all dated to a 2026-07-30 candidate review (`Additional reporting by Bob Howard.`, the Panorama/iPlayer promo line, and the Essex "story suggestion" CTA plus its matching social-follow line). That earlier open question is resolved — this was an out-of-sync local copy at the time, not an intentional omission.
- **`_compile_alternation`/`_NEVER_MATCHES` is still duplicated**, even after the file consolidation — once in `boilerplate.py`, once in `icelandic_text_extractor.py` (used for `_END_OF_ARTICLE_RE`/`_BOILERPLATE_RE` compilation there). The consolidation collapsed most of the old cross-file duplication but not this specific pair.
- **The two separate `y/n/q`-style prompt implementations also persisted through consolidation**: `boilerplate.py` still has its own local `_prompt()`, while `icelandic_text_extractor.py` has a separate `_prompt_yes_no()` built on a `UserQuit`/`GoBack` exception pair. Given the file count dropped from a sprawling set down to seven specifically to reduce this kind of drift, these two holdouts are worth a deliberate look before the paper cites the codebase as fully unified — though the original circular-import concern that kept them apart may still be the reason.
- **`gemini_retry.py`'s module docstring still refers to `boilerplate_detector.py` and `language_voices.py` by name** — their pre-merge filenames, now `boilerplate.py` and `language_detection.py` respectively. Cosmetic, but worth a find-and-replace pass so a future reader (or reviewer) isn't sent looking for files that no longer exist.
- **No standalone accuracy assessment of the language-ID judge panel appears to exist yet** — no benchmark script, no ground-truth comparison output, nothing under a name like `judge_accuracy` or `language_eval`. The panel's *design* is well-documented (weighted voting, abstain-first routing — see Phase 3 above), but an empirical accuracy number for GlotLID/OpenLID-v3/lingua against known-language test data doesn't appear to have been produced yet. Worth deciding explicitly whether this is still owed for the paper or whether the panel's design rationale is meant to stand in for it.

**TATAR — no activity since 2026-07-29.** Only two commits exist total, both that date (initial commit + one upload), totaling 1,367 lines across `engine.py`, `smoke_test.py`, `subordination_engine.py`, `subordination_tables.py`, `tables.py`, `tagging.py`. This confirms the Phase 7 status is still accurate as of today — architecture built and smoke-tested, but nothing further has shipped since. No production infrastructure (API client, circuit breaker, manual-review CLI, checkpointing) has been added, and no corpus-scale run has happened.

One documentation gap worth flagging directly: **`README.md` in this repo is currently a 7-byte placeholder — literally just `# TATAR`.** Given how much of the project's actual argument now runs through this arm (the Phase 6 reframing hinges on the Welsh/Tatar directional-prediction pair), having zero written documentation of the subordination engine's design here — versus the reasonably thorough treatment WELSH gets in its own README and `limitations.txt` — is a real asymmetry. Worth writing even a short design note here before October, both so the paper has something stable to cite and so the reasoning documented in Phase 2/6 of this file isn't the only place it exists.

**Net effect on the "before October" list**: `limitations.txt` existing and being this detailed changes the calculus on the "diagnostic saga vs. methods section" question — much of that decision is now already made *for* you, since the document is written in citable, thesis-ready prose. The Tatar-arm and loanword-arm open items are unchanged; no repo progress has closed either of them.

---

## Phase 9 — PO token diagnosis and fix: 2026-08-15/16

A new, mechanically distinct YouTube-side failure appeared during a
queue run: `download_audio()` throwing a persistent `HTTP Error 403:
Forbidden` on the actual media fetch for a video (Tŷ Cŵn | Heledd a
Mared) whose captions had already fetched successfully (45 segments).
The existing `cleanup_incomplete_video_dirs()` machinery from Phase 5
handled the failure correctly -- removed the 3 orphaned partial
folders, requeued the video -- confirming that error-handling path is
sound and not itself in need of a fix.

**Root cause, confirmed via web research against current (2026)
yt-dlp/YouTube community tracking**: this is a different failure shape
from both the n-signature/JS-challenge issue (Deno, resolved earlier)
and the caption rate-limiting issue (Section 1.4/`youtube_access.py`,
resolved in Phase 5). Captions/metadata succeeding while the media URL
403s is the signature of YouTube's PO (Proof-of-Origin) token
enforcement -- a cryptographic attestation now required on
`googlevideo.com` media URLs for most player clients, industry-wide and
apparently still evolving in shape through 2026 per the yt-dlp issue
tracker. Neither `yt_dlp_cookie_opts()` (cookie auth) nor
`remote_components: ["ejs:github"]` (n-signature solving) addresses
this -- it's a third, independent YouTube-side requirement.

**Fix**: installed `bgutil-pot` (jim60105/bgutil-ytdlp-pot-provider-rs,
a Rust PO-token provider) plus its matching yt-dlp plugin. Two
sub-issues surfaced and were resolved during setup, both worth noting
as they're likely to recur on any future machine setup:
- The provider's default HTTP server bind (`[::]:4416`, IPv6 wildcard)
  did not match yt-dlp's default `base_url`
  (`http://127.0.0.1:4416`, IPv4) -- `curl` against `127.0.0.1`
  failed while `curl` against `[::1]` succeeded, isolating this as an
  IPv4/IPv6 mismatch rather than a firewall block. Resolved by binding
  the server explicitly with `--host 127.0.0.1`.
- Once the provider was reachable and generating real tokens (confirmed
  in both yt-dlp's debug output and the provider's own server log), a
  test download still 403'd -- traced to a leftover `.part` file from
  an earlier failed attempt triggering a resumed byte-range request
  against a freshly re-signed URL, a second, independent 403 cause.
  `yt-dlp --no-continue` against the same URL downloaded cleanly,
  confirming the PO-token fix itself was working and isolating the
  resume-403 as a distinct, separate issue (now documented in
  `limitations.txt` Section 4.3).

**Operational note for future runs**: the PO-token provider is a
persistent background process (`bgutil-pot server --host 127.0.0.1`)
that must be running for the full duration of any pipeline run, same
operational category as Deno needing to be installed and unblocked --
it is not started automatically by any code in this repo. `README.md`
and `limitations.txt` (Sections 1.4, 4.3) both updated same-day to
reflect this as a new, load-bearing, unpaid-maintainer-project
dependency, on top of the two already named in `limitations.txt`
Section 1.3.

**Not yet done (as of 2026-08-16)**: retry of the actual triggering
video (Tŷ Cŵn | Heledd a Mared) through the real pipeline queue, and a
sweep of `AUDIO_DIR` for other stale `.part` files from earlier failed
attempts that could reintroduce the resume-403 on other queued
retries.

### Phase 9 continued — client mismatch, low-view-video asymmetry, mweb fix, nightly report: 2026-08-16/17

With the PO-token infrastructure confirmed working (server generating
and caching real tokens per video ID, visible in its own log), 403s
persisted on real corpus videos even though the rickroll test video
downloaded cleanly. Diagnosed in stages via direct `yt-dlp -v` testing
against the actual failing video (F1gPkVIHzig):

1. First hypothesis (client/token mismatch) was directly observed:
   yt-dlp fetched player data via the `android_vr` client but
   requested a PO token scoped to `web_safari` -- download started
   (real bytes flowing at full speed) then 403'd at 20.5% of the file.
2. Forcing a single consistent client (`--extractor-args
   "youtube:player_client=mweb"`) produced a correctly-matched
   token/client pair (confirmed in debug output) but the SAME video
   still 403'd, this time at 20.6% -- functionally the same byte
   offset on a separately-signed URL. This ruled out the
   client-mismatch theory as sufficient explanation on its own, since
   a real mismatch fix didn't change the outcome.
3. `tv` and `web` clients were tested as alternatives and rejected:
   `tv` hit DRM-protected formats, `web` got SABR'd into image-only
   formats -- neither produced a real media download attempt at all
   regardless of the 403 question.
4. Control test: the SAME `mweb` command against the earlier rickroll
   test video (`dQw4w9WgXcQ`) downloaded cleanly. This isolated the
   failure to something about THIS video (or its CDN routing)
   specifically, not a systemic client/token/network problem --
   directly contradicting an initial local-network-interference theory
   (Windows Defender/antivirus resetting long connections) that had
   been raised as the leading candidate given the near-identical
   cutoff percentage across two different signed URLs.
5. Landed on a structural explanation: heavily-viewed videos are
   pre-warmed across nearly all CDN edge nodes and read by YouTube's
   anti-bot heuristics as an established-legitimate traffic pattern;
   this project's actual corpus (low-view S4C/Welsh-channel uploads)
   does not get that treatment and is disproportionately exposed to
   both edge-node flakiness and stricter automated-traffic scrutiny --
   independent of PO-token validity. This is consistent with, and now
   documented as an extension of, the general YouTube-side fragility
   already in `limitations.txt` Section 1.4.

**Fix applied to `corpus_ops.py`/`youtube_access.py`** (not yet
confirmed against a full batch run): `download_audio()`'s `ydl_opts`
now forces `player_client=["mweb"]` via `extractor_args`, and
`youtube_access.call()` gained an optional `min_interval` parameter so
audio downloads specifically can use a slower pace
(`AUDIO_DOWNLOAD_MIN_INTERVAL = 6.0`) than the shared default used for
lighter caption/metadata calls, without changing that shared default.
Both changes are pragmatic mitigations for a heuristic YouTube-side
behavior, not a guaranteed fix -- see `limitations.txt` Section 1.4 for
the full caveat.

**UNCONFIRMED, same session**: switching yt-dlp from stable to a
nightly build was separately reported to stop producing 403s in quick
manual testing. Plausible (nightly ships extractor fixes faster than
stable), but NOT yet isolated against the `mweb`/pacing fix above --
still open whether nightly alone is sufficient, whether all three
mitigations are independently necessary, or whether nightly's build
just happened to test-download successfully by chance on a small
sample. `requirements.txt` updated with a comment flagging this and
the reproducibility risk of an unpinned nightly build (ties into the
existing environment-drift concern in `limitations.txt` Section 1.2).

**Not yet done**: isolation test (nightly alone, `mweb`/pacing
reverted, against a batch of previously-failing low-view videos) to
determine which mitigation(s) are actually load-bearing; full-batch
confirmation of whichever combination is kept; the Tŷ Cŵn retry and
`AUDIO_DIR` stale-file sweep noted above, still outstanding.

---

## Suggested Paper Structure Mapping

- **Methods — Data Collection**: Phases 0, 3, 5 (corpus sources, multilingual scraper, YouTube/audio pipeline)
- **Methods — Mutation/Erosion Detection**: Phases 0–1, 4 (three-layer corroboration, corroboration-integrity fixes)
- **Methods — Language Identification**: Phase 3 (weighted-vote panel, abstain-first design)
- **Theory / Framing**: Phase 6 (unifying contact-pressure hypothesis, literature grounding)
- **Limitations**: Phase 3 (Cysill lockout), Phase 4 (denominator/cache issues, pre/post-fix comparability), Phase 5 (Whisper hallucination, diagnostic saga), Phase 7 (Tatar ASR risk), Phase 9 (PO-token dependency, resumed-download 403s, client-mismatch/low-view-video CDN asymmetry, unpinned-nightly reproducibility risk)
- **Cross-linguistic Discussion**: Phase 7 (Welsh vs. Tatar structural/scale comparison), Phase 6 (loanword branch rationale)
- **Appendix / Reproducibility**: `SITE_OVERRIDES` table, consensus thresholds, model versions, `limitations.txt`