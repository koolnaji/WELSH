"""
welsh_pipeline.py
===========
Thin entry point / CLI. This is the only file you run directly. It owns
the interactive menu loop (analyze local MP3s, discover videos, process
the queue, test a phrase, manage the queue, run the corpus analyzer,
manually review mutations) and session-level state like the loaded
Whisper model -- all the actual linguistics and corpus I/O live in
mutation_engine.py and corpus_ops.py.

For queue-processed (YouTube) videos, caption corroboration
(fetch_captions.py) now runs automatically per video: captions are fetched
before transcription, and the corroboration pass runs immediately after
that video's mutations are written -- no separate manual step. Local MP3
batches (choice 1) skip this, since local files have no YouTube video ID
to fetch captions against.

A Gmail notification email is sent when video processing (options 1 and 3
only) finishes -- see corpus_ops.send_notification_email for setup.
"""
import os
import torch
import yt_dlp
from tqdm import tqdm
from faster_whisper import WhisperModel
import spacy_tagging

from mutation_engine import (
    BASE_DIR, LOCAL_MP3_DIR, PREVIEW_DIR,
    ensure_dirs, run_stamp, _video_slug, _preview_video_slug, load_spacy,
    reset_cysill_circuit_breaker, load_lemma_cache, save_lemma_cache,
    load_bangor_lexicon,
)
from corpus_ops import (
    load_queue, save_queue, load_processed, save_processed,
    load_failed, record_failure, clear_failure,
    load_local_processed, save_local_processed,
    discover_new_videos, prompt_channel_selection, download_audio,
    analyze, analyze_phrase, save_analysis_outputs, generate_research_summary,
    send_notification_email, build_email_body, channel_display_name,
    TRANSCRIBE_PRESETS, DEFAULT_TRANSCRIBE_PRESET,
)
import corpus_analyzer
# PATCH: fetch_captions is now a normal top-level import rather than
# lazy-imported inside a menu branch -- it's no longer an optional manual
# step (former option 8), it's part of the standard choice-3 video loop
# now (see below), so it needs to be available every time that loop runs.
import fetch_captions

import csv
import pandas as pd
import time

# ========================= MAIN =========================

# ========================= QUEUE MANAGER =========================
def manage_queue():
    """Interactive queue management submenu."""
    while True:
        queue = load_queue()
        print(f"\n── Queue Manager ({len(queue)} videos) ──────────────────")
        print("  a = Show queue")
        print("  b = Remove videos by index")
        print("  c = Remove videos by channel/source")
        print("  d = Filter queue to specific channels only")
        print("  e = Clear entire queue")
        print("  f = Add a single YouTube URL")
        print("  g = Show channel breakdown")
        print("  q = Back to main menu")
        cmd = input("Choice: ").strip().lower()

        if cmd == "q":
            print("Returning to main menu...")
            break

        elif cmd == "a":
            if not queue:
                print("  Queue is empty.")
            else:
                print(f"\n  {'#':<5} {'Title':<50} {'Register':<12} {'Source'}")
                print("  " + "-" * 95)
                for i, v in enumerate(queue):
                    src = v.get("source", "?")
                    # PATCH: was a raw .split("/")[-1] slug of the source
                    # URL -- fine for YouTube ("HanshS4C") but produced
                    # unreadable output for non-YouTube feeds/caches
                    # ("rss", "podcast-cwins.json?v=1"). channel_display_name()
                    # prefers the CURATED_CHANNELS entry's explicit "name".
                    src_short = channel_display_name(src)
                    title = v.get("title", v.get("id", "?"))[:48]
                    reg   = v.get("channel_register", "unverified")
                    print(f"  {i:<5} {title:<50} {reg:<12} {src_short}")

        elif cmd == "b":
            if not queue:
                print("  Queue is empty.")
                continue
            # show queue first
            print(f"\n  {'#':<5} {'Title':<55} {'Source'}")
            print("  " + "-" * 85)
            for i, v in enumerate(queue):
                src_short = channel_display_name(v.get("source", "?"))
                title = v.get("title", v.get("id", "?"))[:53]
                print(f"  {i:<5} {title:<55} {src_short}")
            raw = input("\n  Enter indices to remove (e.g. 0 3 5-8 12): ").strip()
            if not raw:
                continue
            to_remove = set()
            for part in raw.split():
                if "-" in part:
                    try:
                        a, b = part.split("-")
                        to_remove.update(range(int(a), int(b) + 1))
                    except ValueError:
                        print(f"  Skipping invalid range: {part}")
                else:
                    try:
                        to_remove.add(int(part))
                    except ValueError:
                        print(f"  Skipping invalid index: {part}")
            new_queue = [v for i, v in enumerate(queue) if i not in to_remove]
            removed = len(queue) - len(new_queue)
            save_queue(new_queue)
            print(f"  Removed {removed} video(s). Queue now has {len(new_queue)}.")

        elif cmd == "c":
            if not queue:
                print("  Queue is empty.")
                continue
            # show unique sources
            sources = sorted(set(v.get("source", "?") for v in queue))
            print("\n  Sources in queue:")
            for i, s in enumerate(sources):
                count = sum(1 for v in queue if v.get("source") == s)
                # PATCH: show the friendly name (channel_display_name)
                # alongside the raw URL -- `sources` itself still holds
                # the real URL, since that's what the removal logic
                # below matches against.
                print(f"  {i}  {channel_display_name(s):<32} {s}  ({count} videos)")
            raw = input("  Enter source indices to remove: ").strip()
            if not raw:
                continue
            try:
                idxs = {int(x) for x in raw.split()}
                remove_sources = {sources[i] for i in idxs if i < len(sources)}
            except ValueError:
                print("  Invalid input.")
                continue
            new_queue = [v for v in queue if v.get("source") not in remove_sources]
            removed = len(queue) - len(new_queue)
            save_queue(new_queue)
            print(f"  Removed {removed} video(s) from {len(remove_sources)} source(s). "
                  f"Queue now has {len(new_queue)}.")

        elif cmd == "d":
            if not queue:
                print("  Queue is empty.")
                continue
            sources = sorted(set(v.get("source", "?") for v in queue))
            print("\n  Sources in queue:")
            for i, s in enumerate(sources):
                count = sum(1 for v in queue if v.get("source") == s)
                # PATCH: show the friendly name (channel_display_name)
                # alongside the raw URL -- `sources` itself still holds
                # the real URL, since that's what the removal logic
                # below matches against.
                print(f"  {i}  {channel_display_name(s):<32} {s}  ({count} videos)")
            raw = input("  Enter indices of sources to KEEP (all others will be removed): ").strip()
            if not raw:
                continue
            try:
                idxs = {int(x) for x in raw.split()}
                keep_sources = {sources[i] for i in idxs if i < len(sources)}
            except ValueError:
                print("  Invalid input.")
                continue
            new_queue = [v for v in queue if v.get("source") in keep_sources]
            removed = len(queue) - len(new_queue)
            save_queue(new_queue)
            print(f"  Kept {len(keep_sources)} source(s), removed {removed} video(s). "
                  f"Queue now has {len(new_queue)}.")

        elif cmd == "e":
            if not queue:
                print("  Queue is already empty.")
                continue
            confirm = input(f"  Really clear all {len(queue)} videos? (yes/no): ").strip().lower()
            if confirm == "yes":
                save_queue([])
                print("  Queue cleared.")
            else:
                print("  Cancelled.")

        elif cmd == "f":
            url = input("  YouTube URL or video ID: ").strip()
            if not url:
                continue
            # normalise to full URL
            if not url.startswith("http"):
                url = f"https://www.youtube.com/watch?v={url}"
            # PATCH: register isn't inferrable from a one-off URL the way it
            # is from CURATED_CHANNELS, so ask explicitly. Defaults to
            # "unverified" (not "informal"/"formal") if left blank, so it
            # doesn't silently slide into either side of a register
            # comparison without a deliberate choice.
            reg = input("  Channel register (formal/informal/unverified) [unverified]: ").strip().lower() or "unverified"
            if reg not in ("formal", "informal", "unverified"):
                print(f"  Unrecognized register '{reg}', defaulting to 'unverified'.")
                reg = "unverified"
            opts = {"quiet": True, "no_warnings": True}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                vid_id = info.get("id")
                title  = info.get("title", "unknown")
                source = info.get("channel_url") or url
                existing_ids = {v["id"] for v in queue}
                if vid_id in existing_ids:
                    print(f"  Already in queue: {title}")
                else:
                    queue.append({"id": vid_id, "url": url,
                                  "title": title, "source": source,
                                  "channel_register": reg})
                    save_queue(queue)
                    print(f"  Added: {title} [{reg}]")
            except Exception as e:
                print(f"  Failed to fetch video info: {e}")

        elif cmd == "g":
            if not queue:
                print("  Queue is empty.")
                continue
            from collections import Counter
            sources = [(channel_display_name(v.get("source", "?")), v.get("channel_register", "unverified"))
                       for v in queue]
            counts  = Counter(sources).most_common()
            print(f"\n  {'Channel':<40} {'Register':<12} {'Videos':>7}")
            print("  " + "-" * 61)
            for (s, reg), n in counts:
                print(f"  {s:<40} {reg:<12} {n:>7}")
            print("  " + "-" * 61)
            print(f"  {'TOTAL':<40} {'':<12} {len(queue):>7}")
            # PATCH: register-level rollup, useful for eyeballing corpus
            # balance between formal/informal before a big processing run
            reg_counts = Counter(v.get("channel_register", "unverified") for v in queue)
            print("\n  By register:")
            for reg, n in reg_counts.most_common():
                print(f"    {reg:<12} {n:>7}")

        else:
            print("  Unknown command.")

def main(preset=None):
    ensure_dirs()
    # PATCH: resolves and confirms the transcription preset once, up front,
    # rather than silently inside analyze() on the first call. preset=None
    # (no CLI arg given) falls back to DEFAULT_TRANSCRIBE_PRESET, which is
    # "accurate" -- byte-identical to the settings that were hardcoded here
    # before presets existed, so not passing a preset changes nothing.
    active_preset = preset or DEFAULT_TRANSCRIBE_PRESET
    if active_preset not in TRANSCRIBE_PRESETS:
        print(f"⚠️  Unknown preset {active_preset!r} -- using {DEFAULT_TRANSCRIBE_PRESET!r}. "
              f"Valid presets: {', '.join(TRANSCRIBE_PRESETS)}")
        active_preset = DEFAULT_TRANSCRIBE_PRESET
    print(f"Transcription preset: {active_preset}")
    # PATCH: load persisted lemma cache so prior API calls aren't repeated
    load_lemma_cache()
    print("Loading Welsh dependency parser...")
    load_spacy()
    # PATCH: optional offline lookup that reduces live Cysill traffic --
    # see load_bangor_lexicon()'s own docstring for why this is called
    # once here rather than lazily. A missing file just means "run
    # exactly as before this existed", not a startup failure.
    load_bangor_lexicon()

    # PATCH: model is now session-scoped (loaded lazily, cached across menu
    # loops) instead of being re-prompted/re-loaded every single choice.
    model = None

    def _append(df, path, flags, idx):
        df.to_csv(path, mode="a", header=flags[idx], index=False,
                  encoding="utf-8-sig", quoting=1)
        flags[idx] = False

    current_model_size = None

    def _ensure_model():
        nonlocal model, current_model_size
        if model is not None:
            switch = input(f"  Current Whisper model: {current_model_size}. "
                           "Switch models? (y/N): ").strip().lower()
            if switch != "y":
                return model
        print()
        model_size = input("Enter model (small / medium / large-v3 / large-v3-turbo) "
                           "[default: small]: ").strip() or "small"
        print(f"Loading {model_size} model...")
        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        # PATCH: CTranslate2 was left to pick its own cpu_threads default
        # instead of being told this machine's actual logical core count.
        # Zero accuracy tradeoff (unlike beam_size/temperature) -- pure
        # parallelism. 0 on the CUDA path leaves GPU thread management to
        # CTranslate2 as before; this only changes CPU behavior.
        cpu_threads = os.cpu_count() or 4
        model = WhisperModel(model_size, device=device, compute_type=compute_type,
                              cpu_threads=cpu_threads if device == "cpu" else 0)
        current_model_size = model_size
        print("✅ Whisper model loaded!\n")
        return model

    # PATCH: the whole menu now lives inside a loop so every choice -- not
    # just option 5 -- returns to the menu afterward. Only an explicit "q"
    # (at either menu level) breaks out and ends the program.
    #
    # PATCH: the menu was flattened across 8 numbered options, which got
    # crowded and gave no sense of which options are related. Restructured
    # into three themed categories -- Queue & Processing, Analysis,
    # Testing -- each with its own sub-menu. Deliberately minimal-risk:
    # every original numbered option still maps to the exact same `choice`
    # value it always did (CATEGORY_MENUS below), so the entire if/elif
    # chain that follows (all the actual per-option logic) is completely
    # untouched -- only how a person navigates to a given `choice` value
    # changed, not what any `choice` value does.
    #
    # PATCH: category labels/descriptions and each sub-menu's options now
    # live in one data structure (CATEGORY_MENUS) instead of separate
    # hand-spaced print() calls -- the previous version's column alignment
    # drifted out of sync by eye (e.g. "Queue & Processing" vs "Analysis"
    # landing their descriptions one character apart) because nothing
    # actually computed a shared column width; it just hoped hand-counted
    # spaces matched. The "=" border below gives each menu a clearer visual
    # edge instead of blending into whatever printed just above it.
    #
    # NOTE: an earlier version of this also cleared the terminal between
    # menu transitions -- reverted (see PATCH note near _print_top_menu):
    # it swapped screens faster than a person could actually read the
    # previous one's output before it vanished.
    # PATCH: dropped the parenthetical descriptions after each label --
    # README.md now documents what every option does, so repeating it here
    # was redundant and (at typical half-screen terminal widths) is what
    # caused wrapping in the first place, e.g. option "e"'s description
    # wrapping mid-word onto a second line. Bare labels only; details live
    # in one place (the README) instead of two that can drift apart.
    CATEGORY_MENUS = {
        "1": {
            "title": "Queue & Processing",
            "options": [
                ("a", "Discover new videos"),
                ("b", "Process queue"),
                ("c", "Manage queue"),
                ("d", "Manually review mutations"),
                ("e", "Re-run mutation rule(s)"),
            ],
            "choice_map": {"a": "2", "b": "3", "c": "5", "d": "7", "e": "8"},
        },
        "2": {
            "title": "Analysis",
            "options": [
                ("a", "Run corpus analyzer"),
            ],
            "choice_map": {"a": "6"},
        },
        "3": {
            "title": "Testing",
            "options": [
                ("a", "Test a Welsh phrase"),
                ("b", "Analyze local MP3 files"),
            ],
            "choice_map": {"a": "4", "b": "1"},
        },
    }

    # PATCH: was 78 -- still wrapped on a half-screen-width terminal (the
    # explicit goal here), since a split/half window is commonly narrower
    # than that even before accounting for the terminal's own margins.
    # 60 comfortably fits a half-screen window at a normal font size.
    MENU_WIDTH = 60

    def _border():
        print("=" * MENU_WIDTH)

    def _print_top_menu():
        _border()
        for k, m in CATEGORY_MENUS.items():
            print(f"{k} = {m['title']}")
        print("q = Quit")
        _border()

    def _print_sub_menu(category):
        m = CATEGORY_MENUS[category]
        print(f"-- {m['title']} --")
        for key, label in m["options"]:
            print(f"  {key} = {label}")
        print("  q = Back to main menu")
        _border()

    while True:
        _print_top_menu()
        category = input("Enter 1, 2, 3, or q: ").strip().lower()

        if category == "q":
            print("Goodbye!")
            save_lemma_cache()
            return

        if category not in CATEGORY_MENUS:
            print("Unknown choice.")
            continue

        # PATCH: this used to fall all the way back out to the main
        # category menu after every single action -- annoying when doing
        # several related things in a row (e.g. discover, then process).
        # Now it loops back to THIS category's sub-menu instead; only an
        # explicit "q" here returns to the main menu. Everything from
        # here down to the end of the if/elif chain now lives one level
        # deeper inside this loop, so every existing "continue" inside a
        # branch already does the right thing automatically -- it
        # continues this sub-menu loop, not the outer one.
        while True:
            _print_sub_menu(category)
            sub_choice = input("Choice: ").strip().lower()

            if sub_choice == "q":
                break

            choice = CATEGORY_MENUS[category]["choice_map"].get(sub_choice)
            if choice is None:
                print("Unknown choice.")
                continue

            stamp = run_stamp()
            all_mutation_rows = []
            # PATCH: only set True for actual video processing (1, 3) -- not
            # discovery, phrase testing, queue management, or corpus_analyzer --
            # so the completion email only fires for the thing it was asked for.
            notify_on_completion = False
            # PATCH: run-level metadata for the completion email (elapsed time,
            # per-video success/failure) -- reset every loop iteration so a
            # later choice doesn't inherit an earlier run's numbers.
            run_type = None
            run_start_time = None
            videos_attempted = 0
            failed_videos = []

            if choice == "1":
                # PATCH: no caption corroboration here -- local files have no
                # YouTube video ID to fetch captions against. Corroboration
                # (see choice "3" below) only applies to queue-processed,
                # YouTube-sourced videos.
                mp3_files = list(LOCAL_MP3_DIR.glob("*.mp3")) if LOCAL_MP3_DIR.exists() else []
                if not mp3_files:
                    print("No local MP3 files found.")
                    save_lemma_cache()
                    continue
                local_processed = load_local_processed()
                pending_mp3_files = []
                for p in mp3_files:
                    stat = p.stat()
                    record = local_processed.get(str(p.resolve()))
                    if record and record.get("size") == stat.st_size and record.get("mtime_ns") == stat.st_mtime_ns:
                        continue
                    pending_mp3_files.append(p)
                if not pending_mp3_files:
                    print("All local MP3 files have already been processed. Change a file or clear processed_local_mp3s.json to rerun them.")
                    continue
                print(f"Found {len(mp3_files)} local MP3 file(s).")
                print(f"Processing {len(pending_mp3_files)} new or changed file(s).")
                # PATCH: register isn't inferrable from a local filename, so ask
                # once up front and tag the whole batch -- assumes test_audio/
                # is register-homogeneous per run. If you're mixing registers in
                # the same folder, run this twice with different subsets instead
                # of trusting one tag for all of them.
                # Three-tier register scale, most to least formal:
                # formal (e.g. BBC Radio Cymru) > informal (e.g. Hansh/S4C,
                # produced-but-informal content) > casual (fully spontaneous,
                # unscripted peer conversation -- e.g. podcasts like Haclediad).
                local_reg = input("  Channel register for this local batch "
                                  "(formal/informal/casual/unverified) [unverified], or q to cancel: "
                                  ).strip().lower() or "unverified"
                if local_reg == "q":
                    print("  Cancelled.")
                    save_lemma_cache()
                    continue
                if local_reg not in ("formal", "informal", "casual", "unverified"):
                    print(f"  Unrecognized register '{local_reg}', defaulting to 'unverified'.")
                    local_reg = "unverified"
                # PATCH: save-or-preview toggle. Lets you sanity-check a new or
                # unfamiliar audio source (e.g. how well Whisper handles a
                # genuinely spontaneous multi-speaker recording) without it
                # silently becoming part of your real corpus. Preview output
                # still gets written to disk (under PREVIEW_DIR, same idea as
                # PHRASE_TEST_DIR keeping option 4's output out of the
                # corpus-aggregating globs) so you can actually inspect it --
                # it's just never marked processed, never picked up by the
                # Analysis menu or rerun_rules.py, and doesn't trigger a
                # completion email, since nothing was "completed" into the
                # corpus. Re-running the same file(s) later and choosing
                # "save" processes them for real -- preview mode never touches
                # processed_local_mp3s.json, so nothing is consumed by previewing.
                save_choice = input("  Save results to your corpus, or just preview without saving? "
                                    "(save/preview) [save]: ").strip().lower() or "save"
                if save_choice not in ("save", "preview"):
                    print(f"  Unrecognized option '{save_choice}', defaulting to 'save'.")
                    save_choice = "save"
                save_results = (save_choice == "save")
                if not save_results:
                    print("  Preview mode: transcribing and analyzing normally, but nothing will be "
                          "written to your corpus or marked as processed -- rerun and choose "
                          "'save' anytime to keep the results for real.")
                # PATCH: cancel check happens before the (slow, RAM-heavy) model
                # load, not after -- no point loading Whisper just to bail out.
                reset_cysill_circuit_breaker()
                _ensure_model()
                notify_on_completion = save_results
                run_type = "Local MP3 batch" if save_results else "Local MP3 batch (preview)"
                run_start_time = time.time()
                videos_attempted = len(pending_mp3_files)
                keys = ["segments", "words", "lemmas", "pos", "mutations"]
                for p in tqdm(pending_mp3_files, desc="Videos", unit="video"):
                    meta = {"title": p.stem, "url": str(p), "source": "local",
                            "channel_register": local_reg}
                    try:
                        with tqdm(total=4, desc="Starting", leave=False, unit="step") as sub:
                            segs, words, lemmas, pos_r, muts, dur = analyze(str(p), model, meta, substeps=sub, preset=active_preset)
                        all_mutation_rows.extend(muts)
                        vpaths = _video_slug(meta, stamp) if save_results else _preview_video_slug(meta, stamp)
                        h = [True] * 5   # fresh header flags per video (new file each time)
                        any_written = False
                        for data, key, hi in zip([segs, words, lemmas, pos_r, muts], keys, range(5)):
                            if data:
                                _append(pd.DataFrame(data), vpaths[key], h, hi)
                                any_written = True
                        # PATCH: when every output list is empty (all segments
                        # got filtered out as hallucinations/non-speech/near-
                        # duplicates upstream in analyze()), the loop above
                        # writes nothing at all -- no CSVs, no folder contents --
                        # but the video still gets marked processed and reports
                        # "done in Xs" below, identically to a real success. That
                        # made an empty output folder look exactly like a normal
                        # run from the log, with no signal pointing at why.
                        # Surfacing it explicitly here so an empty folder is
                        # something you were told about, not something you have
                        # to notice on your own.
                        if not any_written:
                            tqdm.write(f"  ⚠️  {p.stem}: 0 segments survived filtering -- "
                                       f"nothing saved for this video (folder created but empty).")
                        if save_results:
                            stat = p.stat()
                            local_processed[str(p.resolve())] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                            save_local_processed(local_processed)
                        else:
                            tqdm.write(f"  👁  {p.stem}: preview only -- not saved to your corpus, "
                                       f"not marked processed. Output under "
                                       f"{PREVIEW_DIR.name}/{stamp}/ for inspection.")
                        tqdm.write(f"  {p.stem}: done in {dur:.1f}s")
                        tqdm.write("=" * 60)
                    except Exception as e:
                        tqdm.write(f"  💥 Error on {p.stem}: {e}")
                        tqdm.write("=" * 60)
                        failed_videos.append(p.stem)

                if not save_results:
                    # PATCH: preview batches skip the shared corpus-summary +
                    # email block entirely below (same "continue" pattern
                    # choices 2/5/6/7/8 already use) -- generate_research_summary()
                    # writes real research_summary_*.csv files under SUMMARY_DIR,
                    # and the completion email is for corpus-affecting runs;
                    # neither applies to results that were never saved.
                    succeeded = videos_attempted - len(failed_videos)
                    print(f"\nPreview complete: {succeeded}/{videos_attempted} file(s) transcribed "
                          f"and analyzed, {len(all_mutation_rows)} mutation row(s) found. Nothing "
                          f"written to your corpus -- rerun and choose 'save' to keep these results.")
                    save_lemma_cache()
                    continue

            elif choice == "2":
                selected_channels = prompt_channel_selection()
                raw_limit = input("Discovery limit [50], or q to cancel: ").strip().lower()
                if raw_limit == "q":
                    print("  Cancelled.")
                    save_lemma_cache()
                    continue
                try:
                    limit = int(raw_limit or "50")
                    if limit < 1:
                        raise ValueError
                except ValueError:
                    print("Please enter a whole number greater than zero.")
                    continue
                discover_new_videos(limit, channels=selected_channels)
                save_lemma_cache()
                continue   # skip summary + "Results saved" -- nothing was processed

            elif choice == "3":
                queue = load_queue()
                processed = load_processed()
                if not queue:
                    print("Queue is empty!")
                    save_lemma_cache()
                    continue
                raw_count = input(f"Process count (max {len(queue)}) [10], or q to cancel: ").strip().lower()
                if raw_count == "q":
                    print("  Cancelled.")
                    save_lemma_cache()
                    continue
                try:
                    how_many = int(raw_count or "10")
                    if how_many < 1:
                        raise ValueError
                except ValueError:
                    print("Please enter a whole number greater than zero.")
                    continue
                how_many = min(how_many, len(queue))
                # PATCH: cancel check happens before the (slow, RAM-heavy) model
                # load, not after -- no point loading Whisper just to bail out.
                reset_cysill_circuit_breaker()
                _ensure_model()
                notify_on_completion = True
                run_type = "Queue processing"
                run_start_time = time.time()
                videos_to_process = queue[:how_many]
                remaining_queue   = queue[how_many:]
                retry_queue       = []
                failed_state      = load_failed()
                videos_attempted = len(videos_to_process)
                # PATCH: was `mutation_engine.SPACY_NLP`, which no longer exists
                # -- SPACY_NLP moved to spacy_tagging.py when mutation_engine.py
                # got split up (AttributeError at runtime, since nothing ever
                # re-exported the raw variable, only the load_spacy()/
                # parse_spacy_doc() functions). Reading spacy_tagging.SPACY_NLP
                # directly here -- not via `from spacy_tagging import SPACY_NLP`
                # -- is required: that form would copy the value (None) at
                # import time, before load_spacy() ever runs, and never update.
                # Reading the module attribute after calling load_spacy() below
                # gets the live value.
                nlp = spacy_tagging.SPACY_NLP if load_spacy() else None
                keys = ["segments", "words", "lemmas", "pos", "mutations"]
                for video in tqdm(videos_to_process, desc="Videos", unit="video"):
                    try:
                        # PATCH: explicit start-of-video banner. Caption fetch
                        # now prints BEFORE transcription, so without this the
                        # previous video's tail output (hallucination warnings,
                        # etc.) ran straight into the next video's caption line
                        # with nothing marking the seam -- ambiguous in a long
                        # scrolling log. Mirrors the "done in Xs" line that
                        # already closes each video.
                        tqdm.write(f"\n▶ {video.get('title', video['id'])}")
                        # PATCH: vpaths computed here now, BEFORE caption fetch
                        # (previously computed after, once mp3/analyze were
                        # done) -- captions now nest per-run/per-video the same
                        # way transcriptions and mutations do (captions_dir),
                        # so download_captions() below needs vpaths to exist
                        # first to know where to write. _video_slug() is safe
                        # to call this early: it only derives a slug from
                        # already-available video metadata and creates
                        # directories, it doesn't depend on anything analyze()
                        # produces.
                        vpaths = _video_slug(video, stamp)
                        # PATCH: fetch captions FIRST, before transcription --
                        # this is "getting transcription data and validating
                        # it" as one step, not transcribe-now/validate-later.
                        # Captured regardless of what Whisper does with the
                        # audio, so corroboration data is sitting ready the
                        # moment this video's mutations exist a few lines down
                        # -- no separate manual pass needed. A caption-fetch
                        # failure (network hiccup, no captions at all, video
                        # taken down, etc.) only skips corroboration for THIS
                        # video, never the transcription itself -- losing the
                        # cross-check is not a reason to lose the data.
                        captions_csv_path = None
                        try:
                            manual_cy, auto_cy, _ = fetch_captions.list_available_tracks(video["url"])
                            if manual_cy or auto_cy:
                                vtt_path, cap_lang, cap_kind, _ = fetch_captions.download_captions(
                                    video["url"], out_dir=vpaths["captions_dir"])
                                if vtt_path is not None:
                                    cap_segments = fetch_captions.parse_vtt(vtt_path)
                                    captions_csv_path = vtt_path.with_suffix(".csv")
                                    with open(captions_csv_path, "w", newline="",
                                              encoding="utf-8-sig") as f:
                                        writer = csv.DictWriter(f, fieldnames=[
                                            "segment_start", "segment_end", "segment_text"])
                                        writer.writeheader()
                                        writer.writerows(cap_segments)
                                    tqdm.write(f"  Captions ({cap_kind}, {cap_lang}): "
                                               f"{len(cap_segments)} segment(s)")
                            else:
                                tqdm.write("  No Welsh captions available for this video.")
                        except Exception as e:
                            tqdm.write(f"  Caption fetch failed (continuing without "
                                       f"corroboration): {e}")

                        mp3_path = download_audio(video)
                        with tqdm(total=4, desc="Starting", leave=False, unit="step") as sub:
                            segs, words, lemmas, pos_r, muts, dur = analyze(mp3_path, model, video, substeps=sub, preset=active_preset)
                        all_mutation_rows.extend(muts)
                        h = [True] * 5   # fresh header flags per video (new file each time)
                        any_written = False
                        for data, key, hi in zip([segs, words, lemmas, pos_r, muts], keys, range(5)):
                            if data:
                                _append(pd.DataFrame(data), vpaths[key], h, hi)
                                any_written = True
                        # PATCH: see matching comment in the choice "1" (local MP3)
                        # block above -- when every output list is empty, nothing
                        # gets written and the run still reports "done" below with
                        # no indication the folder is empty.
                        if not any_written:
                            tqdm.write(f"  ⚠️  {video.get('title', video['id'])}: 0 segments "
                                       f"survived filtering -- nothing saved for this video "
                                       f"(folder created but empty).")

                        # PATCH: corroborate immediately, same run -- mutations
                        # + words for this video were just written above, so
                        # everything run_corroboration() needs is on disk now.
                        # cap_kind (manual/automatic) was already computed
                        # above when captions were fetched -- previously
                        # discarded after the console print at line ~490;
                        # now threaded through so it's persisted onto the
                        # mutations CSV as its own column.
                        if captions_csv_path is not None and muts:
                            try:
                                fetch_captions.run_corroboration(
                                    vpaths["mutations"], captions_csv_path, nlp=nlp, cap_kind=cap_kind)
                            except SystemExit:
                                tqdm.write("  Corroboration pass skipped (no matching words file).")
                            except Exception as e:
                                tqdm.write(f"  Corroboration pass failed: {e}")

                        processed.add(video["id"])
                        save_processed(processed)
                        clear_failure(video["id"], failed_state)
                        tqdm.write(f"  {video.get('title', video['id'])}: done in {dur:.1f}s")
                        tqdm.write("=" * 60)
                    except Exception as e:
                        tqdm.write(f"  💥 Error: {e}")
                        tqdm.write("=" * 60)
                        failed_videos.append(video.get("title", video["id"]))
                        if record_failure(video, e, failed_state):
                            retry_queue.append(video)
                            tqdm.write("  Will retry this video in a later queue run.")
                        else:
                            processed.add(video["id"])
                            save_processed(processed)
                            tqdm.write("  Retry limit reached; marked as processed. See failed_videos.json for details.")
                    time.sleep(2.0)   # rate limit buffer + session stabilization between videos
                save_queue(retry_queue + remaining_queue)

            elif choice == "4":
                # PATCH: removed the _ensure_model() call that used to be here --
                # analyze_phrase() takes no model argument and never touches
                # Whisper (it's pure text: regex-tokenize -> enrich_words ->
                # mutation detection). Loading a multi-GB model just to test a
                # typed phrase was pure waste, and actively bad given how tight
                # RAM already is on this machine.
                phrase = input("Enter a Welsh phrase, or q to cancel: ").strip()
                if phrase.lower() == "q":
                    print("  Cancelled.")
                    save_lemma_cache()
                    continue
                if phrase:
                    word_rows, lemma_rows, pos_rows, mutation_rows = analyze_phrase(phrase)
                    all_mutation_rows = mutation_rows
                    save_analysis_outputs(stamp, [], word_rows, lemma_rows, pos_rows, mutation_rows)

            elif choice == "5":
                manage_queue()
                save_lemma_cache()
                continue   # skip summary + "Results saved" -- nothing was processed

            elif choice == "6":
                # PATCH: corpus_analyzer.main() is interactive on its own (batch
                # selection) and calls sys.exit() on a couple of its own
                # early-exit paths (e.g. no mutation CSVs found yet, or nothing
                # left after deletion) -- catch SystemExit here so that just
                # returns to this menu instead of killing the whole pipeline.
                try:
                    corpus_analyzer.main()
                except SystemExit:
                    pass
                save_lemma_cache()
                continue   # skip video-processing summary + email -- nothing was processed here

            elif choice == "7":
                # PATCH: manual_editing.py is intentionally standalone (see its
                # module docstring -- duplicated constants, no dependency on
                # mutation_engine actually loading) so it's imported lazily here
                # rather than at module load time, keeping that independence
                # intact for people who still run it directly with
                # `python manual_editing.py`. Its main() parses sys.argv itself;
                # since this menu invokes it with no extra CLI args, it falls
                # through to its documented default (every mutations_*.csv under
                # mutations/, is_erosion==True rows, skipping already-reviewed
                # ones). For --pick / --sample / --trigger / etc., run
                # manual_editing.py directly from the command line instead.
                # Same SystemExit-catch reasoning as option 6: manual_editing.py
                # calls sys.exit(1) on its own early-exit path (no CSVs found),
                # which should return to this menu, not kill the pipeline.
                import manual_editing
                try:
                    manual_editing.main()
                except SystemExit:
                    pass
                save_lemma_cache()
                continue   # skip video-processing summary + email -- nothing was processed here

            elif choice == "8":
                # PATCH: rerun_rules.py re-evaluates already-cached tagging data
                # (re-parsing spaCy fresh per segment, reusing cached Cysill
                # fields, never re-transcribing or re-hitting Cysill) after a
                # change to a rule in mutation_engine.py or a table in
                # mutation_tables.py -- see rerun_rules.py's module docstring
                # for exactly what it does and doesn't touch. Unlike choice 7,
                # this genuinely has no sensible default (there's no "just rerun
                # everything" mode by design -- that's what choice 3 is for), so
                # the filter has to be gathered here rather than falling through
                # to a documented default the way manual_editing.py's does.
                import rerun_rules
                print("\nRe-run mutation rule(s) on already-transcribed videos.")
                print("Leave both blank to cancel -- at least one is required.")
                trig_input = input("Trigger word(s), comma-separated (e.g. yn,ei) [blank = any]: ").strip()
                rule_input = input("Rule name(s), comma-separated (e.g. word_trigger) [blank = any]: ").strip()
                if not trig_input and not rule_input:
                    print("Nothing specified -- cancelled.")
                    save_lemma_cache()
                    continue
                video_input = input("Video folder-name substring to limit to [default: all]: ").strip() or "all"
                mode_input = input("Write a comparison file first, or apply in place? "
                                   "(compare/apply) [compare]: ").strip().lower()
                commit = mode_input == "apply"
                if commit:
                    confirm = input("This will overwrite matching rows in the real mutations "
                                    "CSVs (manual-reviewed rows are always protected). "
                                    "Type 'yes' to continue: ").strip().lower()
                    if confirm != "yes":
                        print("Cancelled -- nothing was written.")
                        save_lemma_cache()
                        continue
                rerun_rules.run_rerun(trigger_arg=trig_input or None,
                                       rule_arg=rule_input or None,
                                       video=video_input, commit=commit)
                save_lemma_cache()
                continue   # skip video-processing summary + email -- nothing was transcribed here

            else:
                print("Unknown choice.")
                continue

            summary = generate_research_summary(all_mutation_rows, stamp)
            # PATCH: persist lemma cache at end of every run
            save_lemma_cache()
            print(f"\nResults saved in: {BASE_DIR}")

            if notify_on_completion:
                videos_succeeded = videos_attempted - len(failed_videos)
                elapsed = time.time() - run_start_time
                subject = (f"Welsh pipeline: {videos_succeeded}/{videos_attempted} videos, "
                           f"{len(all_mutation_rows)} mutation rows ({stamp})")
                if failed_videos:
                    subject += f" -- {len(failed_videos)} failed"
                body = build_email_body(
                    run_type=run_type,
                    stamp=stamp,
                    elapsed_seconds=elapsed,
                    videos_attempted=videos_attempted,
                    videos_succeeded=videos_succeeded,
                    failed_videos=failed_videos,
                    mutation_rows=all_mutation_rows,
                    summary=summary,
                    base_dir=BASE_DIR,
                )
                send_notification_email(subject=subject, body=body, html=True)


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(
        description="Welsh mutation-erosion pipeline.",
        epilog="Examples:\n"
               "  python welsh_pipeline.py            # accurate (default, unchanged behavior)\n"
               "  python welsh_pipeline.py fast       # beam_size 5, no temperature fallback\n"
               "  python welsh_pipeline.py balanced   # beam_size 5, limited fallback\n"
               "  python welsh_pipeline.py accurate   # beam_size 7, full fallback (old default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _ap.add_argument(
        "preset", nargs="?", default=None, choices=sorted(TRANSCRIBE_PRESETS),
        help="Transcription speed/quality preset -- see corpus_ops.TRANSCRIBE_PRESETS "
             "for exactly what each one sets and why."
    )
    _args = _ap.parse_args()
    main(preset=_args.preset)

# ================================================================
# Dedicated to a language that refused to disappear
# And to all the languages whose voices still struggle to be heard
#
# - A researcher in Korea, 2026
# ================================================================