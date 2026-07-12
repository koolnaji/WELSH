"""
corpus_ops.py
=============
Everything about moving data in and out of the linguistic engine: the
video queue / processed-log on disk, discovering new videos from the
CURATED_CHANNELS (with optional per-channel filtering), downloading audio,
running Whisper + the mutation_engine over it (analyze / analyze_phrase),
writing CSV outputs, and generating the research-summary figures.
"""
import json
import os
import re
import smtplib
import subprocess
import time
from email.mime.text import MIMEText
from pathlib import Path
import pandas as pd
import yt_dlp
from tqdm import tqdm

from mutation_engine import (
    BASE_DIR, AUDIO_DIR, TRANS_DIR, SUMMARY_DIR, VIDEO_QUEUE, PROCESSED_LOG,
    LOCAL_PROCESSED_LOG, FAILED_LOG, FAILED_MAX_RETRIES, CURATED_CHANNELS,
    EROSION_CONFIDENCE_THRESHOLD,
    ensure_dirs, run_stamp, run_paths,
    filter_hallucinated_segments, deduplicate_overlapping_segments,
    preprocess_segment, expand_whisper_tokens, enrich_words,
    normalize_word, get_welsh_lemma,
    cysill_coarse_pos, spacy_coarse_pos, pos_compatible,
    process_comprehensive_mutations,
)

# ========================= QUEUE / LOG HELPERS =========================
def _write_json(path, value):
    """Atomically replace a JSON state file to avoid corrupting it on a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)

def load_queue():
    if VIDEO_QUEUE.exists():
        try:
            data = json.loads(VIDEO_QUEUE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []
    return []

def save_queue(queue):
    _write_json(VIDEO_QUEUE, queue)

def load_processed():
    if PROCESSED_LOG.exists():
        try:
            data = json.loads(PROCESSED_LOG.read_text(encoding="utf-8"))
            return set(data) if isinstance(data, list) else set()
        except (OSError, json.JSONDecodeError):
            return set()
    return set()

def save_processed(processed):
    _write_json(PROCESSED_LOG, sorted(processed))

def load_failed():
    if not FAILED_LOG.exists():
        return {}
    try:
        data = json.loads(FAILED_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def save_failed(failed):
    _write_json(FAILED_LOG, failed)

def record_failure(video, error, failed):
    """Record a failed queue item and return whether it may be retried."""
    video_id = str(video["id"])
    previous = failed.get(video_id, {})
    attempts = int(previous.get("attempts", 0)) + 1
    failed[video_id] = {
        "attempts": attempts,
        "last_error": str(error),
        "title": video.get("title", video_id),
        "last_failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    save_failed(failed)
    return attempts < FAILED_MAX_RETRIES

def clear_failure(video_id, failed):
    if str(video_id) in failed:
        failed.pop(str(video_id))
        save_failed(failed)

def load_local_processed():
    if not LOCAL_PROCESSED_LOG.exists():
        return {}
    try:
        data = json.loads(LOCAL_PROCESSED_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def save_local_processed(processed):
    _write_json(LOCAL_PROCESSED_LOG, processed)


# ========================= EMAIL NOTIFICATION =========================
def send_notification_email(subject, body, html=False):
    """
    Sends a notification email via Gmail SMTP once video processing
    finishes (menu options 1 and 3 -- NOT the corpus_analyzer run, which
    is a separate, usually much quicker, offline step).

    Credentials are read from environment variables rather than hardcoded,
    so nothing sensitive lives in this file:
        GMAIL_SENDER        -- the Gmail address to send FROM. This account
                                needs 2-Step Verification turned on, and an
                                "App Password" generated for this script --
                                NOT your normal Gmail login password.
                                (Google Account -> Security -> 2-Step
                                Verification -> App passwords)
        GMAIL_APP_PASSWORD  -- the 16-character App Password from above.
        NOTIFY_RECIPIENT    -- address to send TO. Defaults to GMAIL_SENDER
                                (i.e. emails yourself) if not set.

    If GMAIL_SENDER / GMAIL_APP_PASSWORD aren't set, this prints a warning
    and returns quietly rather than crashing the pipeline over a
    notification -- a failed email should never lose processed data.

    PATCH: added `html` flag -- build_email_body() below produces an HTML
    body with organized sections instead of three flat text lines; this
    just needs to be sent with the right MIME subtype for Gmail to render
    it instead of showing raw tags. Plain-text callers are unaffected.
    """
    sender       = os.environ.get("GMAIL_SENDER", "").strip()
    app_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    recipient    = os.environ.get("NOTIFY_RECIPIENT", "").strip() or sender

    if not sender or not app_password:
        print("  ⚠️  Email notification skipped -- set GMAIL_SENDER and "
              "GMAIL_APP_PASSWORD environment variables to enable it.")
        return

    msg = MIMEText(body, "html" if html else "plain")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.starttls()
            server.login(sender, app_password)
            server.sendmail(sender, [recipient], msg.as_string())
        print(f"  📧 Notification email sent to {recipient}")
    except Exception as e:
        print(f"  ⚠️  Failed to send notification email: {e}")


def _fmt_hms(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def build_email_body(run_type, stamp, elapsed_seconds, videos_attempted,
                      videos_succeeded, failed_videos, mutation_rows,
                      summary, base_dir):
    """
    Builds an organized HTML run-report email, replacing the old 3-line
    plain-text body. Sections mirror generate_research_summary()'s console
    output (same numbers, same source of truth) plus run-level metadata
    that summary alone doesn't have: elapsed time, per-video success/
    failure, and a channel_register breakdown.

    `summary` is the dict returned by generate_research_summary() (or None
    if there was no mutation data this run -- e.g. every video failed).
    """
    def row(label, value):
        return (f'<tr><td style="padding:2px 14px 2px 0;color:#555;">{label}</td>'
                f'<td style="padding:2px 0;font-weight:600;">{value}</td></tr>')

    def section(title, rows_html):
        return (f'<h3 style="margin:18px 0 6px;font-size:14px;color:#1a1a1a;'
                f'border-bottom:1px solid #ddd;padding-bottom:4px;">{title}</h3>'
                f'<table style="border-collapse:collapse;font-size:13px;">{rows_html}</table>')

    parts = [
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        'max-width:560px;color:#222;">',
        '<h2 style="margin:0 0 4px;font-size:17px;">Welsh pipeline run finished</h2>',
        f'<div style="color:#777;font-size:12px;margin-bottom:10px;">'
        f'{run_type} &middot; run stamp {stamp}</div>',
    ]

    parts.append(section("Run", "".join([
        row("Type", run_type),
        row("Elapsed", _fmt_hms(elapsed_seconds)),
        row("Results saved in", str(base_dir)),
    ])))

    fail_html = ""
    if failed_videos:
        titles = "".join(f"<li>{t}</li>" for t in failed_videos)
        fail_html = (f'<div style="margin-top:6px;color:#b3261e;">'
                     f'<strong>Failed ({len(failed_videos)}):</strong>'
                     f'<ul style="margin:4px 0 0 18px;padding:0;">{titles}</ul></div>')
    parts.append(section("Videos", "".join([
        row("Attempted", videos_attempted),
        row("Succeeded", videos_succeeded),
        row("Failed", len(failed_videos)),
    ])) + fail_html)

    if summary:
        parts.append(section("Mutation data collected this run", "".join([
            row("Total contexts", summary["total_analyzed_contexts"]),
            row("Code-switch cases", f'{summary["code_switch_cases"]} '
                f'({summary["code_switch_rate"]:.1%})'),
            row("Evaluable non-code-switch", summary["evaluable_non_code_switch_contexts"]),
            row("Erosion (all)", f'{summary["erosion_cases_all"]} '
                f'({summary["erosion_rate_all"]:.1%})'),
            row("Erosion (high-confidence)", f'{summary["erosion_cases_high_confidence"]} '
                f'({summary["erosion_rate_high_confidence"]:.1%})'),
            row("Phantom omissions", summary["phantom_omission_cases"]),
            row("Erosion unverified", summary["erosion_unverified_cases"]),
            row("Homograph collision flags", summary["collision_flagged_cases"]),
        ])))

        if mutation_rows:
            df = pd.DataFrame(mutation_rows)
            if "channel_register" in df.columns:
                reg_counts = df["channel_register"].value_counts().to_dict()
                parts.append(section("By channel register", "".join(
                    row(reg.capitalize(), n) for reg, n in
                    sorted(reg_counts.items(), key=lambda kv: -kv[1]))))
    else:
        parts.append('<div style="margin-top:10px;color:#777;font-size:13px;">'
                      'No mutation data collected this run.</div>')

    parts.append('</div>')
    return "".join(parts)


# ========================= SUMMARY =========================
def generate_research_summary(mutation_rows, stamp):
    if not mutation_rows:
        print("No mutation data to summarize.")
        return
    df = pd.DataFrame(mutation_rows)

    total     = len(df)
    cs_cases  = int(df["is_code_switch"].sum()) if "is_code_switch" in df.columns else 0
    unverified = int((df["status"] == "erosion_unverified").sum()) if "status" in df.columns else 0
    non_cs    = total - cs_cases
    evaluable_non_cs = non_cs - unverified
    erosion   = int(df["is_erosion"].sum()) if "is_erosion" in df.columns else 0
    hc_erosion= int(df["is_high_confidence_erosion"].sum()) \
        if "is_high_confidence_erosion" in df.columns else 0

    summary = {
        "total_analyzed_contexts":       total,
        "code_switch_cases":             cs_cases,
        "code_switch_rate":              float(cs_cases / total) if total > 0 else 0,
        "erosion_cases_all":             erosion,
        "erosion_cases_high_confidence": hc_erosion,
        "evaluable_non_code_switch_contexts": evaluable_non_cs,
        "erosion_rate_all":              float(erosion / evaluable_non_cs) if evaluable_non_cs > 0 else 0,
        "erosion_rate_high_confidence":  float(hc_erosion / evaluable_non_cs) if evaluable_non_cs > 0 else 0,
        "correct_mutation_cases":        int((df["status"] == "correct_mutation").sum()),
        "wrong_mutation_type_cases":     int((df["status"] == "wrong_mutation_type").sum()),
        "phantom_omission_cases":        int((df["status"] == "phantom_mutation").sum()),
        "selective_invariancy_cases":    int((df["status"] == "selective_invariancy").sum()),
        "collision_flagged_cases":       int(df["collision_flag"].notna().sum()) if "collision_flag" in df.columns else 0,
        # New suppression / quality-gate counters
        "erosion_unverified_cases":      unverified,
        "bod_suppressed_note":           "bod forms suppressed as targets pre-analysis (not in output rows)",
        "implausible_orth_note":         "implausible-orthography tokens suppressed pre-analysis (not in output rows)",
    }
    (SUMMARY_DIR / "research_summary").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(
        SUMMARY_DIR / "research_summary" / f"research_summary_{stamp}.csv", index=False)

    breakdown_rows = []
    trig_df = df[
        (df["is_code_switch"] == False) &
        (df["trigger_word"] != "[OMITTED]") &
        (df["status"] != "erosion_unverified")
    ] \
        if "is_code_switch" in df.columns else df
    if not trig_df.empty and "expected_mutation" in trig_df.columns:
        for mut_type, group in trig_df.groupby("expected_mutation"):
            n = len(group)
            # PATCH: flag mutation types with small sample sizes (< 10) as unreliable
            low_n_warning = " ⚠️ LOW N" if n < 10 else ""
            breakdown_rows.append({
                "expected_mutation":   mut_type,
                "contexts":            n,
                "erosion_all":         int(group["is_erosion"].sum()),
                "erosion_hc":          int(group["is_high_confidence_erosion"].sum()),
                "erosion_rate_all":    float(group["is_erosion"].mean()),
                "erosion_rate_hc":     float(group["is_high_confidence_erosion"].mean()),
                "low_n_warning":       low_n_warning.strip(),
            })
    if breakdown_rows:
        (SUMMARY_DIR / "erosion_by_trigger_type").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(breakdown_rows).to_csv(
            SUMMARY_DIR / "erosion_by_trigger_type" / f"erosion_by_trigger_type_{stamp}.csv",
            index=False)

    rule_breakdown = []
    if "rule" in df.columns:
        rule_df = df[
            (df["is_code_switch"] == False) &
            (df["status"] != "erosion_unverified")
        ] if "is_code_switch" in df.columns else df
        for rule_name, group in rule_df.groupby("rule"):
            rule_breakdown.append({
                "rule":             rule_name,
                "contexts":         len(group),
                "erosion_all":      int(group["is_erosion"].sum()),
                "erosion_hc":       int(group["is_high_confidence_erosion"].sum()),
                "erosion_rate_all": float(group["is_erosion"].mean()),
                "erosion_rate_hc":  float(group["is_high_confidence_erosion"].mean()),
            })
    if rule_breakdown:
        (SUMMARY_DIR / "erosion_by_rule").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rule_breakdown).to_csv(
            SUMMARY_DIR / "erosion_by_rule" / f"erosion_by_rule_{stamp}.csv", index=False)

    detection_dist = df["detection_source"].value_counts().to_dict() \
        if "detection_source" in df.columns else {}
    agreement_dist = df["tagger_agreement"].value_counts().to_dict() \
        if "tagger_agreement" in df.columns else {}

    print("\n" + "="*70)
    print("RESEARCH SUMMARY - WELSH MUTATION ENGINE v7")
    print("="*70)
    print(f"Total analyzed contexts              : {total}")
    print(f"Code-switch cases                     : {cs_cases} ({cs_cases/total:.1%})"
          if total > 0 else f"Code-switch cases                     : {cs_cases}")
    print(f"Evaluable non-code-switch contexts    : {evaluable_non_cs}")
    er_rate = erosion / evaluable_non_cs if evaluable_non_cs > 0 else 0
    hc_rate = hc_erosion / evaluable_non_cs if evaluable_non_cs > 0 else 0
    print(f"Erosion (all)                         : {erosion} ({er_rate:.1%})")
    print(f"Erosion (high confidence ≥{EROSION_CONFIDENCE_THRESHOLD})       : {hc_erosion} ({hc_rate:.1%})")
    print(f"Erosion unverified (both taggers off)  : {summary['erosion_unverified_cases']}")
    print(f"Phantom omissions                      : {summary['phantom_omission_cases']}")
    print(f"Homograph collision flags              : {summary['collision_flagged_cases']}")
    if breakdown_rows:
        print("-"*70)
        print("Erosion by mutation type (all / high-confidence):")
        for row in breakdown_rows:
            print(f"  {row['expected_mutation']:<20} n={row['contexts']:<5} "
                  f"all={row['erosion_rate_all']:.1%}  hc={row['erosion_rate_hc']:.1%}"
                  f"  {row['low_n_warning']}")
    if rule_breakdown:
        print("-"*70)
        print("Contexts by rule:")
        for row in rule_breakdown:
            print(f"  {row['rule']:<32} n={row['contexts']:<5} "
                  f"all={row['erosion_rate_all']:.1%}  hc={row['erosion_rate_hc']:.1%}")
    if detection_dist:
        print("-"*70)
        print("Detection source:")
        for k, v in sorted(detection_dist.items(), key=lambda x: -x[1]):
            print(f"  {k:<35}: {v}")
    if agreement_dist:
        print("-"*70)
        print("Three-layer agreement:")
        for k, v in sorted(agreement_dist.items(), key=lambda x: -x[1]):
            print(f"  {k:<25}: {v}")
    print("="*70)

    # PATCH: return the summary dict so callers (e.g. the run-completion
    # email in welsh_pipeline.py) can reuse these exact numbers instead of
    # recomputing them -- single source of truth for console + email.
    return summary


# ========================= FILE UTILS =========================
def clean_title_for_file(title: str) -> str:
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    title = re.sub(r'[^\w\s\-.,!()]+', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:70]

def _channel_short_name(url):
    # e.g. "https://www.youtube.com/c/HanshS4C/videos" -> "HanshS4C"
    parts = url.rstrip("/").split("/")
    name  = parts[-2] if parts[-1] == "videos" else parts[-1]
    return name.lstrip("@")

def prompt_channel_selection():
    """Returns a list of CURATED_CHANNELS dicts ({"url", "channel_register"}),
    or None to mean "use all" (discover_new_videos default)."""
    print("\nAvailable channels:")
    for i, ch in enumerate(CURATED_CHANNELS, 1):
        print(f"  {i} = {_channel_short_name(ch['url']):<20} [{ch['channel_register']}]")
    print(f"  a = All channels")
    raw = input("Select channel(s) [a]: ").strip().lower() or "a"
    if raw == "a":
        return None   # None == use all (discover_new_videos default)
    picks = []
    for tok in raw.replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(CURATED_CHANNELS):
            picks.append(CURATED_CHANNELS[int(tok) - 1])
    if not picks:
        print("No valid channel selected, defaulting to all.")
        return None
    return picks

def discover_new_videos(limit, channels=None):
    out = []
    opts = {"quiet": True, "extract_flat": True, "no_warnings": True}
    processed = load_processed()
    queue     = load_queue()
    queue_ids = {v["id"] for v in queue}
    target_channels = channels if channels else CURATED_CHANNELS
    for ch in target_channels:
        channel_url = ch["url"]
        channel_register = ch["channel_register"]
        print(f"Collecting from: {channel_url} [{channel_register}]")
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                for entry in info.get("entries", []):
                    vid_id = entry.get("id")
                    if vid_id and vid_id not in processed and vid_id not in queue_ids:
                        out.append({"id": vid_id,
                                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                                    "title": entry.get("title", "unknown"),
                                    "source": channel_url,
                                    "channel_register": channel_register})
        except Exception as e:
            print(f" 💥 Failed collecting {channel_url}: {e}")
    seen   = set()
    unique = [x for x in out if not (x["id"] in seen or seen.add(x["id"]))]
    if unique:
        queue.extend(unique[:limit])
        save_queue(queue)
        print(f"Added {len(unique[:limit])} videos. Queue now has {len(queue)}.")
    else:
        print("No new videos found.")

# PATCH: yt-dlp has two independent sources of raw console output that
# don't know anything about our tqdm bars and write straight to
# stdout/stderr with their own carriage returns: (1) its download progress
# meter, and (2) its info/warning/error logger. Left alone, either one can
# fire mid-redraw of a "Videos: NN%|..." or per-video sub-progress bar and
# jam its text onto the same terminal line (e.g. "...1770.82s/video]ERROR:
# unable to download..."). "noprogress" kills #1 outright since we already
# show our own "Downloading (attempt N/3)..." line; routing #2 through this
# logger sends yt-dlp's messages through tqdm.write() so they print cleanly
# above whichever bar is currently active instead of colliding with it.
class _YtdlpTqdmLogger:
    def debug(self, msg):
        pass  # yt-dlp's internal debug/info channel is very chatty; drop it
    def info(self, msg):
        pass
    def warning(self, msg):
        tqdm.write(f"  ⚠️ yt-dlp: {msg}")
    def error(self, msg):
        tqdm.write(f"  ⚠️ yt-dlp: {msg}")


def download_audio(video, max_retries=3):
    safe_title = clean_title_for_file(video["title"])
    raw_path   = AUDIO_DIR / f"{safe_title}.mp3"
    norm_path  = AUDIO_DIR / f"{safe_title}_norm.mp3"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{safe_title}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True, "no_warnings": True, "noprogress": True,
        "logger": _YtdlpTqdmLogger(), "retries": 3,
    }
    for attempt in range(1, max_retries + 1):
        try:
            tqdm.write(f" Downloading (attempt {attempt}/{max_retries})...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video["url"]])
            break
        except Exception as e:
            tqdm.write(f" ⚠️ Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
            else:
                raise
    tqdm.write(" Normalizing audio...")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(raw_path), "-af",
             "loudnorm=I=-16:TP=-1.5:LRA=11", "-y", str(norm_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # PATCH: clean up raw file after successful normalization to avoid
        # accumulating double storage over long queue runs
        if norm_path.exists():
            raw_path.unlink(missing_ok=True)
        return str(norm_path)
    except Exception:
        tqdm.write(" ⚠️ Normalization failed, using raw file")
        return str(raw_path)


# ========================= ANALYSIS =========================
def analyze(audio_path, model, video_meta, substeps=None):
    """
    substeps: optional tqdm instance with total=4 (transcribe, preprocess,
    tag+align, mutations). If given, step descriptions go there instead of
    being print()ed -- used by pipeline.py for the per-video sub-progress bar.
    """
    def _step(label):
        if substeps is not None:
            substeps.set_description(label)
            substeps.update(1)
        else:
            print(f" {label}...")

    start_time = time.time()
    _step("Transcribing")
    transcribe_kwargs = {
        "language": "cy", "word_timestamps": True,
        "beam_size": 7, "best_of": 5, "patience": 1.0,
        "vad_filter": True,
        "vad_parameters": dict(threshold=0.6, min_silence_duration_ms=800,
                               max_speech_duration_s=20),
        "temperature": [0.0, 0.2, 0.4],
        "no_repeat_ngram_size": 5,
        "compression_ratio_threshold": 2.0,
        "no_speech_threshold": 0.6,
        "initial_prompt": "Cymraeg. Welsh language speech from S4C, Hansh, "
                          "Rownd a Rownd or Welsh YouTube.",
    }
    segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    # PATCH: info.duration is the actual audio length (seconds) faster-whisper
    # detected in the file -- distinct from `dur` at the end of this function,
    # which is wall-clock processing time. Captured here so it can be stamped
    # onto every output row below, letting corpus_analyzer.py answer "how many
    # minutes of video did this corpus actually cover?" without re-opening
    # any audio files.
    video_duration_seconds = round(getattr(info, "duration", 0.0) or 0.0, 2)
    segments = list(segments)
    segments = filter_hallucinated_segments(segments)
    segments = deduplicate_overlapping_segments(segments)

    # Pre-process: expand contractions, strip edge punctuation
    # Build one flat list of preprocessed words across all segments,
    # preserving segment boundary info for CSV output
    _step("Pre-processing transcription")
    all_preprocessed = []
    seg_boundaries   = []   # (start_idx, end_idx, seg_obj) for reconstruction
    for seg_id, seg in enumerate(segments):
        start_idx = len(all_preprocessed)
        words     = preprocess_segment(seg, seg_id=seg_id)
        all_preprocessed.extend(words)
        seg_boundaries.append((start_idx, len(all_preprocessed), seg))

    # Enrich: Cysill + spaCy via unified gap-tolerant alignment
    _step("Tagging + aligning (Cysill + spaCy)")
    enriched = enrich_words(all_preprocessed)

    # Build output rows
    segment_rows, word_rows, lemma_rows, pos_rows, words_only = [], [], [], [], []

    for seg_start, seg_end, seg in seg_boundaries:
        seg_text = seg.text.strip()
        segment_rows.append({
            "video_title":          video_meta["title"],
            "video_url":            video_meta.get("url", "local_file"),
            "source":               video_meta["source"],
            "channel_register":     video_meta.get("channel_register", "unverified"),
            "video_duration_seconds": video_duration_seconds,
            "segment_start":        round(seg.start, 3),
            "segment_end":          round(seg.end, 3),
            "segment_text":         seg_text,
            "language":             info.language,
            "language_probability": round(info.language_probability, 4),
        })

        for w in enriched[seg_start:seg_end]:
            raw_word  = w["word"]
            norm_word = normalize_word(raw_word)
            if len(norm_word) < 2 or w.get("synthetic"):
                continue

            lemma      = get_welsh_lemma(raw_word)
            conf       = w.get("confidence", 0.0)
            spacy_tok  = w.get("spacy_token")
            cysill_pos = w.get("cysill_pos")
            cysill_mut = w.get("cysill_mutation_type")
            gender     = w.get("gender")
            cpos_coarse = cysill_coarse_pos(cysill_pos)
            spos_coarse = spacy_coarse_pos(spacy_tok)
            pos_ok      = pos_compatible(cysill_pos, spacy_tok)

            word_rows.append({
                "video_title":   video_meta["title"],
                "video_url":     video_meta.get("url", "local_file"),
                "source":        video_meta["source"],
                "channel_register": video_meta.get("channel_register", "unverified"),
                "segment_start": round(seg.start, 3),
                "segment_end":   round(seg.end, 3),
                "word":          raw_word,
                "word_start":    w.get("start", seg.start),
                "word_end":      w.get("end", seg.end),
                "confidence":    conf,
                "language":      info.language,
                "cysill_pos":    cysill_pos,
                "cysill_coarse_pos": cpos_coarse,
                "gender":        gender,
                "spacy_dep":     spacy_tok["dep"] if spacy_tok else None,
                "spacy_pos":     spacy_tok["pos"] if spacy_tok else None,
                "spacy_coarse_pos": spos_coarse,
                "pos_compatible": pos_ok,
                "spacy_mutation":spacy_tok["mutation"] if spacy_tok else None,
            })
            lemma_rows.append({
                "video_title":     video_meta["title"],
                "video_url":       video_meta.get("url", "local_file"),
                "source":          video_meta["source"],
                "channel_register":   video_meta.get("channel_register", "unverified"),
                "word":            raw_word,
                "normalized_word": norm_word,
                "lemma":           lemma,
                "lemma_match":     lemma == norm_word if lemma else None,
                "word_start":      w.get("start"),
                "word_end":        w.get("end"),
                "confidence":      conf,
                "language":        info.language,
                "cysill_pos":      cysill_pos,
            })
            pos_rows.append({
                "video_title":          video_meta["title"],
                "video_url":            video_meta.get("url", "local_file"),
                "source":               video_meta["source"],
                "channel_register":     video_meta.get("channel_register", "unverified"),
                "word":                 raw_word,
                "cysill_pos":           cysill_pos,
                "cysill_mutation_type": cysill_mut,
                "cysill_gender":        w.get("cysill_gender"),
                "cysill_coarse_pos":    cpos_coarse,
                "spacy_dep":            spacy_tok["dep"] if spacy_tok else None,
                "spacy_pos":            spacy_tok["pos"] if spacy_tok else None,
                "spacy_mutation":       spacy_tok["mutation"] if spacy_tok else None,
                "spacy_gender":         spacy_tok["gender"] if spacy_tok else None,
                "spacy_coarse_pos":     spos_coarse,
                "pos_compatible":       pos_ok,
                "gender_unified":       gender,
                "segment_text":         seg_text,
                "word_start":           w.get("start"),
                "word_end":             w.get("end"),
                "confidence":           conf,
                "language":             info.language,
            })
            words_only.append(w)

    _step("Detecting mutations")
    mutation_rows = process_comprehensive_mutations(words_only)
    for row in mutation_rows:
        row.update({
            "video_title": video_meta["title"],
            "video_url":   video_meta.get("url", "local_file"),
            "source":      video_meta["source"],
            "channel_register": video_meta.get("channel_register", "unverified"),
            "video_duration_seconds": video_duration_seconds,
        })

    return segment_rows, word_rows, lemma_rows, pos_rows, mutation_rows, \
        time.time() - start_time


def analyze_phrase(phrase):
    fake_words = [{"word": tok, "start": float(i), "end": float(i) + 0.5,
                   "confidence": 1.0, "synthetic": False}
                  for i, tok in enumerate(
                      re.findall(r"\b[\w''\-]+\b", phrase, flags=re.UNICODE))]

    expanded  = expand_whisper_tokens(fake_words)
    enriched  = enrich_words(expanded)

    word_rows, lemma_rows, pos_rows = [], [], []
    for idx, w in enumerate(enriched):
        if w.get("synthetic"):
            continue
        lemma    = get_welsh_lemma(w["word"])
        norm_tok = normalize_word(w["word"])
        spacy_tok = w.get("spacy_token")
        cysill_pos = w.get("cysill_pos")

        word_rows.append({"token_index": idx, "word": w["word"]})
        lemma_rows.append({
            "token_index": idx, "word": w["word"],
            "normalized_word": norm_tok, "lemma": lemma,
            "lemma_match": lemma == norm_tok if lemma else None,
        })
        pos_rows.append({
            "token_index":          idx,
            "word":                 w["word"],
            "cysill_pos":           cysill_pos,
            "cysill_mutation_type": w.get("cysill_mutation_type"),
            "cysill_coarse_pos":    cysill_coarse_pos(cysill_pos),
            "gender":               w.get("gender"),
            "spacy_dep":            spacy_tok["dep"] if spacy_tok else None,
            "spacy_pos":            spacy_tok["pos"] if spacy_tok else None,
            "spacy_coarse_pos":     spacy_coarse_pos(spacy_tok),
            "pos_compatible":       pos_compatible(cysill_pos, spacy_tok),
            "spacy_mutation":       spacy_tok["mutation"] if spacy_tok else None,
        })

    mutation_rows = process_comprehensive_mutations(enriched)
    return word_rows, lemma_rows, pos_rows, mutation_rows


def save_analysis_outputs(stamp, segments, words, lemmas, pos_rows, mutations):
    paths = run_paths(stamp)
    if segments: pd.DataFrame(segments).to_csv(paths["segments"], index=False, encoding="utf-8-sig", quoting=1)
    if words:    pd.DataFrame(words).to_csv(paths["words"],    index=False, encoding="utf-8-sig", quoting=1)
    if lemmas:   pd.DataFrame(lemmas).to_csv(paths["lemmas"],  index=False, encoding="utf-8-sig", quoting=1)
    if pos_rows: pd.DataFrame(pos_rows).to_csv(paths["pos"],   index=False, encoding="utf-8-sig", quoting=1)
    if mutations:pd.DataFrame(mutations).to_csv(paths["mutations"], index=False, encoding="utf-8-sig", quoting=1)
