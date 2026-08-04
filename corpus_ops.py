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
from urllib.parse import unquote, urlparse
import pandas as pd
import requests
import yt_dlp
from tqdm import tqdm

from mutation_engine import (
    AUDIO_DIR, SUMMARY_DIR, VIDEO_QUEUE, PROCESSED_LOG,
    LOCAL_PROCESSED_LOG, FAILED_LOG, FAILED_MAX_RETRIES, CURATED_CHANNELS,
    EROSION_CONFIDENCE_THRESHOLD,
    run_paths,
    filter_hallucinated_segments, deduplicate_overlapping_segments,
    preprocess_segment, expand_whisper_tokens, enrich_words,
    normalize_word, get_welsh_lemma, is_english_code_switch,
    cysill_coarse_pos, spacy_coarse_pos, pos_compatible,
    process_comprehensive_mutations,
)
# PATCH: single source of truth (see mutation_tables.py for rationale) --
# was three separate inline copies of this same 4-status list in this
# file, plus a fourth, DIFFERENT (and wrong) computation for the headline
# email number. All four now reference this one import.
from mutation_tables import EVALUABLE_STATUSES

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
            row("Total words (this run's videos)", summary["total_words"]),
            row("Code-switch words (whole transcript)", f'{summary["code_switch_words_total"]} '
                f'({summary["code_switch_rate"]:.1%} of {summary["code_switch_rate_denominator"]})'),
            row("Code-switch cases at mutation triggers", summary["code_switch_cases_at_triggers"]),
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

            # PATCH: tagger_agreement + detection rule breakdowns -- without
            # these, "erosion (all)" is a single trust-everything number.
            # tagger_agreement tells you how much of it is backed by actual
            # multi-tagger confirmation (full_agreement) vs a single
            # heuristic firing alone (heuristic_only) -- the same
            # distinction manual_editing.py and corpus_stats.py already
            # surface, just not previously in the email. `rule` breaks
            # down WHICH detection layer is producing the volume (plain
            # word_trigger vs spacy_obj_soft vs phantom_check etc.), so a
            # spike from one rule (e.g. the mae/gan false-trigger pattern)
            # is visible at a glance instead of buried in the aggregate.
            if "tagger_agreement" in df.columns:
                agreement_counts = df["tagger_agreement"].value_counts().to_dict()
                parts.append(section("By tagger agreement", "".join(
                    row(agreement.replace("_", " ").capitalize(), n) for agreement, n in
                    sorted(agreement_counts.items(), key=lambda kv: -kv[1]))))

            if "rule" in df.columns:
                rule_counts = df["rule"].value_counts().to_dict()
                parts.append(section("By detection rule", "".join(
                    row(rule, n) for rule, n in
                    sorted(rule_counts.items(), key=lambda kv: -kv[1]))))

            # PATCH: everything below is new -- previously the email
            # reported erosion as one or two flat numbers, with no way to
            # see WHERE it's concentrated without opening the CSVs
            # yourself. These sections all derive from columns already
            # sitting in mutation_rows for this run; nothing here reads
            # anything beyond what was already collected.

            # Erosion RATE (not just row count) broken out by mutation
            # type -- the exact axis corpus_analyzer.py's own charts use,
            # and the one that answers "is erosion concentrated in one
            # mutation type or spread evenly" at a glance.
            if {"expected_mutation", "is_erosion"} <= set(df.columns):
                evaluable = df[df["status"].isin(
                    EVALUABLE_STATUSES
                )] if "status" in df.columns else df
                if len(evaluable):
                    by_type = evaluable.groupby("expected_mutation")["is_erosion"].agg(["sum", "count"])
                    parts.append(section("Erosion rate by mutation type", "".join(
                        row(mut_type, f'{int(r["sum"])}/{int(r["count"])} '
                            f'({r["sum"] / r["count"]:.1%})')
                        for mut_type, r in by_type.sort_values("count", ascending=False).iterrows()
                    )))

            # Erosion RATE by channel_register -- the actual formal-vs-
            # informal research comparison this whole project is built
            # around; previously only row COUNTS per register were shown,
            # which says nothing about whether erosion differs between them.
            if {"channel_register", "is_erosion"} <= set(df.columns):
                evaluable = df[df["status"].isin(
                    EVALUABLE_STATUSES
                )] if "status" in df.columns else df
                if len(evaluable):
                    by_reg = evaluable.groupby("channel_register")["is_erosion"].agg(["sum", "count"])
                    parts.append(section("Erosion rate by channel register", "".join(
                        row(reg.capitalize(), f'{int(r["sum"])}/{int(r["count"])} '
                            f'({r["sum"] / r["count"]:.1%})')
                        for reg, r in by_reg.sort_values("count", ascending=False).iterrows()
                    )))

            # Full status distribution as percentages -- the same
            # breakdown as corpus_analyzer.py's stacked-bar chart
            # (correct / selective-invariancy / erosion / wrong-type /
            # mismatch / phantom / code-switch), so a skim of the email
            # gives the same picture without opening a chart.
            if "status" in df.columns:
                status_counts = df["status"].value_counts()
                n_total = len(df)
                parts.append(section("Full status distribution", "".join(
                    row(s.replace("_", " ").capitalize(), f'{n} ({n / n_total:.1%})')
                    for s, n in status_counts.items()
                )))

            # Per-video breakdown -- title, length, how many mutation
            # contexts were found, and that video's own erosion rate.
            # Previously the only per-video visibility was the
            # succeeded/failed title lists above; this is the first place
            # you can see at a glance whether one video is behaving very
            # differently from the rest of the batch.
            if "video_title" in df.columns:
                video_rows = []
                for title, vdf in df.groupby("video_title", sort=False):
                    dur = vdf["video_duration_seconds"].iloc[0] \
                        if "video_duration_seconds" in vdf.columns and pd.notna(vdf["video_duration_seconds"].iloc[0]) \
                        else None
                    v_evaluable = vdf[vdf["status"].isin(
                        EVALUABLE_STATUSES
                    )] if "status" in vdf.columns else vdf
                    v_erosion_rate = (v_evaluable["is_erosion"].mean()
                                       if len(v_evaluable) and "is_erosion" in v_evaluable.columns else None)
                    dur_str = _fmt_hms(dur) if dur is not None else "?"
                    rate_str = f'{v_erosion_rate:.1%}' if v_erosion_rate is not None else "n/a"
                    short_title = (title[:44] + "…") if isinstance(title, str) and len(title) > 45 else title
                    video_rows.append(row(short_title, f'{dur_str} &middot; {len(vdf)} contexts &middot; erosion {rate_str}'))
                parts.append(section("By video", "".join(video_rows)))

            # Caption corroboration summary, if this run actually
            # corroborated anything -- previously invisible in the email
            # entirely despite being a whole subsystem that runs
            # automatically inside choice 3.
            if "caption_corroboration" in df.columns and df["caption_corroboration"].notna().any():
                corrob_counts = df["caption_corroboration"].value_counts().to_dict()
                parts.append(section("Caption corroboration", "".join(
                    row(outcome.replace("_", " ").capitalize(), n) for outcome, n in
                    sorted(corrob_counts.items(), key=lambda kv: -kv[1]))))

            # How many rows this run left flagged for manual follow-up
            # (caption corroboration disputes, low-confidence POS
            # struggle, etc.) -- a direct pointer to "here's how much
            # manual_editing.py work this run generated."
            if "flagged" in df.columns:
                flagged_n = int(df["flagged"].fillna(False).sum())
                if flagged_n:
                    parts.append(section("Needs manual review", "".join([
                        row("Rows flagged this run", flagged_n),
                    ])))
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

    # PATCH: was `evaluable_non_cs = non_cs - unverified`, which still
    # counted phantom_mutation and selective_invariancy rows as
    # "evaluable" -- neither one can register erosion by construction
    # (phantom rows have no trigger word to evaluate against; selective-
    # invariancy rows have no mutation expected at all). That meant the
    # headline "Erosion (all)" / "Erosion (high-confidence)" numbers at
    # the top of every completion email were computed on a looser,
    # larger denominator than the breakdowns further down this same
    # email (which already correctly filtered to EVALUABLE_STATUSES, see
    # the three call sites below) -- confirmed live on a real run where
    # this understated the true erosion rate by ~18 points. Now uses the
    # same EVALUABLE_STATUSES filter as everywhere else in the pipeline.
    if "status" in df.columns:
        evaluable_df      = df[df["status"].isin(EVALUABLE_STATUSES)]
        evaluable_non_cs  = len(evaluable_df)
        erosion           = int(evaluable_df["is_erosion"].sum()) if "is_erosion" in evaluable_df.columns else 0
        hc_erosion        = int(evaluable_df["is_high_confidence_erosion"].sum()) \
            if "is_high_confidence_erosion" in evaluable_df.columns else 0
    else:
        non_cs            = total - cs_cases
        evaluable_non_cs  = non_cs - unverified
        erosion           = int(df["is_erosion"].sum()) if "is_erosion" in df.columns else 0
        hc_erosion        = int(df["is_high_confidence_erosion"].sum()) \
            if "is_high_confidence_erosion" in df.columns else 0

    # PATCH: numerator changed to a true whole-transcript code-switch word
    # count (video_codeswitch_word_count -- every English word detected
    # anywhere in the transcript, via the same classifier get_welsh_lemma
    # already runs on every word), not cs_cases (mutation-trigger-context
    # rows only, a much narrower and less representative slice). Both
    # numerator and denominator are per-video totals repeated across every
    # row from that video, so both dedupe by video_url before summing --
    # otherwise a video with more trigger contexts would silently inflate
    # its own contribution to either total. cs_cases itself is kept as its
    # own field below (still meaningful: how many mutation TARGETS
    # specifically were code-switched), just no longer used as the
    # headline code-switch rate's numerator. Older cached rows (predating
    # these fields, or rebuilt via rerun_rules.py, which doesn't currently
    # carry them through) fall back to the old contexts-based rate so this
    # doesn't divide by zero on that data.
    if {"video_word_count", "video_codeswitch_word_count", "video_url"} <= set(df.columns):
        video_level      = df.dropna(subset=["video_url"]).drop_duplicates("video_url")
        total_words      = int(video_level["video_word_count"].fillna(0).sum())
        codeswitch_words = int(video_level["video_codeswitch_word_count"].fillna(0).sum())
    else:
        total_words, codeswitch_words = 0, None
    if total_words > 0 and codeswitch_words is not None:
        code_switch_rate = float(codeswitch_words / total_words)
        code_switch_denominator_label = "total_words"
    else:
        codeswitch_words = cs_cases
        code_switch_rate = float(cs_cases / total) if total > 0 else 0
        code_switch_denominator_label = "total_contexts (video_word_count unavailable)"

    summary = {
        "total_analyzed_contexts":       total,
        "code_switch_cases_at_triggers": cs_cases,
        "code_switch_words_total":       codeswitch_words,
        "total_words":                   total_words,
        "code_switch_rate":              code_switch_rate,
        "code_switch_rate_denominator":  code_switch_denominator_label,
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
    print(f"Code-switch words (whole transcript)  : {summary['code_switch_words_total']} "
          f"({summary['code_switch_rate']:.1%} of {summary['code_switch_rate_denominator']})")
    print(f"Code-switch cases at mutation triggers : {cs_cases}")
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
    # PATCH: non-YouTube feed URLs often end in a generic filename
    # (rss/feed/episodes/a cache .json, sometimes with a trailing
    # ?v=... cache-busting query string) that tells you nothing about
    # the show -- fall back to the domain in that case instead of
    # printing e.g. "rss" or "podcast-cwins.json?v=1" in the channel
    # picker. This is now only a fallback for entries with no explicit
    # "name" -- see channel_display_name() below, which every current
    # non-YouTube CURATED_CHANNELS entry actually has.
    bare = name.split("?")[0].lower()
    if bare in ("rss", "feed", "episodes") or bare.endswith(".json"):
        name = urlparse(url).netloc
    return name.lstrip("@")

def channel_display_name(ch_or_url):
    """
    Human-readable name for a CURATED_CHANNELS entry (or a bare source
    URL, e.g. one already stored on a queue entry). Prefers the explicit
    "name" field -- added specifically because non-YouTube feed/cache
    URLs are opaque to a human (an RSS path or cache filename says
    nothing about the show) -- falling back to _channel_short_name()'s
    URL-derived slug for entries that don't define one (the existing
    YouTube channels, whose slug already reads fine, e.g. "HanshS4C").
    Accepts either a CURATED_CHANNELS dict directly, or a bare URL
    string (looked up against CURATED_CHANNELS by exact URL match) --
    the latter covers queue entries, which only persist "source", not
    the full channel dict that produced them.
    """
    if isinstance(ch_or_url, dict):
        name = ch_or_url.get("name")
        return name if name else _channel_short_name(ch_or_url["url"])
    for ch in CURATED_CHANNELS:
        if ch["url"] == ch_or_url:
            return ch.get("name") or _channel_short_name(ch_or_url)
    return _channel_short_name(ch_or_url)

def prompt_channel_selection():
    """Returns a list of CURATED_CHANNELS dicts ({"url", "channel_register"}),
    or None to mean "use all" (discover_new_videos default)."""
    print("\nAvailable channels:")
    for i, ch in enumerate(CURATED_CHANNELS, 1):
        print(f"  {i} = {channel_display_name(ch):<30} [{ch['channel_register']}]")
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

def _resolve_entry_url(entry, channel_url):
    """
    extract_flat entries vary by extractor: YouTube's flat channel
    listing only gives a bare video id (no scheme) in "url", so that
    case needs reconstructing into a real watch URL. Podcast RSS
    entries, on the other hand, already carry a fully-qualified
    url/webpage_url pointing at the actual episode -- reconstructing
    anything for those would silently point every episode at a bogus
    youtube.com/watch link instead.
    """
    candidate = entry.get("webpage_url") or entry.get("url")
    if candidate and candidate.startswith("http"):
        return candidate
    vid_id = entry.get("id")
    if vid_id and "youtube.com" in channel_url:
        return f"https://www.youtube.com/watch?v={vid_id}"
    return candidate  # may be None -- caller skips if so


def _extract_direct_media_url(anchor_play_url):
    """
    Anchor/Spotify-for-Podcasters wraps each episode's real media URL
    in a tracking redirect:
    https://anchor.fm/s/<show_id>/podcast/play/<ep_id>/<urlencoded-real-url>
    yt-dlp can usually follow this redirect on its own, but pulling the
    real CDN URL out directly is more robust -- it doesn't depend on
    yt-dlp recognizing Anchor's specific redirect scheme, just a plain
    HTTP(S) media file, which the generic downloader always handles.
    """
    m = re.search(r'/play/\d+/(https?%3A.*)$', anchor_play_url, re.IGNORECASE)
    return unquote(m.group(1)) if m else anchor_play_url


def _discover_ypod_json(source_url, channel_register, processed, queue_ids):
    """
    Y Pod hosts some shows (e.g. Pryd Ar Dafod) via an internal cache
    JSON rather than exposing a plain RSS feed -- reverse-engineered
    from the site's own Network requests, not a documented public API,
    so this may need adjusting if Y Pod changes their cache format.
    Returns a list of queue-entry dicts in the same shape yt-dlp-backed
    discovery produces, so the rest of discover_new_videos() doesn't
    need to know which path a given source came through.
    """
    out = []
    try:
        resp = requests.get(source_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f" 💥 Failed collecting {source_url}: {e}")
        return out
    for ep in data.get("Episodes", []):
        audio_url = ep.get("audio")
        if not audio_url:
            continue
        direct_url = _extract_direct_media_url(audio_url)
        m = re.search(r'/play/(\d+)/', audio_url)
        ep_id = m.group(1) if m else direct_url
        if ep_id in processed or ep_id in queue_ids:
            continue
        out.append({"id": ep_id, "url": direct_url,
                    "title": ep.get("title", "unknown"),
                    "source": source_url, "channel_register": channel_register})
    return out


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
        # PATCH: Y Pod's internal cache JSON isn't RSS/Atom, so yt-dlp's
        # generic extractor can't enumerate it as a playlist -- branch
        # to a dedicated adapter instead of forcing it through yt-dlp.
        if ch.get("type") == "ypod_json":
            out.extend(_discover_ypod_json(channel_url, channel_register,
                                            processed, queue_ids))
            continue
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                for entry in info.get("entries", []):
                    # PATCH: was hardcoded to reconstruct a
                    # youtube.com/watch?v=<id> URL regardless of
                    # source -- silently wrong for any non-YouTube
                    # feed (e.g. podcast RSS), which already carries
                    # its own real episode URL. See
                    # _resolve_entry_url()'s docstring.
                    vid_id    = entry.get("id") or entry.get("url")
                    entry_url = _resolve_entry_url(entry, channel_url)
                    if vid_id and entry_url and vid_id not in processed and vid_id not in queue_ids:
                        out.append({"id": vid_id,
                                    "url": entry_url,
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


# PATCH: getaddrinfo failed / "Failed to resolve" (Windows errno 11001,
# WSAHOST_NOT_FOUND -- confirmed via nslookup on 2026-08-03 that this was
# a transient resolver blip, not a blocked/misconfigured domain: the same
# host resolved cleanly seconds later through the normal system resolver)
# is a DIFFERENT failure shape than the other things download_audio()
# retries for. A read-timeout or a 5xx means the server was reachable but
# slow/unhappy -- 3s/6s/9s is a reasonable amount of patience for that.
# A DNS resolution failure means the resolver never got an answer at all,
# and the previous incident showed the existing schedule burns all 3
# attempts back-to-back before the blip has any real chance to clear.
# DNS_BACKOFF_SECONDS is deliberately longer and used ONLY for this error
# shape; every other exception still uses the original 3*attempt schedule
# below, unchanged.
DNS_BACKOFF_SECONDS = (5, 15, 30)


def _is_dns_resolution_error(exc):
    """
    True if this exception is (or was caused by) a DNS resolution
    failure specifically -- checked by substring on str(exc) rather than
    exception type, since yt-dlp wraps the underlying
    socket.gaierror/urllib3 NewConnectionError in its own
    DownloadError/ExtractorError, so the original exception type doesn't
    survive to this level, but its message text does (confirmed against
    the actual error text in the 2026-08-03 log: "Failed to resolve
    'media24.fireside.fm' ([Errno 11001] getaddrinfo failed)").
    Covers both the Windows-specific errno text and the generic
    "getaddrinfo failed" phrasing Linux/Mac raise instead, so this isn't
    tied to one OS.
    """
    msg = str(exc).lower()
    return "getaddrinfo failed" in msg or "failed to resolve" in msg


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
                if _is_dns_resolution_error(e):
                    wait = DNS_BACKOFF_SECONDS[min(attempt - 1, len(DNS_BACKOFF_SECONDS) - 1)]
                    tqdm.write(f"    (DNS resolution failure -- waiting {wait}s "
                               f"for resolver to recover, longer than the usual "
                               f"backoff)")
                else:
                    wait = 3 * attempt
                time.sleep(wait)
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


# ========================= TRANSCRIPTION PRESETS =========================
# One source of truth for the beam_size/best_of/temperature knobs that
# actually drive Whisper decode time -- previously hardcoded inline in
# analyze() below, meaning "try a faster setting" meant editing this file
# directly. "accurate" is byte-for-byte what was hardcoded before this
# existed, so picking it (or omitting --preset) reproduces prior behavior
# exactly -- nothing changes unless you deliberately pick "fast" or
# "balanced".
#
# What each knob actually costs, so the tradeoff here is legible rather
# than three unlabeled dials:
#   - beam_size: search width during decoding. Cost scales roughly with
#     this number. 5 is the standard Whisper default; 7 (the old hardcoded
#     value) is a real step up in compute for typically modest accuracy
#     gains on already-decent audio.
#   - temperature: NOT "try three temperatures and blend them" -- it's
#     SEQUENTIAL FALLBACK. Whisper decodes at the first value; only if
#     that decode fails quality checks (compression_ratio_threshold,
#     no_speech_threshold) does it re-decode the ENTIRE segment from
#     scratch at the next value. Each fallback attempt pays the full
#     beam_size/best_of cost again. On messy, code-switching, or noisy
#     audio -- exactly this project's profile -- fallback can trigger
#     often enough to multiply decode time by up to len(temperature) on
#     the affected segments. This is the single most likely place a
#     multi-hour run is actually going. "fast" cuts fallback entirely
#     (temperature=[0.0]) -- safe to try on THIS project specifically
#     because filter_hallucinated_segments() already exists downstream to
#     catch and drop garbled output that fallback would otherwise have
#     tried to fix, so you're not removing your only safety net, just the
#     expensive one.
#   - best_of: only relevant when temperature > 0 (it's a sampling-pass
#     parameter, unused during the temperature=0.0 beam-search pass) --
#     irrelevant for "fast" (single temperature=0.0 pass), included in
#     "balanced"/"accurate" since those still fall back to temp>0 passes.
TRANSCRIBE_PRESETS = {
    "fast": {
        "beam_size": 5, "best_of": None, "patience": 1.0,
        "temperature": [0.0],
    },
    "balanced": {
        "beam_size": 5, "best_of": 5, "patience": 1.0,
        "temperature": [0.0, 0.2],
    },
    "accurate": {
        "beam_size": 7, "best_of": 5, "patience": 1.0,
        "temperature": [0.0, 0.2, 0.4],
    },
}
DEFAULT_TRANSCRIBE_PRESET = "accurate"  # = old hardcoded behavior, unchanged default


# ========================= SAMPLING =========================
# PATCH: added so a single long recording (2hr+ podcasts, in particular)
# doesn't have to be transcribed/tagged in full just to get a rate
# estimate -- this project's metrics are all proportions (mutation
# application rate, code-switch rate, erosion rate), not raw counts, so
# a fixed-length sample per video is a legitimate way to cut wall-clock
# time without changing what's being measured. Deliberately does NOT use
# faster-whisper's own `clip_timestamps` param for this: per the
# faster-whisper docs, passing clip_timestamps makes vad_filter get
# silently IGNORED -- and this pipeline's vad_filter + vad_parameters is
# load-bearing for the existing 4-layer hallucination defense
# (filter_hallucinated_segments() downstream assumes VAD already did its
# job). That's exactly the "two things that look interchangeable but
# aren't" shape of bug this project has been bitten by before (see
# mutation_engine.py's tagger_agreement/DOM-Vnoun-conflation comments),
# so instead this physically trims the audio FILE before Whisper ever
# sees it -- vad_filter runs exactly as it always has, just over a
# shorter file.
#
# PATCH: sample window now starts `skip_seconds` in rather than at 0:00.
# Sponsor reads, cold opens, "thanks for watching" scripted intros, etc.
# aren't representative of the casual/spontaneous speech this project is
# actually trying to measure -- sampling from 0:00 would systematically
# feed the pipeline the LEAST representative minutes of every video.
# Deliberate tradeoff, by request: if a video has an ad or a musical
# intro sitting inside the skip window, that's just lost, not detected
# and skipped around -- this is a fixed offset, not ad-detection.
def _probe_duration_seconds(path):
    """
    ffprobe-based duration lookup, used only to decide whether a video
    is even long enough to support skip_seconds + sample_seconds.
    Returns None on any failure (missing ffprobe, corrupt file, etc.) --
    callers treat None the same as "unknown, don't skip", since guessing
    wrong here means either silently sampling 0 seconds (skip stacked
    past the end) or crashing on a bad seek, neither of which is
    acceptable for something as unattended as a queue-processing run.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, check=True)
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception:
        return None


def sample_audio_window(audio_path, sample_seconds, skip_seconds=300, pad_seconds=20):
    """
    Returns (path, bounds).

    `path` is an audio file containing the TRUE sample window
    (`sample_seconds` starting `skip_seconds` in, default 5 min -- see
    module comment above) PLUS `pad_seconds` (default 20s) of extra
    audio on each side, clamped to the real file boundaries. The padding
    exists purely so Whisper sees complete sentences straddling the true
    window's edges instead of audio cut mid-word/mid-sentence -- mutation
    detection needs at least sentence-level context, so a segment that's
    truncated by an arbitrary cut point is unusable, not just noisy.

    `bounds` is None if sample_seconds is falsy (no sampling requested --
    caller should treat segment/word timestamps as already true-video-
    relative, unchanged behavior). Otherwise it's a dict describing the
    TRUE (unpadded) window in original-video time, which the caller
    (analyze(), via _shift_and_trim_padded_segments()) uses to (1) shift
    Whisper's clip-relative timestamps back to true-video time -- Whisper
    only ever sees the trimmed file and counts from 0 -- and (2) drop any
    segment that only exists because of the padding:
        extraction_start: where the audio was actually cut from (padded)
        true_start / true_end: the real sample window boundaries
        at_video_start / at_video_end: whether true_start/true_end sit at
            the genuine edges of the video (0:00, or the file's actual
            end) -- a segment straddling a GENUINE edge is not a padding
            artifact and should not be dropped for it. If the file's
            duration couldn't be probed, at_video_end is conservatively
            False (better to drop a possibly-genuine trailing segment
            than to keep one that might be a padding artifact).

    Falls back to starting at 0:00 instead ONLY if the video is too short
    to support the skip (i.e. skipping would leave less than
    `sample_seconds` of material, or duration couldn't be determined at
    all) -- a short video still gets sampled, just without the
    intro-skipping, rather than silently producing an empty/near-empty
    clip.

    If sample_seconds is falsy, returns (audio_path, None) unchanged --
    no pointless copy when no sampling was requested at all.

    Uses `-c copy` (stream copy, no re-encode) since the input is always
    an already-normalized mp3 by the time this is called -- this is a
    near-instant container-level cut, not a real transcode, so it adds
    negligible time to the run it's meant to be shortening.

    Output goes next to the source file with a `_sampleSTART-ENDs`
    suffix (true window, not the padded extraction) so repeated runs
    with the same window reuse/overwrite predictably rather than
    accumulating junk files.
    """
    if not sample_seconds:
        return audio_path, None
    src = Path(audio_path)
    duration = _probe_duration_seconds(src)
    true_start = skip_seconds
    if duration is None or duration < skip_seconds + sample_seconds:
        true_start = 0
    true_end = true_start + sample_seconds

    extraction_start = max(0, true_start - pad_seconds)
    extraction_end = true_end + pad_seconds
    if duration is not None:
        extraction_end = min(duration, extraction_end)
    extraction_len = extraction_end - extraction_start

    out = src.with_name(
        f"{src.stem}_sample{int(true_start)}-{int(true_end)}s{src.suffix}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(extraction_start), "-i", str(src),
             "-t", str(extraction_len), "-c", "copy", str(out)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if out.exists() and out.stat().st_size > 0:
            bounds = {
                "extraction_start": extraction_start,
                "true_start": true_start,
                "true_end": true_end,
                "at_video_start": true_start == 0,
                "at_video_end": duration is not None and true_end >= duration - 0.5,
            }
            return str(out), bounds
    except Exception as e:
        tqdm.write(f" ⚠️ Sampling trim failed ({e}) -- using full audio file instead.")
    return audio_path, None


def _shift_and_trim_padded_segments(segments, sample_bounds):
    """
    Converts Whisper's clip-relative segment/word timestamps into true-
    video time, and drops any segment that only exists because of the
    padding sample_audio_window() adds around the true sample window.

    Without this, two bugs ship silently: (1) every timestamp in the
    CSVs is relative to the trimmed/padded clip rather than the real
    video -- breaking caption corroboration in fetch_captions.py, which
    aligns against the real caption track's real timestamps, and making
    the timestamp columns meaningless against the actual video; and (2)
    a sentence that only transcribed cleanly because padding gave
    Whisper the rest of it, sitting outside the true sample window,
    would get counted as sampled content when only the padding-covered
    fragment of it was ever supposed to be in this sample.

    faster-whisper's Segment/Word are plain (non-frozen) dataclasses, so
    timestamps are shifted in place rather than rebuilt.

    No-op (segments returned unchanged) when sample_bounds is None --
    i.e. no sampling happened, so segments already came from the full,
    unpadded file and are already in true-video time.
    """
    if sample_bounds is None:
        return segments

    offset     = sample_bounds["extraction_start"]
    true_start = sample_bounds["true_start"]
    true_end   = sample_bounds["true_end"]
    at_start   = sample_bounds["at_video_start"]
    at_end     = sample_bounds["at_video_end"]

    kept = []
    for seg in segments:
        seg.start += offset
        seg.end   += offset
        for w in seg.words:
            w.start += offset
            w.end   += offset

        # A boundary at the genuine edge of the video isn't an artifact
        # of where we chose to cut -- don't drop a segment just for
        # starting at 0:00 or ending at the video's real end. 0.05s
        # tolerance absorbs float rounding, not a real grace window.
        starts_early = seg.start < true_start - 0.05 and not at_start
        ends_late    = seg.end   > true_end   + 0.05 and not at_end
        if starts_early or ends_late:
            tqdm.write(
                f" ✂️  Dropping padding-only segment [{seg.start:.1f}s-"
                f"{seg.end:.1f}s] -- outside true sample window "
                f"[{true_start:.1f}s-{true_end:.1f}s]")
            continue
        kept.append(seg)
    return kept


# ========================= ANALYSIS =========================
def analyze(audio_path, model, video_meta, substeps=None, preset=None,
            sample_seconds=None, skip_seconds=300):
    """
    substeps: optional tqdm instance with total=4 (transcribe, preprocess,
    tag+align, mutations). If given, step descriptions go there instead of
    being print()ed -- used by pipeline.py for the per-video sub-progress bar.

    preset: one of TRANSCRIBE_PRESETS' keys ("fast"/"balanced"/"accurate"),
    or None to use DEFAULT_TRANSCRIBE_PRESET. An unrecognized string falls
    back to the default with a warning rather than raising -- a typo in a
    CLI arg shouldn't crash a run that's already underway.

    sample_seconds: if given, only `sample_seconds` of the audio are
    transcribed/analyzed, starting `skip_seconds` in (see
    sample_audio_window() above for why: skip past intros/ads, and why
    this trims the file rather than using clip_timestamps). None (the
    default) processes the full file -- unchanged behavior unless
    explicitly requested.

    skip_seconds: where the sample window starts, in seconds (default
    300 = 5 min). Only matters when sample_seconds is set.
    """
    def _step(label):
        if substeps is not None:
            substeps.set_description(label)
            substeps.update(1)
        else:
            print(f" {label}...")

    preset_name = preset or DEFAULT_TRANSCRIBE_PRESET
    if preset_name not in TRANSCRIBE_PRESETS:
        print(f" ⚠️ Unknown transcription preset {preset_name!r} -- "
              f"falling back to {DEFAULT_TRANSCRIBE_PRESET!r}. "
              f"Valid presets: {', '.join(TRANSCRIBE_PRESETS)}")
        preset_name = DEFAULT_TRANSCRIBE_PRESET
    preset_kwargs = dict(TRANSCRIBE_PRESETS[preset_name])
    if preset_kwargs.get("best_of") is None:
        preset_kwargs.pop("best_of", None)  # faster-whisper expects it omitted, not None

    audio_path, sample_bounds = sample_audio_window(
        audio_path, sample_seconds, skip_seconds=skip_seconds)

    start_time = time.time()
    _step("Transcribing")
    transcribe_kwargs = {
        "language": "cy", "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": dict(threshold=0.6, min_silence_duration_ms=800,
                               max_speech_duration_s=20),
        "no_repeat_ngram_size": 5,
        "compression_ratio_threshold": 2.0,
        "no_speech_threshold": 0.6,
        "initial_prompt": "Cymraeg. Welsh language speech from S4C, Hansh, "
                          "Rownd a Rownd or Welsh YouTube.",
        **preset_kwargs,
    }
    segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    # PATCH: info.duration is the actual audio length (seconds) faster-whisper
    # detected in the file -- distinct from `dur` at the end of this function,
    # which is wall-clock processing time. Captured here so it can be stamped
    # onto every output row below, letting corpus_analyzer.py answer "how many
    # minutes of video did this corpus actually cover?" without re-opening
    # any audio files.
    # PATCH: when sampled, this must be the TRUE (unpadded) sample-window
    # length, not info.duration -- info.duration is the length of the
    # padded clip actually fed to Whisper, which is `pad_seconds` longer
    # on each side than what was really sampled. Using info.duration here
    # would silently inflate every "minutes of corpus covered" total in
    # corpus_analyzer.py by 2*pad_seconds per sampled video.
    if sample_bounds is not None:
        video_duration_seconds = round(
            sample_bounds["true_end"] - sample_bounds["true_start"], 2)
    else:
        video_duration_seconds = round(getattr(info, "duration", 0.0) or 0.0, 2)
    segments = list(segments)
    # PATCH: shift clip-relative timestamps to true-video time and drop
    # padding-only partial segments BEFORE any other filtering -- see
    # _shift_and_trim_padded_segments()'s docstring. A no-op when this
    # video wasn't sampled (sample_bounds is None).
    segments = _shift_and_trim_padded_segments(segments, sample_bounds)
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

    # PATCH: total content-word count for this video, captured the same
    # way as video_duration_seconds above -- once per video, then stamped
    # onto every output row below. Nothing before this tracked total
    # words spoken at all, only mutation-trigger-context counts, so
    # anything wanting a genuine rate-per-speech-volume metric (e.g.
    # code-switch rate against total words rather than against trigger
    # contexts only) had no denominator to use. preprocess_segment()
    # already drops punctuation-only tokens (see its docstring), so this
    # is a content-word count, not a raw whitespace-split token count.
    video_word_count = len(all_preprocessed)

    # PATCH: true whole-transcript code-switch count, not just trigger-
    # adjacent instances. is_english_code_switch() already runs on every
    # single word in this video -- it's called inside get_welsh_lemma(),
    # which the word_rows loop below calls for every content word (one of
    # the four per-video CSVs this pipeline already writes) -- but that
    # result was only ever used internally to decide whether to skip the
    # Cysill/simplemma lookup, never surfaced or counted. No new
    # transcription, tagging, or API work needed: just counting a
    # classification that was already happening on the full word stream.
    # This is a separate concept from mutation_rows' own "is_code_switch"
    # column, which flags only whether a mutation-trigger TARGET word was
    # English (relevant to mutation detection itself, unchanged here) --
    # this new count is corpus-wide, across every word in the video.
    video_codeswitch_word_count = sum(
        1 for w in all_preprocessed
        if is_english_code_switch(w["word"], None)
    )

    # Enrich: Cysill + spaCy via unified gap-tolerant alignment
    _step("Tagging + aligning (Cysill + spaCy)")
    # PATCH: stable per-video key for enrich_words()'s chunk-level
    # checkpointing (see that function's own docstring for why this
    # matters -- an interrupted run inside "Tagging + aligning" used to
    # lose ALL of a video's tagging progress, not just the stuck chunk).
    # Preference order: video id (queue videos always have one) -> url
    # (queue videos' actual source URL, or the local mp3 path for local
    # files, per the meta dict welsh_pipeline.py builds for that branch)
    # -> audio_path itself as a last resort so checkpointing degrades
    # gracefully instead of silently disabling itself if video_meta ever
    # arrives without either field. Whichever key is used, re-running the
    # SAME video (same id/url/path) on a later run is what makes
    # resumption find its checkpoint again -- a genuinely different video
    # naturally gets a different key and starts fresh.
    checkpoint_key = str(video_meta.get("id") or video_meta.get("url") or audio_path)
    enriched = enrich_words(all_preprocessed, checkpoint_key=checkpoint_key)

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
            "video_word_count":     video_word_count,
            "video_codeswitch_word_count": video_codeswitch_word_count,
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
                # PATCH: carries enrich_words()'s locally_resolved flag into
                # the cache file so rerun_rules.py (and any future re-analysis)
                # can tell a genuine Cysill answer apart from a Bangor-lexicon/
                # code-switch substitute -- see mutation_engine.py's
                # compute_confidence()/_build_row() for why that distinction
                # matters for corroboration-metric correctness, not just
                # bookkeeping.
                "locally_resolved":     bool(w.get("locally_resolved")),
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
            "video_word_count": video_word_count,
            "video_codeswitch_word_count": video_codeswitch_word_count,
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
            "locally_resolved":     bool(w.get("locally_resolved")),
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