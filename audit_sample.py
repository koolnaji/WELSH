"""
Precision audit for the five detection branches.

    python audit_sample.py              # draw the samples (Siarad only)
    python audit_sample.py --source all # ... or every source
    python audit_sample.py --score      # score the filled-in verdicts

Sampling: for every branch, the evaluable rows (latest run per video, same
rule as corpus_analyzer.load_branch_rows) are split into an "erosion" and a
"correct" stratum and a fixed-seed random sample is drawn from each (the
whole stratum when it is smaller than the sample size). Each sampled row
gets the utterance it came from, the utterances either side, and -- for
Siarad -- the transcribers' English gloss, written to
analysis/audit/audit_<branch>.csv.

Verdicts go in the "verdict" column:
    ok       the detector's label is right (a real context, and the speaker
             really did / didn't apply the rule)
    wrong    a real context for the rule, but the label is flipped
             (flagged erosion is actually correct, or the reverse)
    invalid  not a context for this rule at all (misparse, wrong trigger,
             homonym, transcription artefact) -- should not be counted
    unsure   can't tell; left out of the scoring

--score turns the verdicts into (a) the precision of each label and (b) a
corrected erosion rate per branch: flagged erosions that hold up plus
"correct" rows that were really erosions, over the contexts that are real.
The 95% interval on the corrected rate is a bootstrap over the audited rows.

Re-running the sampler keeps every verdict: the sample is seeded, so the
same data gives the same rows, and verdicts are carried over by audit_id
(checked against the row's following word). Verdicts in
claude_verdicts_<branch>.csv are filled into rows that have none yet. If
the data changed and saved verdicts no longer match, the file is left alone
unless --force is given.
"""
import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from corpus_io import MUT_DIR, OUT_DIR
from mutation_tables import EVALUABLE_STATUSES

# Round 1 (2026-09-29, Siarad, version 966c440c62) lives in analysis/audit/.
# Round 2 (decided 2026-10-01: ~50 Siarad + ~50 YouTube mutation rows, on the
# frozen version) gets its own folder per source, so neither overwrites the
# other or round 1: analysis/audit2/<siarad|youtube|...>/.
AUDIT_ROOT = OUT_DIR / "audit2"
AUDIT_DIR = AUDIT_ROOT / "siarad"   # set per source by _use_source()
SEED = 20261001
VERDICTS = ("ok", "wrong", "invalid", "unsure")


def _use_source(source):
    global AUDIT_DIR
    AUDIT_DIR = AUDIT_ROOT / re.sub(r"[^a-z0-9-]+", "_", source.lower())
    return AUDIT_DIR


# name, file prefix, sample size per stratum, detail columns shown to the reviewer
# For the quantifier branch the "erosion" stratum is the singular CANDIDATES
# (status erosion_unverified -- quantifier_engine never counts them on its
# own, so the branch reads 0% by construction until they're judged): verdict
# "ok" = a real slip, "wrong" = fine (a mass/degree reading).
BRANCHES = [
    ("mutation", "mutations", 25,
     ["trigger_word", "following_word", "lemma", "rule", "expected_mutation", "mutation_found",
      "status", "cysill_pos", "spacy_pos", "spacy_dep", "detection_source", "note"]),
    ("prep", "prep_mutations", 20,
     ["preposition", "person", "surface_form", "following_pronoun", "expected_forms",
      "mutation_found", "status", "before_verb_noun", "note"]),
    ("numeral", "numeral_mutations", 20,
     ["numeral_surface", "following_word", "number_found", "number_source", "status", "note"]),
    ("rhai", "plural_mutations", 20,
     ["trigger_word", "following_word", "number_found", "number_source", "status", "note"]),
    ("quantifier", "quantifier_mutations", 30,
     ["quantifier_surface", "following_word", "noun_lemma", "number_found",
      "is_loan_quantifier", "status", "note"]),
]

# CHAT time bullets are wrapped in \x15 control characters, hence \D*
BULLET_RE = re.compile(r"(\d+)_(\d+)\D*$")
# the column a verdict file's "check" value must match, so a verdict can't
# land on a different row if the sample ever changes
CHECK_COL = {"mutation": "following_word", "prep": "surface_form", "numeral": "following_word",
             "rhai": "following_word", "quantifier": "following_word"}
STAMP_RE = re.compile(r"\d{8}_\d{6}")


def _read(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _stamp(path):
    m = STAMP_RE.search(path.name)
    return m.group(0) if m else ""


def load_rows(prefix, source):
    """Evaluable rows of one branch, latest run per video, with the folder
    each row came from (needed to pull its context)."""
    files = [p for p in MUT_DIR.rglob(f"{prefix}_*.csv")
             if "_deleted" not in p.parts and not p.name.endswith("_precaption_backup.csv")
             and not p.name.startswith("rerun_candidate_")]
    if prefix == "mutations":
        by_dir = {}
        for p in files:
            by_dir.setdefault(p.parent, []).append(p)
        files = []
        for fs in by_dir.values():
            corroborated = [f for f in fs if f.name.startswith("mutations_corroborated_")]
            original = [f for f in fs if f.name.startswith("mutations_original_")]
            files.extend(corroborated or original or fs)
    frames = []
    for p in files:
        d = _read(p)
        if d.empty or "video_url" not in d.columns or "is_erosion" not in d.columns:
            continue
        frames.append(d.assign(_run=_stamp(p), folder=str(p.parent)))
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["_run"] == rows.groupby("video_url")["_run"].transform("max")]
    candidate = (rows["status"] == "erosion_unverified") if prefix == "quantifier_mutations" \
        else pd.Series(False, index=rows.index)
    rows = rows[rows["status"].isin(EVALUABLE_STATUSES) | candidate]
    candidate = candidate.loc[rows.index]
    if source == "youtube":     # every Whisper video: their "source" is the channel URL
        keep = rows["source"].str.startswith("http")
        rows, candidate = rows[keep], candidate[keep]
    elif source != "all":
        keep = rows["source"] == source
        rows, candidate = rows[keep], candidate[keep]
    rows = rows.assign(stratum=np.where((rows["is_erosion"].str.lower() == "true") | candidate,
                                        "erosion", "correct"))
    return rows.drop(columns="_run").reset_index(drop=True)


class FolderContext:
    """Segments (+ Siarad English glosses) of one output folder, cached."""
    _cache = {}

    @classmethod
    def get(cls, folder):
        if folder not in cls._cache:
            cls._cache[folder] = cls(Path(folder))
        return cls._cache[folder]

    def __init__(self, folder):
        seg_files = sorted(folder.glob("segments_*.csv"))
        self.segs = _read(seg_files[-1]) if seg_files else pd.DataFrame()
        if not self.segs.empty:
            self.segs["start"] = self.segs["segment_start"].astype(float)
            self.segs["end"] = self.segs["segment_end"].astype(float)
            self.segs = self.segs.sort_values("start").reset_index(drop=True)
        self.gloss = {}
        utt_files = sorted(folder.glob("utterances_*.csv"))
        if utt_files:
            for _, u in _read(utt_files[-1]).iterrows():
                m = BULLET_RE.search(u.get("main_tier", ""))
                if m and u.get("eng", ""):
                    self.gloss[int(m.group(1))] = u["eng"]

    def around(self, t, speaker):
        """(previous, this, next) utterance text and this one's gloss."""
        if self.segs.empty or t is None:
            return "", "", "", ""
        hit = self.segs[(self.segs["start"] <= t + 0.01) & (self.segs["end"] >= t - 0.01)]
        if len(hit) > 1 and speaker and "speaker" in hit.columns:
            same = hit[hit["speaker"] == speaker]
            hit = same if not same.empty else hit
        if hit.empty:
            i = int((self.segs["start"] - t).abs().idxmin())
        else:
            i = int(hit.index[0])

        def line(j):
            if 0 <= j < len(self.segs):
                s = self.segs.iloc[j]
                who = s.get("speaker", "")
                return f"{who}: {s['segment_text']}" if who else s["segment_text"]
            return ""
        gloss = self.gloss.get(round(self.segs.iloc[i]["start"] * 1000), "")
        return line(i - 1), line(i), line(i + 1), gloss


def _start_seconds(ts):
    try:
        return float(str(ts).split("-")[0].strip().rstrip("s"))
    except ValueError:
        return None


def _saved_verdicts(name, out):
    """Verdicts to carry into a redrawn file, keyed by (audit_id, check value):
    those already in the audit file, then any in claude_verdicts_<branch>.csv
    for rows the file has no verdict for."""
    saved = {}
    if out.exists():
        for _, r in _read(out).iterrows():
            if r.get("verdict", "").strip():
                saved[(r["audit_id"], r.get(CHECK_COL[name], ""))] = (
                    r["verdict"], r.get("reviewer", ""), r.get("verdict_note", ""), "file")
    sidecar = AUDIT_DIR / f"claude_verdicts_{name}.csv"
    if sidecar.exists():
        for _, r in _read(sidecar).iterrows():
            saved.setdefault((r["audit_id"], r["check"]),
                             (r["verdict"], "claude", r.get("verdict_note", ""), "sidecar"))
    return saved


def draw(source, force):
    _use_source(source)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Audit files for '{source}' in {AUDIT_DIR}")
    for name, prefix, n, detail in BRANCHES:
        out = AUDIT_DIR / f"audit_{name}.csv"
        saved = _saved_verdicts(name, out)
        rows = load_rows(prefix, source)
        if rows.empty:
            print(f"  [skip] {name}: no evaluable rows")
            continue
        parts = []
        for stratum, grp in rows.groupby("stratum"):
            parts.append(grp.sample(n=min(n, len(grp)), random_state=SEED))
        sample = pd.concat(parts).sort_values(["stratum", "video_url", "timestamp"])
        records = []
        for k, (_, r) in enumerate(sample.iterrows(), 1):
            prev, this, nxt, gloss = FolderContext.get(r["folder"]).around(
                _start_seconds(r["timestamp"]), r.get("speaker", ""))
            rec = {"audit_id": f"{name}-{k:03d}", "stratum": r["stratum"],
                   "video_url": r["video_url"], "speaker": r.get("speaker", ""),
                   "timestamp": r["timestamp"]}
            rec.update({c: r.get(c, "") for c in detail})
            verdict, reviewer, note, _ = saved.pop((rec["audit_id"], rec.get(CHECK_COL[name], "")),
                                                   ("", "", "", ""))
            rec.update({"context_before": prev, "utterance": this, "context_after": nxt,
                        "english_gloss": gloss, "verdict": verdict, "reviewer": reviewer,
                        "verdict_note": note})
            records.append(rec)
        lost = [k for k, v in saved.items() if v[3] == "file"]
        if lost and not force:
            print(f"  [keep] {out.name}: {len(lost)} verdicts would not match the new sample "
                  f"(the data changed?) -- file left as is; --force redraws and drops them")
            continue
        if saved:
            print(f"  ⚠️ {name}: {len(saved)} saved verdicts matched no sampled row and were dropped")
        pd.DataFrame(records).to_csv(out, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        counts = rows["stratum"].value_counts()
        n_verdicts = sum(1 for r in records if r["verdict"])
        print(f"  {name:<11} population: {counts.get('erosion', 0):>6,} erosion / "
              f"{counts.get('correct', 0):>6,} correct -> sampled {len(records)} rows "
              f"({n_verdicts} with verdicts) -> {out.name}")
    print(f"\nFill the 'verdict' column ({' / '.join(VERDICTS)}), then: python audit_sample.py --score")


def _wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _corrected_rate(E, C, v_ero, v_cor):
    """Estimated real erosions / real contexts, from verdict arrays per stratum.
    An unaudited stratum is taken at face value (all 'ok')."""
    def shares(v):
        if len(v) == 0:
            return 1.0, 0.0, 0.0
        return (v == "ok").mean(), (v == "wrong").mean(), (v == "invalid").mean()
    ok_e, wrong_e, inv_e = shares(v_ero)
    ok_c, wrong_c, inv_c = shares(v_cor)
    real_erosions = E * ok_e + C * wrong_c
    real_contexts = E * (1 - inv_e) + C * (1 - inv_c)
    return real_erosions / real_contexts if real_contexts else float("nan")


def score(source):
    _use_source(source)
    print(f"\nScoring '{source}' ({AUDIT_DIR})")
    rng = np.random.default_rng(SEED)
    print(f"{'branch':<11} {'stratum':<8} {'pop.':>7} {'judged':>6}  "
          f"{'ok':>4} {'wrong':>5} {'invalid':>7} {'unsure':>6}   label precision (95% CI)")
    summary = []
    for name, prefix, _n, _detail in BRANCHES:
        path = AUDIT_DIR / f"audit_{name}.csv"
        if not path.exists():
            continue
        audit = _read(path)
        audit["verdict"] = audit["verdict"].str.strip().str.lower()
        bad = audit[~audit["verdict"].isin(VERDICTS + ("",))]
        if not bad.empty:
            print(f"  ⚠️ {name}: unknown verdicts {sorted(bad['verdict'].unique())} -- ignored")
        pop = load_rows(prefix, source)["stratum"].value_counts()
        judged = {}
        for stratum in ("erosion", "correct"):
            v = audit.loc[(audit["stratum"] == stratum) & audit["verdict"].isin(("ok", "wrong", "invalid")),
                          "verdict"].to_numpy()
            judged[stratum] = v
            n_uns = int(((audit["stratum"] == stratum) & (audit["verdict"] == "unsure")).sum())
            k = int((v == "ok").sum())
            lo, hi = _wilson(k, len(v))
            prec = f"{k / len(v):.0%} ({lo:.0%}-{hi:.0%})" if len(v) else "not audited"
            print(f"{name:<11} {stratum:<8} {pop.get(stratum, 0):>7,} {len(v):>6}  "
                  f"{k:>4} {int((v == 'wrong').sum()):>5} {int((v == 'invalid').sum()):>7} {n_uns:>6}   {prec}")
        E, C = pop.get("erosion", 0), pop.get("correct", 0)
        raw = E / (E + C) if E + C else float("nan")
        est = _corrected_rate(E, C, judged["erosion"], judged["correct"])
        boots = [_corrected_rate(E, C,
                                 rng.choice(judged["erosion"], len(judged["erosion"])) if len(judged["erosion"]) else judged["erosion"],
                                 rng.choice(judged["correct"], len(judged["correct"])) if len(judged["correct"]) else judged["correct"])
                 for _ in range(2000)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        summary.append((name, raw, est, lo, hi))
    print(f"\n{'branch':<11} {'raw rate':>9} {'corrected':>10}   95% CI (bootstrap)")
    for name, raw, est, lo, hi in summary:
        print(f"{name:<11} {raw:>9.1%} {est:>10.1%}   {lo:.1%}-{hi:.1%}")
    print("\nWhen a stratum was sampled in full (e.g. only 4 numeral erosions), the bootstrap "
          "still resamples it, so its interval is a little wider than it needs to be.")
    print("Quantifier: its 'raw rate' counts every singular CANDIDATE; only the corrected "
          "rate (candidates judged 'ok' = real slips) is a finding.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true", help="score the filled-in audit files")
    ap.add_argument("--source", default="siarad",
                    help="siarad (default), youtube (every video), patagonia, news-narration, ..., or all")
    ap.add_argument("--force", action="store_true",
                    help="rewrite a file even if some of its verdicts no longer match the sample")
    args = ap.parse_args()
    if args.score:
        score(args.source)
    else:
        draw(args.source, args.force)


if __name__ == "__main__":
    main()
