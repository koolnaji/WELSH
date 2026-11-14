"""Tests for the full (all-branch) rerun in mutation_rerun_rules.py.

Detection itself (corpus_ops.rows_from_tagged_cache) is replaced by a stub
returning fixed rows, so these check what the rerun does around it:
stacking, diffing, dry run vs --commit, review carry-over, and keeping
stretches that have no tagged file.

    python -m pytest tests/test_rerun_rules.py -v
"""
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mutation_rerun_rules as rr  # noqa: E402
import output_merge as om  # noqa: E402

URL = "https://www.youtube.com/watch?v=abcdefghijk"
STAMP = "20260930_000000"
VIDEO = {"video_title": "Vid", "video_url": URL, "source": "yt"}
COUNTS = {"video_duration_seconds": 60, "video_word_count": 50,
          "video_codeswitch_word_count": 5}


def _segment(a, b, version):
    return {**VIDEO, **COUNTS, "segment_start": a, "segment_end": b,
            "segment_text": f"seg {a}", "pipeline_version": version}


def _word(t, version):
    return {**VIDEO, "word": f"w{t}", "word_start": t, "word_end": t + 0.5,
            "pipeline_version": version}


def _mutation(t, following, status, version, **extra):
    return {"timestamp": f"{t}s - {t + 0.5}s", "trigger_word": "yn", "rule": "word_trigger",
            "following_word": following, "status": status,
            "is_erosion": str(status == "erosion"), **VIDEO, **COUNTS,
            "pipeline_version": version, **extra}


def _prep(t, version):
    return {"timestamp": f"{t}s - {t + 0.5}s", "preposition": "i", "following_pronoun": "nhw",
            "status": "erosion", "is_erosion": "True", **VIDEO, **COUNTS,
            "pipeline_version": version}


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", quoting=1)


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


@pytest.fixture
def folder(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(om, "RUNS_DIR", runs)
    monkeypatch.setattr(om, "DELETED_DIR", runs / "_deleted")
    monkeypatch.setattr(rr, "RUNS_DIR", runs)
    f = runs / f"{STAMP}_youtube-1" / "Vid"
    f.mkdir(parents=True)
    return f


def _existing(folder, segments, words, mutations):
    _write(folder / f"segments_{STAMP}_Vid.csv", segments)
    _write(folder / f"words_{STAMP}_Vid.csv", words)
    _write(folder / f"mutations_original_{STAMP}_Vid.csv", mutations)


def _stub_detection(monkeypatch, folder, rows_by_key, window, speakers=None):
    """A tagged file whose 'detection' returns rows_by_key."""
    (folder / f"tagged_{STAMP}_Vid.json.gz").write_bytes(b"")
    segs = rows_by_key.get("segments", [])
    data = {"window": window, "tagged_at": "2026-09-30T00:00:00", "video_meta": VIDEO,
            **COUNTS,
            "segments": [[0, 0, s["segment_start"], s["segment_end"], s["segment_text"],
                          (speakers or [None] * len(segs))[i]] for i, s in enumerate(segs)],
            "words": [{"start": w["word_start"], "_seg_id": 0} for w in rows_by_key.get("words", [])]}
    rows = tuple(rows_by_key.get(k, []) for k in rr.OUTPUT_KEYS)
    fake = types.ModuleType("corpus_ops")
    fake.rows_from_tagged_cache = lambda path: (rows, data)
    monkeypatch.setitem(sys.modules, "corpus_ops", fake)


def test_dry_run_reports_and_writes_candidates_only(folder, monkeypatch):
    reviewed = {"manual_reviewed": "True", "is_erosion_original": "True"}
    _existing(folder, [_segment(0, 10, "old"), _segment(20, 30, "old")],
              [_word(5, "old"), _word(25, "old")],
              [_mutation(5, "diogel", "erosion", "old"),
               _mutation(25, "mynd", "erosion", "old", **reviewed, is_erosion="False")])
    _stub_detection(monkeypatch, folder, {
        "segments": [_segment(0, 10, "new"), _segment(20, 30, "new")],
        "words": [_word(5, "new"), _word(25, "new")],
        "mutations": [_mutation(5, "diogel", "correct_mutation", "new"),
                      _mutation(25, "mynd", "erosion", "new")],
        "prep_mutations": [_prep(25, "new")],
    }, window=[0, 60])
    original = (folder / f"mutations_original_{STAMP}_Vid.csv").read_bytes()

    summary = rr.rerun_folder(folder, list(rr.BRANCHES.values()), True, commit=False)

    assert summary["files"]["mutations"]["changed"] == 1          # diogel: erosion -> correct
    assert summary["files"]["mutations"]["reviewed_changed"] == 0  # human is_erosion isn't a change
    assert summary["files"]["prep_mutations"]["added"] == 1
    assert summary["files"]["mutations"]["statuses"][("erosion", "correct_mutation")] == 1
    assert (folder / f"mutations_original_{STAMP}_Vid.csv").read_bytes() == original
    assert (folder / f"rerun_candidate_mutations_original_{STAMP}_Vid.csv").exists()
    assert (folder / f"rerun_candidate_prep_mutations_{STAMP}_Vid.csv").exists()


def test_commit_replaces_files_and_keeps_reviews(folder, monkeypatch):
    reviewed = {"manual_reviewed": "True", "is_erosion_original": "True"}
    _existing(folder, [_segment(0, 10, "old"), _segment(20, 30, "old")],
              [_word(5, "old"), _word(25, "old")],
              [_mutation(5, "diogel", "erosion", "old"),
               _mutation(25, "mynd", "erosion", "old", **reviewed, is_erosion="False")])
    _stub_detection(monkeypatch, folder, {
        "segments": [_segment(0, 10, "new"), _segment(20, 30, "new")],
        "words": [_word(5, "new"), _word(25, "new")],
        "mutations": [_mutation(5, "diogel", "correct_mutation", "new"),
                      _mutation(25, "mynd", "erosion", "new")],
        "prep_mutations": [_prep(25, "new")],
    }, window=[0, 60])

    rr.rerun_folder(folder, list(rr.BRANCHES.values()), True, commit=True)

    mutations = _read(folder / f"mutations_original_{STAMP}_Vid.csv")
    assert list(mutations["status"]) == ["correct_mutation", "erosion"]
    assert list(mutations["pipeline_version"]) == ["new", "new"]
    assert list(mutations["manual_reviewed"]) == ["", "True"]     # the review moved over
    assert list(mutations["is_erosion"]) == ["False", "False"]    # incl. the human decision
    assert set(_read(folder / f"segments_{STAMP}_Vid.csv")["pipeline_version"]) == {"new"}
    assert (folder / f"prep_mutations_{STAMP}_Vid.csv").exists()
    assert not (folder / om.REVIEWS_FILE).exists()
    assert not list(folder.glob("rerun_candidate_*"))


def test_changed_result_on_reviewed_row_is_set_aside(folder, monkeypatch):
    reviewed = {"manual_reviewed": "True", "is_erosion_original": "True"}
    _existing(folder, [_segment(20, 30, "old")], [_word(25, "old")],
              [_mutation(25, "mynd", "erosion", "old", **reviewed, is_erosion="False")])
    _stub_detection(monkeypatch, folder, {
        "segments": [_segment(20, 30, "new")], "words": [_word(25, "new")],
        "mutations": [_mutation(25, "mynd", "correct_mutation", "new")],
    }, window=[0, 60])

    summary = rr.rerun_folder(folder, ["mutations"], False, commit=True)

    assert summary["files"]["mutations"]["reviewed_changed"] == 1
    stray = _read(folder / om.REVIEWS_FILE)
    assert list(stray["following_word"]) == ["mynd"]


def test_stretch_without_tagged_file_is_kept(folder, monkeypatch):
    _existing(folder, [_segment(0, 10, "old"), _segment(100, 110, "old")],
              [_word(5, "old"), _word(105, "old")],
              [_mutation(5, "diogel", "erosion", "old"), _mutation(105, "peth", "erosion", "old")])
    _stub_detection(monkeypatch, folder, {
        "segments": [_segment(0, 10, "new")], "words": [_word(5, "new")],
        "mutations": [_mutation(5, "diogel", "erosion", "new")],
    }, window=[0, 60])

    frames, n_untagged, _ = rr.regenerate_folder(folder)

    assert n_untagged == 1
    assert list(frames["segments"]["segment_start"]) == ["0", "100"]
    assert list(frames["mutations"]["pipeline_version"]) == ["new", "old"]


def test_speakers_are_restored(folder, monkeypatch):
    _existing(folder, [_segment(0, 10, "old")], [_word(5, "old")],
              [_mutation(5, "diogel", "erosion", "old")])
    _stub_detection(monkeypatch, folder, {
        "segments": [_segment(0, 10, "new")], "words": [_word(5, "new")],
        "mutations": [_mutation(5, "diogel", "erosion", "new")],
    }, window=[0, 60], speakers=["MBL"])

    frames, _, _ = rr.regenerate_folder(folder)

    assert list(frames["segments"]["speaker"]) == ["MBL"]
    assert list(frames["mutations"]["speaker"]) == ["MBL"]


def test_folder_without_tagged_file_is_skipped(folder):
    _existing(folder, [_segment(0, 10, "old")], [_word(5, "old")],
              [_mutation(5, "diogel", "erosion", "old")])
    assert rr.rerun_folder(folder, ["mutations"], False, commit=False) is None
