"""Tests for output_merge.py. Every test points the module at a temporary
runs/ folder, so real output is never touched.

    python -m pytest tests/test_output_merge.py -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import output_merge as om  # noqa: E402

URL = "https://www.youtube.com/watch?v=abcdefghijk"


@pytest.fixture
def runs(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(om, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(om, "DELETED_DIR", runs_dir / "_deleted")
    return runs_dir


def _vpaths(folder, stamp, slug="Vid"):
    folder.mkdir(parents=True, exist_ok=True)
    return {key: folder / f"{prefix}_{stamp}_{slug}.csv" for key, prefix in om.DATA_FILES.items()}


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", quoting=1)


def _video_cols(version, duration, words, codeswitch):
    return {"video_title": "Vid", "video_url": URL, "source": "yt",
            "video_duration_seconds": duration, "video_word_count": words,
            "video_codeswitch_word_count": codeswitch, "pipeline_version": version}


def _segments(spans, **video):
    return [{**video, "segment_start": a, "segment_end": b, "segment_text": f"seg {a}"}
            for a, b in spans]


def _words(starts, version):
    return [{"video_title": "Vid", "video_url": URL, "word": f"w{t}", "word_start": t,
             "word_end": t + 0.5, "pipeline_version": version} for t in starts]


def _mutation(t, version, **extra):
    return {"timestamp": f"{t}s - {t + 0.5}s", "trigger_word": "yn", "rule": "word_trigger",
            "following_word": "diogel", "status": "erosion", "is_erosion": "True",
            **_video_cols(version, 240, 400, 40), **extra}


def _old_folder(runs, spans, word_starts, mutation_rows, stamp="20260101_000000"):
    folder = runs / f"{stamp}_youtube-1" / "Vid"
    paths = _vpaths(folder, stamp)
    video = _video_cols("old", 240, 400, 40)
    _write(paths["segments"], _segments(spans, **video))
    _write(paths["words"], _words(word_starts, "old"))
    _write(paths["mutations"], mutation_rows)
    (folder / "abcdefghijk_Vid_norm_sample0-240s.mp3").write_bytes(b"")
    return folder


def _new_folder(runs, spans, word_starts, mutation_rows, stamp="20260102_000000"):
    folder = runs / f"{stamp}_youtube-1" / "Vid"
    paths = _vpaths(folder, stamp)
    video = _video_cols("new", 120, 100, 10)
    _write(paths["segments"], _segments(spans, **video))
    _write(paths["words"], _words(word_starts, "new"))
    _write(paths["mutations"], mutation_rows)
    return folder, paths


def _starts(path, column):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return [float(v.split("s - ")[0]) for v in df[column]]


def test_windows_stack_and_old_folder_is_retired(runs):
    # earlier run: 0-240s; new run: 180-300s -> 0-300s
    old = _old_folder(runs, [(0, 10), (60, 70), (170, 185), (200, 210)],
                      [5, 65, 175, 205], [_mutation(5, "old"), _mutation(205, "old")])
    _, paths = _new_folder(runs, [(186, 195), (250, 260)], [190, 255], [_mutation(190, "new")])

    result = om.merge_with_previous(paths, URL, [180, 300])

    assert result["merged"] == 1
    # 200-210 lies inside the new window -> replaced; 170-185 crosses the new
    # window's edge and no new segment overlaps it -> kept
    assert _starts(paths["segments"], "segment_start") == [0, 60, 170, 186, 250]
    assert _starts(paths["words"], "word_start") == [5, 65, 175, 190, 255]
    assert _starts(paths["mutations"], "timestamp") == [5, 190]
    segments = pd.read_csv(paths["segments"], dtype=str, encoding="utf-8-sig")
    assert set(segments["video_duration_seconds"]) == {"300"}
    # 100 new words + 400 old * (3 of 4 old words kept)
    assert set(segments["video_word_count"]) == {"400"}
    assert set(segments["video_codeswitch_word_count"]) == {"40"}
    assert set(segments["pipeline_version"]) == {"old", "new"}
    manifest = json.loads((paths["segments"].parent / om.COVERAGE_FILE).read_text("utf-8"))
    assert manifest["intervals"] == [[0.0, 300.0]]
    assert manifest["video_key"] == "abcdefghijk"
    assert not old.exists()
    assert not old.parent.exists()          # its session folder was left empty
    assert (runs / "_deleted" / old.parent.name / "Vid").exists()
    assert "MERGED" in (runs / "_deleted" / "deleted_log.txt").read_text("utf-8")


def test_tagged_words_of_kept_rows_move_with_them(runs):
    old = _old_folder(runs, [(0, 10), (200, 210)], [5, 205], [_mutation(5, "old")])
    (old / "tagged_20260101_000000_Vid.json.gz").write_bytes(b"old")
    _, paths = _new_folder(runs, [(250, 260)], [255], [_mutation(255, "new")])
    om.merge_with_previous(paths, URL, [180, 300])
    assert (paths["segments"].parent / "tagged_20260101_000000_Vid.json.gz").exists()


def test_new_segment_overlapping_the_edge_segment_wins(runs):
    _old_folder(runs, [(0, 10), (170, 185)], [5, 175], [_mutation(5, "old")])
    _, paths = _new_folder(runs, [(178, 190)], [180], [_mutation(180, "new")])
    om.merge_with_previous(paths, URL, [180, 300])
    assert _starts(paths["segments"], "segment_start") == [0, 178]


def test_reviews_follow_the_new_row_or_are_saved(runs):
    reviewed = {"manual_reviewed": "True", "is_erosion_original": "True",
                "review_log": "decided is_erosion=False"}
    old_rows = [_mutation(205, "old", **{**reviewed, "is_erosion": "False"}),
                _mutation(220, "old", **{**reviewed, "following_word": "peth"})]
    _old_folder(runs, [(200, 225)], [205, 220], old_rows)
    _, paths = _new_folder(runs, [(200, 225)], [205], [_mutation(205.3, "new")])

    result = om.merge_with_previous(paths, URL, [180, 300])

    merged = pd.read_csv(paths["mutations"], dtype=str, keep_default_na=False,
                         encoding="utf-8-sig")
    assert list(merged["manual_reviewed"]) == ["True"]
    assert list(merged["is_erosion"]) == ["False"]     # the human decision
    assert list(merged["pipeline_version"]) == ["new"]
    assert result["unmatched_reviews"] == 1            # "peth" row: no new match
    stray = pd.read_csv(paths["segments"].parent / om.REVIEWS_FILE, dtype=str,
                        encoding="utf-8-sig")
    assert list(stray["following_word"]) == ["peth"]


def test_first_run_only_writes_coverage(runs):
    _, paths = _new_folder(runs, [(60, 70)], [65], [_mutation(65, "new")])
    before = paths["segments"].read_bytes()
    assert om.merge_with_previous(paths, URL, [60, 600])["merged"] == 0
    assert paths["segments"].read_bytes() == before
    manifest = json.loads((paths["segments"].parent / om.COVERAGE_FILE).read_text("utf-8"))
    assert manifest["intervals"] == [[60.0, 600.0]]


def test_captions_are_reused(runs):
    old = _old_folder(runs, [(0, 10)], [5], [_mutation(5, "old")])
    (old / "abcdefghijk.cy.vtt").write_text("WEBVTT\n", encoding="utf-8")
    new_dir = runs / "20260102_000000_youtube-1" / "Vid"
    new_dir.mkdir(parents=True)
    assert om.copy_previous_captions(URL, new_dir) == 1
    assert (new_dir / "abcdefghijk.cy.vtt").exists()


def test_video_key_ignores_url_spelling():
    assert om.video_key("https://youtu.be/abcdefghijk") == om.video_key(URL)
    assert om.video_key("siarad:davies1") == "siarad:davies1"
