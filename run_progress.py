"""
run_progress.py
===============
The per-video progress bar under "Videos" in welsh_pipeline.py's queue run.

It used to be four fixed steps (transcribe, pre-process, tag, detect), each
worth 25% whatever it cost. Now every phase measures its own real work:

    Transcribing          seconds of audio transcribed
    Checking language     segments checked (the English filter)
    Tagging               words tagged by Cysill + spaCy
    Looking up lemmas     words through the word/lemma tables
    Detecting mutations   (quick)

welsh_pipeline.py passes a VideoProgress to corpus_ops.analyze() as its
`substeps` (the step labels arrive through set_description), and one-line
hooks in corpus_ops.py / mutation_engine.py call phase()/advance() here.
Those hook lines end in "# unversioned", so they don't change the pipeline
version (corpus_io._versioned_bytes). With no bar active -- Siarad runs,
reruns, phrase tests -- every hook is a no-op.

Transcript runs (menu 3: Siarad / Patagonia / CorCenCC / news) get the same
two bars since 2026-10-06: an overall one per run (start_run / item /
item_done / end_run -- one-line calls, so they can sit in the versioned
corpus_*.py files tagged "# unversioned") and this per-document one under
it. While a run bar is up, print() goes through tqdm.write so the per-file
summary lines don't break the bars. files() is a plain bar for the slow
file-reading loops (analyzer, formality scores, audit).
"""
import sys

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

_current = None   # the active VideoProgress, if any
_run = None       # (overall bar, the real sys.stdout) of a transcript run

_FORMAT = ("{desc:<22} {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f} {unit} "
           "[{elapsed}<{remaining}]")


class VideoProgress:
    """Drop-in for the old `tqdm(total=4, ...)` sub-bar."""

    def __init__(self):
        self._bar = None

    def __enter__(self):
        global _current
        self._bar = tqdm(total=1, desc="Starting", leave=False, bar_format=_FORMAT, unit="")
        _current = self
        return self

    def __exit__(self, *exc):
        global _current
        _current = None
        self._bar.close()
        return False

    # --- the interface analyze() already uses for its step labels ---
    def set_description(self, label):
        self.phase(label)

    def update(self, n=1):
        pass   # step counting is replaced by the phases' own totals

    # --- phases ---
    def phase(self, label=None, total=None, unit=""):
        """Starts a phase. label=None keeps the current label (a hook that
        only knows the size of the work, not its name)."""
        self._bar.reset(total=max(float(total or 1), 1e-9))
        self._bar.unit = unit
        if label:
            self._bar.set_description_str(label)
        self._bar.refresh()

    def advance(self, n=1):
        self._bar.update(n)


def phase(label=None, total=None, unit=""):
    if _current is not None:
        _current.phase(label, total, unit)


def advance(n=1):
    if _current is not None:
        _current.advance(n)


def step(label):
    """The `step` callback for analyze_segments: names the phase on the
    per-document bar, or prints it as before when no bar is up."""
    if _current is not None:
        _current.phase(label)
    else:
        print(f" {label}...")


# --- transcript runs (menu 3) ---
def start_run(total, unit):
    """Overall bar for a run of `total` documents. Prints go through
    tqdm.write (to stderr, where the bars are -- the same console window)
    until end_run()."""
    global _run
    end_run()
    bar = tqdm(total=total, desc="Overall", unit=unit, dynamic_ncols=True,
               bar_format="{desc:<22} {percentage:3.0f}%|{bar}| {n}/{total} {unit} "
                          "[{elapsed}<{remaining}]")
    _run = (bar, sys.stdout)
    sys.stdout = DummyTqdmFile(sys.stderr)


def item(label=""):
    """Opens the per-document bar for the next document."""
    if _run is None:
        return
    if _current is not None:
        _current.__exit__(None, None, None)
    _run[0].set_postfix_str(str(label)[:40], refresh=False)
    VideoProgress().__enter__()


def item_done():
    """The document just processed counts on the overall bar."""
    if _run is not None:
        _run[0].update(1)


def end_run():
    """Closes both bars and puts print() back. Safe to call twice."""
    global _run
    if _current is not None:
        _current.__exit__(None, None, None)
    if _run is not None:
        bar, real_stdout = _run
        _run = None
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = real_stdout
        bar.close()


_check = None     # the "Checking what's done" bar, if up


def checking(total, desc="Checking what's done"):
    """Bar for the skip-what's-done pass; the _already_done functions call
    tick() (a no-op when this bar isn't up)."""
    global _check
    checked()
    _check = tqdm(total=total, desc=desc, unit="files", leave=False, dynamic_ncols=True)


def tick():
    if _check is not None:
        _check.update(1)


def checked():
    global _check
    if _check is not None:
        _check.close()
        _check = None


def files(iterable, desc, unit="files", total=None):
    """A plain bar over a slow loop (reading result CSVs)."""
    return tqdm(iterable, desc=desc, unit=unit, total=total, leave=False,
                dynamic_ncols=True)


def track_transcription(segments, info):
    """Passes faster-whisper's segment generator through, advancing the bar by
    the audio each segment covers (the work happens as the generator is
    consumed). The tail after the last segment -- silence VAD skipped -- is
    added at the end so the phase finishes at 100%."""
    total = float(getattr(info, "duration", 0.0) or 0.0)
    phase(total=total, unit="s of audio")
    done = 0.0
    for seg in segments:
        end = min(float(seg.end), total) if total else float(seg.end)
        if end > done:
            advance(end - done)
            done = end
        yield seg
    if total > done:
        advance(total - done)
