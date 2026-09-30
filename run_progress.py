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
"""
from tqdm import tqdm

_current = None   # the active VideoProgress, if any

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
