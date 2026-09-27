"""
gemini_retry.py
================
Shared retry/backoff/key-rotation loop for calling the Gemini API.

Factored out because boilerplate_detector.py and language_voices.py were
each carrying their own near-identical copy of this loop (same shape of
try/except-429/backoff/rotate-key/give-up, same shape of a module-level
"have I given up for this run" flag). That's the exact kind of
duplication this project has already been bitten by once before on the
Welsh side (BASE_DIR hardcoded three different ways, channel_register vs
register_class) -- two copies of control-flow logic that are identical
today and under no obligation to stay that way, since nothing forces
them to be edited together. This exists so there's one loop, not two
that can silently diverge on the next tweak to either one.

Does NOT own the Gemini key pool itself -- that stays in
boilerplate_detector.py (get_gemini_client / rotate_gemini_key /
mark_gemini_key_exhausted / gemini_key_count), since quota exhaustion is
genuinely shared across both callers: one Google account, one quota,
regardless of which voice made the call. What's caller-SPECIFIC and
deliberately NOT shared is each voice's own "have I given up for this
run" flag -- language ID hitting its quota shouldn't silently disable
boilerplate detection or vice versa (see language_voices.py's own module
docstring for why that independence is worth keeping). This module
takes the key-pool functions in as plain callables rather than importing
boilerplate_detector.py directly, specifically to avoid a circular
import (boilerplate_detector.py is what constructs a GeminiRetryCaller
for its own use, and language_voices.py already imports FROM
boilerplate_detector.py -- this module sits underneath both).
"""

import time


class GeminiRetryCaller:
    """One instance per calling voice (boilerplate detection, language
    ID, or any future one). Each instance gets its OWN quota_exhausted
    flag -- constructed fresh per voice, reset independently of the
    others -- while every instance still routes through the SAME
    underlying key pool via the get_client/rotate_key/mark_exhausted/
    key_count callables passed in at construction.
    """

    def __init__(self, label, get_client, rotate_key, mark_exhausted, key_count,
                 max_retries, initial_backoff_seconds, fallback_note=""):
        """label: short human-readable name for this voice, used in the
        rotation/give-up print messages (e.g. "boilerplate detection",
        "the LLM language voice").

        fallback_note: appended to the give-up message, describing what
        keeps happening without this voice (e.g. "language detection
        keeps going on the local lingua voice alone, just without a
        second opinion."). Each caller can carry its own message this
        way without duplicating the surrounding print/loop logic."""
        self.label = label
        self._get_client = get_client
        self._rotate_key = rotate_key
        self._mark_exhausted = mark_exhausted
        self._key_count = key_count
        self._max_retries = max_retries
        self._initial_backoff_seconds = initial_backoff_seconds
        self._fallback_note = fallback_note
        self.quota_exhausted = False

    def reset(self):
        """Call at the start of each top-level run (mirrors the old
        per-module reset_quota_flag() functions) -- does NOT reset the
        shared key pool itself; callers that also need that call the
        key pool's own reset separately (see boilerplate_detector.py's
        reset_quota_flag() and language_voices.py's
        reset_language_quota_flag())."""
        self.quota_exhausted = False

    def call(self, model, sample, system_prompt, max_output_tokens,
             response_mime_type="application/json"):
        """Blocking. Returns the raw API response object, or None if the
        call couldn't be completed: quota already marked exhausted this
        run, every retry+key-rotation attempt was exhausted, or a
        non-rate-limit error came back (retrying/rotating wouldn't fix
        that anyway, so this gives up on the first one rather than
        burning retries on an error that isn't going away)."""
        if self.quota_exhausted:
            return None

        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
        )
        while True:
            client = self._get_client()  # current active key -- may have
                                          # changed since the last call if
                                          # a rotation happened partway
                                          # through this run (possibly
                                          # triggered by a DIFFERENT
                                          # voice -- they share one pool)
            for attempt in range(self._max_retries):
                try:
                    return client.models.generate_content(
                        model=model, contents=sample, config=config,
                    )
                except Exception as e:
                    msg = str(e)
                    is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                    if is_rate_limit and attempt < self._max_retries - 1:
                        time.sleep(self._initial_backoff_seconds * (2 ** attempt))
                        continue
                    if not is_rate_limit:
                        return None  # not a rate-limit issue -- rotating keys wouldn't help
                    break  # persistent 429s on THIS key after max_retries

            # Current key looks genuinely exhausted (not just a transient
            # blip -- that's what the retry loop above already absorbed).
            # Mark it and try the next key in the pool before giving up.
            self._mark_exhausted()
            if self._key_count() > 1 and self._rotate_key():
                print(f"\n🔁 Gemini key exhausted -- rotating to the next key in "
                      f"GEMINI_API_KEYS and retrying ({self.label})...")
                continue

            self.quota_exhausted = True
            which = "the only configured key" if self._key_count() == 1 else "every configured key"
            note = f" {self._fallback_note}" if self._fallback_note else ""
            print(f"\n⚠️  Gemini API rate/quota limit hit repeatedly on {which} -- "
                  f"pausing {self.label} for the rest of this run.{note}")
            return None