"""Focused offline checks for the shared YouTube access coordinator."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import youtube_access


class YouTubeAccessTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        youtube_access.configure(Path(self.tempdir.name))
        youtube_access._not_before = 0.0
        youtube_access._last_request = 0.0

    def tearDown(self):
        self.tempdir.cleanup()

    def test_429_sets_persisted_shared_cooldown(self):
        with patch.object(youtube_access, "MIN_REQUEST_INTERVAL", 0), \
             patch.object(youtube_access, "MAX_IN_PROCESS_WAIT", 0), \
             patch.object(youtube_access.random, "uniform", return_value=1):
            with self.assertRaises(youtube_access.YouTubeRateLimited):
                youtube_access.call(
                    "caption discovery",
                    lambda: (_ for _ in ()).throw(RuntimeError("HTTP Error 429: Too Many Requests")),
                    max_attempts=1,
                )
        self.assertTrue((Path(self.tempdir.name) / "youtube_rate_limit.json").exists())

    def test_non_rate_limit_failure_is_retried(self):
        attempts = [0]
        def action():
            attempts[0] += 1
            if attempts[0] == 1:
                raise RuntimeError("temporary network failure")
            return "ok"
        with patch.object(youtube_access, "MIN_REQUEST_INTERVAL", 0), \
             patch("youtube_access.time.sleep"):
            self.assertEqual("ok", youtube_access.call("audio", action, max_attempts=2))
        self.assertEqual(2, attempts[0])


if __name__ == "__main__":
    unittest.main()
