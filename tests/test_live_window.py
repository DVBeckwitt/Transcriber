from __future__ import annotations

import unittest

from transcriber.live_window import format_caption_text
from transcriber.live_wlk import CaptionState


class LiveWindowFormattingTests(unittest.TestCase):
    def test_caption_text_keeps_last_committed_lines_and_partial(self) -> None:
        state = CaptionState(
            committed_lines=("one", "two", "three", "four"),
            partial_line="five",
        )

        text = format_caption_text(state, max_committed_lines=3)

        self.assertEqual(text, "two\nthree\nfour\nfive")

    def test_caption_text_omits_empty_partial(self) -> None:
        state = CaptionState(committed_lines=("hello",), partial_line="")

        text = format_caption_text(state, max_committed_lines=6)

        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()
