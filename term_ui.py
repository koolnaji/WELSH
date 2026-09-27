"""
term_ui.py
==========
One tiny shared helper: how wide should the interactive interface
(banner, separator rules, wrapped prompt text) draw itself.

Both icelandic_text_extractor.py's interactive_menu() and
boilerplate_review.py's run_review() print banner/separator lines and
long yes/no question text -- previously each hardcoded a flat 60 chars
for the rules, and just let the terminal auto-wrap the long question
strings at whatever the full terminal width happened to be. Centralized
here for the same reason DETECTION_MODEL etc. live in one place in
boilerplate_detector.py rather than being copy-pasted: one spot to
change the sizing rule, not two copies that can drift apart.

Deliberately split into three small functions rather than one "print a
box" helper -- callers that just want a separator rule call rule();
callers that need to wrap a paragraph of prompt text call wrap(); a
caller that needs the raw number (e.g. to size a fragment preview) calls
half_width() directly.
"""

import shutil
import textwrap

# Floor keeps the banner/rules/wrapped text from collapsing into
# something illegible in a genuinely narrow split-pane terminal. Ceiling
# keeps line length from sprawling past comfortable reading width just
# because the terminal happened to be maximized on a huge monitor --
# "half the screen" is the target, not "however wide the screen is".
MIN_WIDTH = 40
MAX_WIDTH = 100


def half_width():
    """Half the current terminal's column count, clamped to
    [MIN_WIDTH, MAX_WIDTH].

    shutil.get_terminal_size() falls back to its `fallback` argument
    (80x24) when stdout isn't a real terminal -- piped output, a
    redirected log file, run under a debugger, etc. -- so this never
    raises; it just quietly assumes a normal-sized terminal in those
    cases, same as it would have before this module existed.
    """
    columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(MIN_WIDTH, min(MAX_WIDTH, columns // 2))


def rule(char="="):
    """A single separator line, char * half_width() -- the direct
    replacement for every literal '=' * 60 / '-' * 60 in the
    interactive menu and the boilerplate review loop."""
    return char * half_width()


def wrap(text, indent=""):
    """Wraps text to half_width(), for the long prompt/question strings
    that previously relied on the terminal's own auto-wrap (which, at
    full terminal width, doesn't keep the interface inside a
    half-screen column the way the rules/banner now do).

    indent is prepended to every wrapped line (and subtracted from the
    fill width first, so the wrapped text plus indent still fits within
    half_width() rather than overflowing it)."""
    width = max(MIN_WIDTH, half_width() - len(indent))
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)