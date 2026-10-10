"""PSD postprocess: stem pairing, natural order, quiet skips.

Pure-logic side (no images, no pepeline): matched stems pair first, leftovers
pair in OS-like order, a stem missing on either side is skipped — never an
error.
"""

from __future__ import annotations

import sys

from _harness import Checks, finish

from reline_ws.postprocess import natural_key, pair_stems


def main() -> int:
    checks = Checks("postprocess")

    checks.eq(
        "numeric runs sort as numbers",
        sorted(["img10.png", "img2.png", "img1.png"], key=natural_key),
        ["img1.png", "img2.png", "img10.png"],
    )
    checks.eq(
        "letter case does not decide the order",
        sorted(["B.png", "a.png"], key=natural_key),
        ["a.png", "B.png"],
    )
    checks.eq("no sources means no pairs", pair_stems({"a": "x"}, {}), [])
    checks.eq("no output means no pairs", pair_stems({}, {"a": "x"}), [])
    checks.eq("both sides empty", pair_stems({}, {}), [])

    checks.eq(
        "unpaired tails zip by position, extras dropped",
        [stem for stem, _, _ in pair_stems({"a": "x", "c": "y"}, {"b": "z"})],
        ["a"],
    )

    checks.eq(
        "leftover pairing walks both sides in order until one runs out",
        [
            (proc, src)
            for _, proc, src in pair_stems(
                {"p1": "o/p1", "p2": "o/p2", "p3": "o/p3"},
                {"s1": "s/s1", "s2": "s/s2"},
            )
        ],
        [("o/p1", "s/s1"), ("o/p2", "s/s2")],
    )

    return finish(checks)


if __name__ == "__main__":
    sys.exit(main())
