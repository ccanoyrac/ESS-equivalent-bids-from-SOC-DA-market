from __future__ import annotations

from collections.abc import Sequence

from terminal_value.base import TerminalValueFunction


def piecewise_linear_bid(
    breakpoints: Sequence[float], marginal_values: Sequence[float]
) -> TerminalValueFunction:
    """Construct a general concave terminal value function from explicit breakpoints of the
    end-of-horizon SOC deviation and the non-increasing marginal willingness-to-pay on each
    segment between them (`breakpoints` must include 0.0). This is the building block for
    research direction 1 in ROADMAP.md: comparing perfect-foresight, forecast, learned, and
    misspecified terminal value functions under the same market mechanism.
    """
    return TerminalValueFunction(breakpoints=tuple(breakpoints), marginal_values=tuple(marginal_values))
