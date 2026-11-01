from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalValueFunction:
    """The bid benefit b(.) of deviating the end-of-horizon SOC from sbar_T (paper Section 4.3).

    b is the definite integral of a non-increasing marginal willingness-to-pay function w,
    so b is concave with b(0) = 0 -- in a dynamic-programming reading, b is the value-to-go
    beyond the end of the market horizon (paper footnote 14). Here b is represented as
    piecewise-linear: `breakpoints` are the deviation values bounding each segment (must
    include 0.0, the reference point) and `marginal_values[k]` is w on the segment between
    breakpoints[k] and breakpoints[k+1]. Concavity requires marginal_values to be non-increasing.
    """

    breakpoints: tuple[float, ...]
    marginal_values: tuple[float, ...]

    def __post_init__(self) -> None:
        bp, mv = self.breakpoints, self.marginal_values
        if len(bp) != len(mv) + 1:
            raise ValueError("breakpoints must have exactly one more element than marginal_values")
        if any(bp[i] >= bp[i + 1] for i in range(len(bp) - 1)):
            raise ValueError("breakpoints must be strictly increasing")
        if 0.0 not in bp:
            raise ValueError("breakpoints must include 0.0 (the reference SOC deviation)")
        if any(mv[i] < mv[i + 1] for i in range(len(mv) - 1)):
            raise ValueError("marginal_values must be non-increasing for b to be concave")

    def value(self, deviation: float) -> float:
        """b(deviation) = integral of w from 0 to `deviation`, evaluated segment by segment."""
        idx0 = self.breakpoints.index(0.0)
        total = 0.0
        if deviation >= 0:
            remaining = deviation
            for k in range(idx0, len(self.marginal_values)):
                width = self.breakpoints[k + 1] - self.breakpoints[k]
                take = min(width, remaining)
                total += take * self.marginal_values[k]
                remaining -= take
                if remaining <= 0:
                    break
        else:
            remaining = -deviation
            for k in range(idx0 - 1, -1, -1):
                width = self.breakpoints[k + 1] - self.breakpoints[k]
                take = min(width, remaining)
                total -= take * self.marginal_values[k]
                remaining -= take
                if remaining <= 0:
                    break
        return total

    def scaled(self, multiplier: float) -> TerminalValueFunction:
        """A copy of this bid with every marginal value scaled by `multiplier` -- e.g.
        to model an owner raising or lowering its whole willingness-to-pay schedule
        between market clearings. Breakpoints (deviation, in MWh) are unchanged; only
        marginal_values ($/MWh) are scaled, so a positive multiplier preserves the
        non-increasing order that keeps b concave.
        """
        return TerminalValueFunction(
            breakpoints=self.breakpoints,
            marginal_values=tuple(mv * multiplier for mv in self.marginal_values),
        )

    def lp_segments(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Split b into (upside, downside) segments as (width, marginal_value) pairs, each
        ordered starting from the segment adjacent to the Δ=0 reference point. Used to embed
        b as a linear objective term via non-negative segment-fill variables: because b is
        concave, an LP relaxation with no explicit ordering constraint fills the best segment
        first automatically at optimality.
        """
        idx0 = self.breakpoints.index(0.0)
        upside = [
            (self.breakpoints[k + 1] - self.breakpoints[k], self.marginal_values[k])
            for k in range(idx0, len(self.marginal_values))
        ]
        downside = [
            (self.breakpoints[k + 1] - self.breakpoints[k], self.marginal_values[k])
            for k in range(idx0 - 1, -1, -1)
        ]
        return upside, downside
