from __future__ import annotations

from terminal_value.base import TerminalValueFunction


def constant_willingness_to_pay(w: float, cap: float = 1e7) -> TerminalValueFunction:
    """A flat marginal willingness-to-pay w ($/MWh) in either direction of deviation, so
    b(delta) = w * delta. Reproduces the linear bids used in Baldick's numerical examples
    (w = 40 and w = 55 $/MWh). `cap` bounds how far the SOC deviation can run in the LP;
    it should exceed any deviation reachable given the storage's SOC and power limits.
    """
    return TerminalValueFunction(breakpoints=(-cap, 0.0, cap), marginal_values=(w, w))
