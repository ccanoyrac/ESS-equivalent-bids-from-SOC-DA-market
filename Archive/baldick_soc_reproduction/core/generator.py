from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalGenerator:
    """A dispatchable resource with a fixed offer price and constant capacity limits."""

    name: str
    cost: float
    p_max: float
    p_min: float = 0.0

    def __post_init__(self) -> None:
        if self.p_min < 0 or self.p_max < self.p_min:
            raise ValueError("require 0 <= p_min <= p_max")


@dataclass(frozen=True)
class RenewableGenerator:
    """A zero-marginal-cost resource whose available capacity varies by interval (RAC)."""

    name: str
    capacity: tuple[float, ...]
    cost: float = 0.0

    def __post_init__(self) -> None:
        if any(c < 0 for c in self.capacity):
            raise ValueError("renewable available capacity (RAC) must be non-negative")
