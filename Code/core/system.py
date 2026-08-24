from __future__ import annotations

from dataclasses import dataclass

from core.generator import RenewableGenerator, ThermalGenerator
from core.storage import StorageParams


@dataclass(frozen=True)
class System:
    """A copperplate (single-node) market with fixed demand, any number of thermal and
    renewable generators, and any number of ESS -- paper Section 4.7's stylized system
    generalized per ROADMAP.md stage 2 to multiple participants of each type."""

    demand: tuple[float, ...]
    thermal: tuple[ThermalGenerator, ...]
    renewables: tuple[RenewableGenerator, ...]
    storages: tuple[StorageParams, ...]
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if any(d < 0 for d in self.demand):
            raise ValueError("demand must be non-negative")
        if not self.storages:
            raise ValueError("System requires at least one storage unit")
        for gen in self.renewables:
            if len(gen.capacity) != self.T:
                raise ValueError(f"renewable '{gen.name}' capacity length must equal T={self.T}")

    @property
    def T(self) -> int:
        return len(self.demand)
