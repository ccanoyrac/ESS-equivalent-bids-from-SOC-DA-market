from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageParams:
    """Technical characteristics of an ESS, as communicated to the ISO (paper Section 4).

    SOC evolves as s_t = dt*alpha*p_t - dt*beta*g_t + gamma*s_{t-1} (eq. 4), with
    alpha/beta < 1 the round-trip efficiency and gamma the per-interval self-discharge
    retention. epsilon and zeta are the charging/discharging degradation costs ($/MWh).
    """

    alpha: float
    beta: float
    gamma: float
    s_min: float
    s_max: float
    s0: float
    p_max: float = math.inf
    g_max: float = math.inf
    epsilon: float = 0.0
    zeta: float = 0.0
    name: str = "storage"

    def __post_init__(self) -> None:
        if not (self.beta > self.alpha > 0):
            raise ValueError("require beta > alpha > 0 (round-trip efficiency alpha/beta < 1)")
        if not (0 < self.gamma <= 1):
            raise ValueError("require 0 < gamma <= 1")
        if not (self.s_min <= self.s0 <= self.s_max):
            raise ValueError("require s_min <= s0 <= s_max")
        if self.p_max < 0 or self.g_max < 0:
            raise ValueError("power limits must be non-negative")
        if self.epsilon < 0 or self.zeta < 0:
            raise ValueError("degradation costs must be non-negative")

    def sbar_T(self, T: int) -> float:
        """Reference end-of-horizon SOC if the ESS neither charges nor discharges (paper eq. after 4)."""
        return (self.gamma**T) * self.s0
