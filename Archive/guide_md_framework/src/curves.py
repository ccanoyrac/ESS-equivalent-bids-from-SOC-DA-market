"""Model 3 (GUIDE.md section 6): turn a period's marginal-value function `v_t` into a
price/quantity block stack, starting at the expected SOC and working outward.

A note on GUIDE.md section 6's own wording: it describes the discharge block as taking
SOC "from s to s - Delta/beta, delivering Delta*beta MWh". Working the SOC dynamics
backwards (`s_new = s - d/beta`, `delivered = d`) shows that "delivering Delta*beta MWh"
is only consistent with a SOC *step* of Delta (not Delta/beta) -- i.e. `d = Delta*beta`.
The charge block's own wording ("s to s+Delta, buying Delta/alpha MWh") is internally
consistent as written. So both blocks are implemented here with a SOC step of `Delta`
(the DP grid spacing) on the *storage* side -- the natural choice anyway, since it makes
each block's boundaries exactly consecutive DP grid points and its price exactly the
already-computed `v_t` at the block's starting grid index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.baldick import StorageSpec
from src.dp import ValueFunction


@dataclass(frozen=True)
class Block:
    price: float  # EUR/MWh
    quantity: float  # MWh at the grid (delivered, for discharge; bought, for charge)
    soc_from: float  # MWh
    soc_to: float  # MWh


def build_curve(
    vf: ValueFunction, t: int, storage: StorageSpec, s_expected: float
) -> tuple[list[Block], list[Block]]:
    """Returns `(discharge_blocks, charge_blocks)` for period `t`, each sorted from the
    block nearest `s_expected` outward. Discharge prices increase with quantity (v_t
    rises as SOC falls, since V is concave); charge prices decrease with quantity --
    exactly GUIDE.md section 6's shape, as long as `dp.is_concave(vf.V[t])` holds.
    """
    grid = vf.soc_grid
    delta = vf.delta
    n = len(grid)
    i0 = int(np.argmin(np.abs(grid - s_expected)))

    discharge_blocks: list[Block] = []
    i = i0
    while i > 0:
        v = vf.v[t, i]
        price = v / storage.beta + storage.zeta
        quantity = delta * storage.beta
        discharge_blocks.append(Block(price=price, quantity=quantity, soc_from=grid[i], soc_to=grid[i - 1]))
        i -= 1

    charge_blocks: list[Block] = []
    i = i0
    while i < n - 1:
        v = vf.v[t, i]
        price = storage.alpha * v - storage.epsilon
        quantity = delta / storage.alpha
        charge_blocks.append(Block(price=price, quantity=quantity, soc_from=grid[i], soc_to=grid[i + 1]))
        i += 1

    return discharge_blocks, charge_blocks


def single_price_curve(mu_t: float, storage: StorageSpec) -> tuple[list[Block], list[Block]]:
    """The degenerate one-block curve implied by a single dual value `mu_t`: offer the
    full discharge power at `mu_t + zeta`, bid the full charge power at `mu_t - epsilon`.
    This is what GUIDE.md section 9.2's perfect-foresight equivalence test needs -- at
    k=0, `v_t(s*_t) = mu_t` along the optimal trajectory, so a single block at `mu_t`
    (rather than the full DP/multi-block machinery) is the exactly-equivalent curve.
    """
    discharge = [Block(price=mu_t + storage.zeta, quantity=storage.d_max, soc_from=np.nan, soc_to=np.nan)]
    charge = [Block(price=mu_t - storage.epsilon, quantity=storage.c_max, soc_from=np.nan, soc_to=np.nan)]
    return discharge, charge
