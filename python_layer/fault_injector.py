# -*- coding: utf-8 -*-
# =============================================================================
#  SwarmRaft Fault Injector
#  ---------------------------------------------------------------------------
#  Edit ACTIVE_FAULT at the bottom to configure the simulation scenario.
#  swarm_sim.py reads this at startup -- edit nothing else.
#
#  FailureMode options:
#     RANDOM_ATTACK    -- random drone, instant destruction
#     RANDOM_COLLISION -- random drone, spin-out crash over ~20 frames
#     TARGETED_ATTACK  -- you specify the drone ID
#     MECHANICAL       -- jitters ~30 frames then goes offline
#
#  FormationType options:
#     RING     -- circular perimeter (classic)
#     DIAMOND  -- diamond / rhombus perimeter
#     GRID     -- rectangular 4xN grid
#     V_SHAPE  -- V-formation with leading tip
#     RANDOM   -- picked randomly each run (default)
# =============================================================================

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class FailureMode(Enum):
    RANDOM_ATTACK    = "random_attack"
    RANDOM_COLLISION = "random_collision"
    TARGETED_ATTACK  = "targeted_attack"
    MECHANICAL       = "mechanical"


class FormationType(Enum):
    RING    = "ring"
    DIAMOND = "diamond"
    GRID    = "grid"
    V_SHAPE = "v_shape"
    RANDOM  = "random"   # resolved to a concrete type at startup


@dataclass
class FaultConfig:
    """User-facing config. Edit ACTIVE_FAULT below -- nothing else."""

    # Failure type
    mode: FailureMode = FailureMode.RANDOM_ATTACK

    # Formation the swarm holds before the fault fires
    formation: FormationType = FormationType.RANDOM

    # TARGETED_ATTACK only: which drone to destroy (0 .. n_drones-1)
    # Ignored for RANDOM_* modes.
    target_drone_id: int = -1

    # Frame at which failure fires.  None = random within the failure window.
    failure_frame: Optional[int] = None

    # Fixed seed for reproducible runs.  None = different every run.
    seed: Optional[int] = None

    def resolve(self, n_drones: int, fail_start: int, fail_end: int) -> "ResolvedFault":
        """Called once at startup; returns a fully-determined scenario."""

        # True randomness: use os.urandom so rapid successive runs differ
        raw_seed = (self.seed if self.seed is not None
                    else int.from_bytes(os.urandom(8), "big"))
        rng = random.Random(raw_seed)

        # -- Failed drone -------------------------------------------------
        if (self.mode == FailureMode.TARGETED_ATTACK
                and 0 <= self.target_drone_id < n_drones):
            failed_id = self.target_drone_id
        else:
            failed_id = rng.randint(0, n_drones - 1)

        # -- Spoofed drones (for C++ mesh compatibility; not used in sim) --
        candidates = [i for i in range(n_drones) if i != failed_id]
        spoofed = rng.sample(candidates, min(2, len(candidates)))

        # -- Failure frame ------------------------------------------------
        if self.failure_frame is not None:
            frame = max(fail_start + 5, self.failure_frame)
        else:
            mid = (fail_start + fail_end) // 2
            frame = rng.randint(fail_start + 10, mid)

        # -- Formation type -----------------------------------------------
        if self.formation == FormationType.RANDOM:
            resolved_fmt = rng.choice([
                FormationType.RING,
                FormationType.DIAMOND,
                FormationType.GRID,
                FormationType.V_SHAPE,
            ])
        else:
            resolved_fmt = self.formation

        return ResolvedFault(
            failed_drone_id=failed_id,
            failure_frame=frame,
            failure_mode=self.mode,
            spoofed_ids=spoofed,
            resolved_formation=resolved_fmt,
        )


@dataclass
class ResolvedFault:
    """Fully resolved scenario consumed by swarm_sim.py."""
    failed_drone_id:     int
    failure_frame:       int
    failure_mode:        FailureMode
    spoofed_ids:         List[int]
    resolved_formation:  FormationType

    def print_summary(self) -> None:
        w = 50
        sep = "=" * w
        lbl = {
            "Mode":         self.failure_mode.value,
            "Formation":    self.resolved_formation.value,
            "Target Drone": str(self.failed_drone_id),
            "Failure Frame": str(self.failure_frame),
        }
        print(f"+{sep}+")
        print(f"|{'SwarmRaft Fault Injector -- ACTIVE SCENARIO':^{w}}|")
        print(f"+{sep}+")
        for k, v in lbl.items():
            print(f"|  {k:<20}: {v:<{w - 25}}|")
        print(f"+{sep}+")
        print()


# =============================================================================
#  EDIT THIS BLOCK to change the scenario.
#  Everything else is resolved automatically at startup.
# =============================================================================

ACTIVE_FAULT = FaultConfig(
    mode=FailureMode.RANDOM_ATTACK,     # failure mode
    formation=FormationType.RANDOM,     # formation type (RANDOM picks one each run)

    # -- override specific values by uncommenting: --

    # mode=FailureMode.TARGETED_ATTACK,
    # target_drone_id=3,                # destroy drone 3 specifically

    # mode=FailureMode.MECHANICAL,      # gradual jitter then failure
    # mode=FailureMode.RANDOM_COLLISION, # spin-out crash

    # formation=FormationType.RING,
    # formation=FormationType.DIAMOND,
    # formation=FormationType.GRID,
    # formation=FormationType.V_SHAPE,

    # failure_frame=80,                 # fix when the failure fires
    # seed=42,                          # fix seed for reproducible runs
)

# =============================================================================
