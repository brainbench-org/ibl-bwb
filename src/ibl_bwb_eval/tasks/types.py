"""Enums shared by more than one suite's readouts.

Each suite's own tasks and readout specs live in its module next to this one.
"""

from enum import Enum
from typing import Protocol


class TargetResolution(Enum):
    """How many targets a task has along the input's time axis: one, or one per timestep.

    This is the output's shape and not the kind of value in it (:class:`DataType`). What
    entity a target belongs to (a trial window in TS1, a unit in TS3) is an orthogonal
    axis carried by the task id, so it is deliberately not encoded here.

    Attributes:
        SEQUENCE: one target for the whole window (e.g. the choice made in 1s).
        TIMESTEP: one target per timestep (e.g. wheel speed @50Hz).
    """

    SEQUENCE = 0
    TIMESTEP = 1


class DataType(Enum):
    """What kind of value each target is, independent of :class:`TargetResolution`.

    A task can be classification and per-timestep at once, so size a readout head from
    TargetResolution and pick a loss from this.

    Attributes:
        CONTINUOUS: For continuous-valued variables
        BINARY: For binary variables
        MULTINOMIAL: For multi-class variables
        EVENT_RATE: For variables representing event rates
    """

    CONTINUOUS = 0
    BINARY = 1
    MULTINOMIAL = 2
    EVENT_RATE = 3


class ReadoutSpec(Protocol):
    """What a model needs to size a readout head, and nothing more.

    The intersection of the suites' specs, not a base class, so they stay independent.
    ``dim`` and ``num_timesteps`` are properties so a spec deriving them (TS3) and one
    storing them (TS1) both qualify. ``num_timesteps`` is the length of the axis
    :class:`TargetResolution` names, 1 for a sequence-level task and the target's sample
    count for a timestep-level one, so a model sizes its head without branching on it.
    """

    id: str
    target_resolution: TargetResolution

    @property
    def dim(self) -> int: ...

    @property
    def num_timesteps(self) -> int: ...
