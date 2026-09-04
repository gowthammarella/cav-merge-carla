"""V2V communication abstraction across the mentor brief's four
information-richness levels (A-D). No CARLA dependency — pure schemas plus
real serialization, so byte-size accounting used for the communication-cost
metrics is exact (measured from an actual encoded payload), not guessed.

Condition D (compact neural feature sharing) is intentionally NOT
implemented yet: the mentor's own project-scoping slide says to start with
B vs C and treat D as an extension "if time permits" — `message_size_for_condition`
raises `NotImplementedError` for it so that gap is loud, not silent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional


class CommCondition(Enum):
    LOCAL_ONLY = "A"       # no communication
    STATE = "B"            # position, speed, acceleration, lane, intent
    OBJECT = "C"           # detected objects: type, position, velocity, confidence
    COMPACT_FEATURE = "D"  # deferred extension — not implemented


@dataclass(frozen=True)
class StateMessage:
    """Condition B payload: one vehicle's own kinematic state."""

    vehicle_id: str
    position_m: tuple[float, float]
    speed_mps: float
    accel_mps2: float
    lane_id: int
    intent: str  # e.g. "yield", "hold", "accelerate", "unknown"

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.to_bytes())


@dataclass(frozen=True)
class DetectedObject:
    """One entry in a Condition C payload: a detected/tracked object."""

    object_type: str  # e.g. "car", "truck"
    position_m: tuple[float, float]
    velocity_mps: tuple[float, float]
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class ObjectMessage:
    """Condition C payload: a sender's full list of detected objects for one tick."""

    sender_id: str
    objects: tuple[DetectedObject, ...]

    def to_bytes(self) -> bytes:
        payload = {
            "sender_id": self.sender_id,
            "objects": [asdict(o) for o in self.objects],
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @property
    def size_bytes(self) -> int:
        return len(self.to_bytes())


def message_size_for_condition(
    condition: CommCondition,
    *,
    state: Optional[StateMessage] = None,
    objects: Optional[ObjectMessage] = None,
) -> int:
    """Bytes actually transmitted on one tick, for one sender, under `condition`."""
    if condition is CommCondition.LOCAL_ONLY:
        return 0
    if condition is CommCondition.STATE:
        if state is None:
            raise ValueError("state message required for CommCondition.STATE")
        return state.size_bytes
    if condition is CommCondition.OBJECT:
        if objects is None:
            raise ValueError("object message required for CommCondition.OBJECT")
        return objects.size_bytes
    if condition is CommCondition.COMPACT_FEATURE:
        raise NotImplementedError(
            "Condition D (compact neural feature sharing) is a deferred extension, "
            "not implemented — see the mentor's own project-scoping guidance."
        )
    raise ValueError(f"unknown condition {condition}")
