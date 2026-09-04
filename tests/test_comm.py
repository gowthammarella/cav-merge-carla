import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vision_comm.comm import (
    CommCondition,
    DetectedObject,
    ObjectMessage,
    StateMessage,
    message_size_for_condition,
)


def make_state() -> StateMessage:
    return StateMessage(
        vehicle_id="ramp_0",
        position_m=(31.2, 0.0),
        speed_mps=18.5,
        accel_mps2=-1.2,
        lane_id=2,
        intent="yield",
    )


def make_objects(n: int) -> ObjectMessage:
    objs = tuple(
        DetectedObject(
            object_type="car",
            position_m=(10.0 + i, 0.0),
            velocity_mps=(15.0, 0.0),
            confidence=0.9,
        )
        for i in range(n)
    )
    return ObjectMessage(sender_id="main_0", objects=objs)


def test_local_only_has_zero_size():
    assert message_size_for_condition(CommCondition.LOCAL_ONLY) == 0


def test_state_message_has_positive_size():
    size = message_size_for_condition(CommCondition.STATE, state=make_state())
    assert size > 0


def test_state_requires_state_arg():
    with pytest.raises(ValueError):
        message_size_for_condition(CommCondition.STATE)


def test_object_message_requires_objects_arg():
    with pytest.raises(ValueError):
        message_size_for_condition(CommCondition.OBJECT)


def test_object_message_size_grows_with_object_count():
    small = message_size_for_condition(CommCondition.OBJECT, objects=make_objects(1))
    large = message_size_for_condition(CommCondition.OBJECT, objects=make_objects(5))
    assert large > small


def test_compact_feature_condition_not_implemented():
    with pytest.raises(NotImplementedError):
        message_size_for_condition(CommCondition.COMPACT_FEATURE)


def test_detected_object_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        DetectedObject(object_type="car", position_m=(0, 0), velocity_mps=(0, 0), confidence=1.5)


def test_message_size_is_deterministic():
    state = make_state()
    a = message_size_for_condition(CommCondition.STATE, state=state)
    b = message_size_for_condition(CommCondition.STATE, state=state)
    assert a == b


def test_state_and_object_messages_are_distinct_payloads():
    state_size = message_size_for_condition(CommCondition.STATE, state=make_state())
    object_size = message_size_for_condition(CommCondition.OBJECT, objects=make_objects(1))
    # not asserting a specific ordering (payload shapes differ), just that
    # they're independently computed, non-trivial values
    assert state_size > 0 and object_size > 0
