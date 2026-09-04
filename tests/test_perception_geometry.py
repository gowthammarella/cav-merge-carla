import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vision_comm.perception_geometry import (
    CameraIntrinsics,
    CameraPose,
    bbox_from_detection,
    estimate_bearing_rad,
    estimate_range_m,
    estimate_world_position_m,
    finite_difference_velocity_mps,
)


def default_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(image_width_px=1280, image_height_px=720, fov_deg=90.0)


def test_focal_length_positive():
    assert default_intrinsics().focal_length_px > 0


def test_invalid_fov_rejected():
    with pytest.raises(ValueError):
        CameraIntrinsics(image_width_px=1280, image_height_px=720, fov_deg=200.0)


def test_invalid_image_dims_rejected():
    with pytest.raises(ValueError):
        CameraIntrinsics(image_width_px=0, image_height_px=720, fov_deg=90.0)


def test_closer_object_has_wider_bbox_for_same_range_estimate():
    intr = default_intrinsics()
    far_range = estimate_range_m(bbox_width_px=50.0, intrinsics=intr)
    near_range = estimate_range_m(bbox_width_px=200.0, intrinsics=intr)
    assert near_range < far_range


def test_range_estimate_zero_width_rejected():
    with pytest.raises(ValueError):
        estimate_range_m(bbox_width_px=0.0, intrinsics=default_intrinsics())


def test_bearing_zero_at_image_center():
    intr = default_intrinsics()
    bearing = estimate_bearing_rad(bbox_center_x_px=intr.image_width_px / 2.0, intrinsics=intr)
    assert math.isclose(bearing, 0.0, abs_tol=1e-9)


def test_bearing_positive_to_the_right():
    intr = default_intrinsics()
    bearing = estimate_bearing_rad(bbox_center_x_px=intr.image_width_px * 0.9, intrinsics=intr)
    assert bearing > 0


def test_bearing_negative_to_the_left():
    intr = default_intrinsics()
    bearing = estimate_bearing_rad(bbox_center_x_px=intr.image_width_px * 0.1, intrinsics=intr)
    assert bearing < 0


def test_world_position_straight_ahead():
    camera = CameraPose(x_m=0.0, y_m=0.0, yaw_rad=0.0)
    x, y = estimate_world_position_m(camera, range_m=10.0, bearing_rad=0.0)
    assert math.isclose(x, 10.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)


def test_world_position_accounts_for_camera_yaw():
    camera = CameraPose(x_m=0.0, y_m=0.0, yaw_rad=math.pi / 2)  # facing +y
    x, y = estimate_world_position_m(camera, range_m=10.0, bearing_rad=0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 10.0, abs_tol=1e-6)


def test_finite_difference_velocity_basic():
    vx, vy = finite_difference_velocity_mps((0.0, 0.0), (5.0, 0.0), dt_s=0.5)
    assert math.isclose(vx, 10.0)
    assert math.isclose(vy, 0.0)


def test_finite_difference_velocity_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        finite_difference_velocity_mps((0.0, 0.0), (1.0, 0.0), dt_s=0.0)


def test_bbox_from_detection_shape():
    x1, y1, x2, y2 = bbox_from_detection(center_x_px=100, center_y_px=50, width_px=20, height_px=10)
    assert x1 == 90 and x2 == 110
    assert y1 == 45 and y2 == 55
