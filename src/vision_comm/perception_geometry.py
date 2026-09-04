"""Pure monocular-camera geometry: turns a tracked 2D bounding box into an
estimated world-frame range/position/velocity. Standard pinhole-camera math
(known-object-size range estimation, bearing-from-pixel-offset, camera-to-
world transform) — deliberately NOT a new detection/tracking algorithm,
per the mentor's "don't invent a new model" scoping guidance. No CARLA
dependency; `perception.py` is the CARLA+YOLO-dependent caller that feeds
this module real tracked boxes and camera parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

AVERAGE_VEHICLE_WIDTH_M = 1.8


@dataclass(frozen=True)
class CameraIntrinsics:
    image_width_px: int
    image_height_px: int
    fov_deg: float

    def __post_init__(self) -> None:
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if not 0 < self.fov_deg < 180:
            raise ValueError(f"fov_deg must be in (0, 180), got {self.fov_deg}")

    @property
    def focal_length_px(self) -> float:
        return (self.image_width_px / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)


@dataclass(frozen=True)
class CameraPose:
    """Camera position and heading in world coordinates (2D, ground plane)."""

    x_m: float
    y_m: float
    yaw_rad: float


def estimate_range_m(
    bbox_width_px: float,
    intrinsics: CameraIntrinsics,
    known_width_m: float = AVERAGE_VEHICLE_WIDTH_M,
) -> float:
    """Range from apparent bounding-box width vs a known real-world width —
    the standard "similar triangles" monocular distance estimate.
    """
    if bbox_width_px <= 0:
        raise ValueError(f"bbox_width_px must be positive, got {bbox_width_px}")
    return (known_width_m * intrinsics.focal_length_px) / bbox_width_px


def estimate_bearing_rad(bbox_center_x_px: float, intrinsics: CameraIntrinsics) -> float:
    """Angle (radians) of the detection off the camera's forward axis,
    positive = to the right of center.
    """
    offset_px = bbox_center_x_px - intrinsics.image_width_px / 2.0
    return math.atan2(offset_px, intrinsics.focal_length_px)


def estimate_world_position_m(
    camera_pose: CameraPose, range_m: float, bearing_rad: float
) -> tuple[float, float]:
    """World-frame (x, y) of a detection given the camera's pose and the
    detection's estimated range + bearing relative to the camera's heading.
    """
    absolute_angle = camera_pose.yaw_rad + bearing_rad
    x = camera_pose.x_m + range_m * math.cos(absolute_angle)
    y = camera_pose.y_m + range_m * math.sin(absolute_angle)
    return x, y


def finite_difference_velocity_mps(
    position_prev_m: tuple[float, float],
    position_curr_m: tuple[float, float],
    dt_s: float,
) -> tuple[float, float]:
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    return (
        (position_curr_m[0] - position_prev_m[0]) / dt_s,
        (position_curr_m[1] - position_prev_m[1]) / dt_s,
    )


def bbox_from_detection(
    center_x_px: float, center_y_px: float, width_px: float, height_px: float
) -> tuple[float, float, float, float]:
    """Convenience: (center, size) -> (x1, y1, x2, y2), the format
    `comm_metrics.iou` expects.
    """
    return (
        center_x_px - width_px / 2.0,
        center_y_px - height_px / 2.0,
        center_x_px + width_px / 2.0,
        center_y_px + height_px / 2.0,
    )
