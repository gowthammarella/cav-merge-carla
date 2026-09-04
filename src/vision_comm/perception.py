"""Wraps a pretrained YOLO model's built-in ByteTrack tracker over one
CAV's RGB camera feed, and turns tracked detections into world-frame
"local traffic state" using perception_geometry.py's pure math.

Uses `ultralytics.YOLO(...).track(tracker="bytetrack.yaml")` — this gives
YOLO detection AND ByteTrack tracking from one pretrained model call, so
there is no separate tracker library to wire up, per the mentor's "use
pretrained vision models" scoping. Requires `carla` and `ultralytics`;
first executable on Colab.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import carla
import numpy as np
from ultralytics import YOLO

from vision_comm.perception_geometry import (
    CameraIntrinsics,
    CameraPose,
    estimate_bearing_rad,
    estimate_range_m,
    estimate_world_position_m,
    finite_difference_velocity_mps,
)

VEHICLE_CLASS_NAMES = {"car", "truck", "bus"}  # COCO classes YOLO reports for vehicles
CAMERA_MOUNT_FORWARD_M = 1.5
CAMERA_MOUNT_HEIGHT_M = 2.0


@dataclass(frozen=True)
class TrackedObject:
    track_id: int
    object_type: str
    confidence: float
    bbox_px: tuple[float, float, float, float]
    position_m: tuple[float, float]
    velocity_mps: tuple[float, float]


class CameraPerception:
    """One CAV's local perception module: owns an attached RGB camera,
    runs YOLO+ByteTrack per frame, and keeps per-track position history
    for finite-difference velocity estimation.
    """

    def __init__(
        self,
        world: "carla.World",
        vehicle: "carla.Vehicle",
        model_name: str = "yolov8n.pt",
        image_width: int = 1280,
        image_height: int = 720,
        fov_deg: float = 90.0,
    ):
        self.world = world
        self.vehicle = vehicle
        self.model = YOLO(model_name)
        self.intrinsics = CameraIntrinsics(image_width, image_height, fov_deg)
        self._track_history: dict[int, tuple[float, float]] = {}
        self._latest_frame: Optional[np.ndarray] = None
        self.camera = self._spawn_camera(image_width, image_height, fov_deg)

    def _spawn_camera(self, image_width: int, image_height: int, fov_deg: float):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(image_width))
        bp.set_attribute("image_size_y", str(image_height))
        bp.set_attribute("fov", str(fov_deg))
        transform = carla.Transform(
            carla.Location(x=CAMERA_MOUNT_FORWARD_M, z=CAMERA_MOUNT_HEIGHT_M)
        )
        camera = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        camera.listen(self._on_image)
        return camera

    def _on_image(self, image: "carla.Image") -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self._latest_frame = np.ascontiguousarray(array[:, :, :3][:, :, ::-1])  # BGRA -> RGB

    def _camera_pose(self) -> CameraPose:
        transform = self.camera.get_transform()
        yaw_rad = math.radians(transform.rotation.yaw)
        return CameraPose(x_m=transform.location.x, y_m=transform.location.y, yaw_rad=yaw_rad)

    def perceive(self, dt_s: float) -> list[TrackedObject]:
        """Runs one detection+tracking pass on the most recent camera
        frame. Returns an empty list if no frame has arrived yet (e.g.
        the very first tick before the camera callback fires).
        """
        if self._latest_frame is None:
            return []

        results = self.model.track(
            self._latest_frame, tracker="bytetrack.yaml", persist=True, verbose=False
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        camera_pose = self._camera_pose()
        boxes = results[0].boxes
        tracked: list[TrackedObject] = []

        for box_xyxy, track_id, cls, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy(),
        ):
            class_name = self.model.names[int(cls)]
            if class_name not in VEHICLE_CLASS_NAMES:
                continue

            x1, y1, x2, y2 = (float(v) for v in box_xyxy)
            width_px = x2 - x1
            if width_px <= 0:
                continue
            center_x_px = (x1 + x2) / 2.0

            range_m = estimate_range_m(width_px, self.intrinsics)
            bearing_rad = estimate_bearing_rad(center_x_px, self.intrinsics)
            position_m = estimate_world_position_m(camera_pose, range_m, bearing_rad)

            track_id_int = int(track_id)
            prev_position = self._track_history.get(track_id_int)
            velocity_mps = (
                finite_difference_velocity_mps(prev_position, position_m, dt_s)
                if prev_position is not None
                else (0.0, 0.0)
            )
            self._track_history[track_id_int] = position_m

            tracked.append(
                TrackedObject(
                    track_id=track_id_int,
                    object_type=class_name,
                    confidence=float(conf),
                    bbox_px=(x1, y1, x2, y2),
                    position_m=position_m,
                    velocity_mps=velocity_mps,
                )
            )

        return tracked

    def close(self) -> None:
        if self.camera is not None and self.camera.is_alive:
            self.camera.destroy()
