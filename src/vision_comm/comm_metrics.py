"""Perception-quality and communication-cost metrics — the two metric
categories the mentor's brief adds on top of the existing driving-safety /
merging-performance metrics in `merge_sim.metrics`. No CARLA dependency;
all functions operate on plain boxes/sequences/message sizes so they're
unit-testable with synthetic fixtures.
"""
from __future__ import annotations

from typing import Optional, Sequence

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


def iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections(
    pred_boxes: Sequence[Box], gt_boxes: Sequence[Box], iou_threshold: float = 0.5
) -> tuple[float, float]:
    """Greedy IoU matching -> (precision, recall). Ties broken by
    highest-IoU-first; each prediction/ground-truth box matches at most once.
    """
    if not gt_boxes and not pred_boxes:
        return 1.0, 1.0
    if not pred_boxes:
        return 1.0, 0.0
    if not gt_boxes:
        return 0.0, 1.0

    pairs = []
    for pi, pbox in enumerate(pred_boxes):
        for gi, gbox in enumerate(gt_boxes):
            score = iou(pbox, gbox)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    matched_pred, matched_gt = set(), set()
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    true_positives = len(matched_pred)
    precision = true_positives / len(pred_boxes)
    recall = true_positives / len(gt_boxes)
    return precision, recall


def count_id_switches(track_id_sequence: Sequence[Optional[int]]) -> int:
    """Standard MOT-style ID-switch count: how many times the predicted
    track ID assigned to a (single, continuously-visible) ground-truth
    target changes from one non-null value to a different non-null value.
    A transition through `None` (target briefly lost) does not itself
    count — only reappearing under a DIFFERENT id counts as a switch.
    """
    switches = 0
    last_seen_id: Optional[int] = None
    for track_id in track_id_sequence:
        if track_id is None:
            continue
        if last_seen_id is not None and track_id != last_seen_id:
            switches += 1
        last_seen_id = track_id
    return switches


def occlusion_robustness(
    detected_flags: Sequence[bool], occluded_flags: Sequence[bool]
) -> Optional[float]:
    """Recall specifically during frames flagged as occluded — how well
    detection holds up when the target is partly hidden. Returns None if
    there were no occluded frames to evaluate (undefined, not zero).
    """
    if len(detected_flags) != len(occluded_flags):
        raise ValueError("detected_flags and occluded_flags must be the same length")
    occluded_count = sum(occluded_flags)
    if occluded_count == 0:
        return None
    detected_while_occluded = sum(
        1 for detected, occluded in zip(detected_flags, occluded_flags) if occluded and detected
    )
    return detected_while_occluded / occluded_count


def communication_cost(message_sizes_bytes: Sequence[int], dt_s: float) -> dict:
    """Aggregate communication-cost metrics from a sequence of per-tick
    message sizes (0 for ticks under CommCondition.LOCAL_ONLY where nothing
    was sent).
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    sent = [size for size in message_sizes_bytes if size > 0]
    total_bytes = sum(message_sizes_bytes)
    n_ticks = len(message_sizes_bytes)
    duration_s = n_ticks * dt_s
    return {
        "total_bytes": total_bytes,
        "mean_bytes_per_message": (sum(sent) / len(sent)) if sent else 0.0,
        "messages_per_second": (len(sent) / duration_s) if duration_s > 0 else 0.0,
        "bytes_per_second": (total_bytes / duration_s) if duration_s > 0 else 0.0,
    }
