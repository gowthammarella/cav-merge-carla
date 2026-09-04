import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from vision_comm.comm_metrics import (
    communication_cost,
    count_id_switches,
    iou,
    match_detections,
    occlusion_robustness,
)


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 5.0, 5.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # overlap area 5x10=50, union = 100+100-50=150
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_match_detections_perfect_match():
    boxes = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]
    precision, recall = match_detections(boxes, boxes)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_match_detections_no_predictions():
    precision, recall = match_detections([], [(0.0, 0.0, 10.0, 10.0)])
    assert precision == 1.0
    assert recall == 0.0


def test_match_detections_no_ground_truth():
    precision, recall = match_detections([(0.0, 0.0, 10.0, 10.0)], [])
    assert precision == 0.0
    assert recall == 1.0


def test_match_detections_both_empty():
    precision, recall = match_detections([], [])
    assert precision == 1.0 and recall == 1.0


def test_match_detections_partial():
    pred = [(0.0, 0.0, 10.0, 10.0), (500.0, 500.0, 510.0, 510.0)]  # one good, one spurious
    gt = [(0.0, 0.0, 10.0, 10.0)]
    precision, recall = match_detections(pred, gt)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(1.0)


def test_count_id_switches_none_when_stable():
    assert count_id_switches([1, 1, 1, 1]) == 0


def test_count_id_switches_counts_changes():
    assert count_id_switches([1, 1, 2, 2, 3]) == 2


def test_count_id_switches_ignores_none_gaps():
    # target briefly lost (None) then reappears under the SAME id -> no switch
    assert count_id_switches([1, 1, None, None, 1]) == 0


def test_count_id_switches_reappearing_under_new_id_counts():
    assert count_id_switches([1, None, 2]) == 1


def test_occlusion_robustness_none_when_no_occluded_frames():
    result = occlusion_robustness(detected_flags=[True, True], occluded_flags=[False, False])
    assert result is None


def test_occlusion_robustness_computes_recall_during_occlusion():
    detected = [True, False, True, False]
    occluded = [True, True, False, False]
    # only first two frames are occluded; detected in 1 of those 2
    result = occlusion_robustness(detected, occluded)
    assert result == pytest.approx(0.5)


def test_occlusion_robustness_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        occlusion_robustness([True], [True, False])


def test_communication_cost_all_local_only():
    result = communication_cost([0, 0, 0], dt_s=0.1)
    assert result["total_bytes"] == 0
    assert result["mean_bytes_per_message"] == 0.0
    assert result["messages_per_second"] == 0.0


def test_communication_cost_basic():
    result = communication_cost([100, 100, 0, 200], dt_s=0.5)
    assert result["total_bytes"] == 400
    assert result["mean_bytes_per_message"] == pytest.approx(400 / 3)
    # duration = 4 ticks * 0.5s = 2s; 3 nonzero messages sent
    assert result["messages_per_second"] == pytest.approx(1.5)
    assert result["bytes_per_second"] == pytest.approx(200.0)


def test_communication_cost_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        communication_cost([100], dt_s=0.0)
