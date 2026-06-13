"""extract_pose_yolo: the producer/consumer contract with the eval notebooks' kpts_of() — tested
without YOLO/torch-hub (only the pure formatter + primary-instance picker)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "extract_pose_yolo", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "extract_pose_yolo.py")
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def _consume_like_kpts_of(item):
    """Replicate the eval notebooks' kpts_of(): -> flat [51] normalized, or None when not 'ok'."""
    if item.get("status") != "ok" or not item.get("instances"):
        return None
    W, H = item["width"], item["height"]
    flat = []
    for x, y, c in item["instances"][0]["keypoints_xyc"]:
        flat += [x / W, y / H, c]
    return flat if len(flat) == 51 else None


def test_to_item_schema_and_roundtrip():
    kpts = [[float(i), float(i) + 1, 0.9] for i in range(17)]
    it = ep.to_item(kpts, [10, 20, 110, 220], width=200, height=400)
    assert it["status"] == "ok" and it["width"] == 200 and it["height"] == 400
    assert len(it["instances"][0]["keypoints_xyc"]) == 17
    flat = _consume_like_kpts_of(it)
    assert flat is not None and len(flat) == 51
    assert all(0.0 <= flat[j] <= 1.0 for j in range(0, 51, 3))      # x normalized
    assert all(0.0 <= flat[j] <= 1.0 for j in range(1, 51, 3))      # y normalized
    assert it["primary_bbox_norm_xyxy"] == [10 / 200, 20 / 400, 110 / 200, 220 / 400]


def test_empty_item_skipped_by_consumer():
    assert _consume_like_kpts_of(ep.empty_item(300, 300)) is None


def test_pick_primary_chooses_largest_box():
    boxes = [[0, 0, 10, 10], [0, 0, 100, 100]]                      # 2nd is far larger
    kdata = [[[1, 1, 0.5]] * 17, [[2, 2, 0.8]] * 17]
    kpts, box = ep.pick_primary(boxes, kdata)
    assert box == [0, 0, 100, 100] and kpts[0] == [2.0, 2.0, 0.8]


def test_pick_primary_none_when_empty():
    assert ep.pick_primary([], []) is None
