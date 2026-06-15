"""vitpose_extract: the HF-result -> arrays adapter (hf_to_arrays) + that it reuses the YOLO
formatters so the json schema is the SAME contract. Pure logic, no transformers/torch needed."""
import importlib.util
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vp = _load("vitpose_extract")
ep = _load("extract_pose_yolo")


def test_reuses_yolo_formatters():
    # same schema contract => formatters come FROM extract_pose_yolo, not re-implemented in vitpose
    assert vp.to_item.__module__ == "extract_pose_yolo"
    assert vp.empty_item.__module__ == "extract_pose_yolo"
    assert vp.pick_primary.__module__ == "extract_pose_yolo"


def test_hf_to_arrays_keypoints2d_plus_scores():
    # transformers schema A: keypoints [17,2] + separate scores [17]
    person = {"keypoints": [[float(i), float(i) + 1] for i in range(17)], "scores": [0.5] * 17}
    boxes, kdata = vp.hf_to_arrays([person])
    assert boxes == [[0.0, 1.0, 16.0, 17.0]]              # keypoint extent
    assert len(kdata[0]) == 17 and kdata[0][0] == [0.0, 1.0, 0.5]
    # flows into pick_primary -> to_item unchanged
    kpts, box = ep.pick_primary(boxes, kdata)
    it = ep.to_item(kpts, box, width=200, height=400)
    assert len(it["instances"][0]["keypoints_xyc"]) == 17 and it["status"] == "ok"


def test_hf_to_arrays_keypoints3d_conf_baked_in():
    # transformers schema B: keypoints [17,3] (conf in the 3rd column, no separate scores)
    person = {"keypoints": [[2.0, 3.0, 0.8]] * 17}
    boxes, kdata = vp.hf_to_arrays([person])
    assert boxes == [[2.0, 3.0, 2.0, 3.0]]               # degenerate extent (all same point)
    assert kdata[0][0] == [2.0, 3.0, 0.8]


def test_hf_to_arrays_handles_real_hf_torch_tensors():
    # mimic HF VitPose post_process_pose_estimation per-person output: torch tensors, not lists
    import torch
    person = {"keypoints": torch.tensor([[float(i), float(i) + 1] for i in range(17)]),
              "scores": torch.tensor([0.7] * 17),
              "bbox": torch.tensor([1.0, 2.0, 3.0, 4.0]), "labels": torch.tensor(0)}
    boxes, kdata = vp.hf_to_arrays([person])
    assert len(kdata) == 1 and len(kdata[0]) == 17
    assert kdata[0][0][:2] == [0.0, 1.0] and abs(kdata[0][0][2] - 0.7) < 1e-6   # x, y, conf -> python float
    # flows through the formatters unchanged -> valid 51-flat item
    kpts, box = ep.pick_primary(boxes, kdata)
    it = ep.to_item(kpts, box, 64, 64)
    assert it["status"] == "ok" and len(it["instances"][0]["keypoints_xyc"]) == 17
    assert all(len(t) == 3 for t in it["instances"][0]["keypoints_xyc"])


def test_hf_to_arrays_picks_largest_person():
    small = {"keypoints": [[float(i % 5), float(i % 5), 0.5] for i in range(17)]}   # extent 0..4
    big = {"keypoints": [[float(i * 6), float(i * 6), 0.9] for i in range(17)]}     # extent 0..96
    boxes, kdata = vp.hf_to_arrays([small, big])
    kpts, box = ep.pick_primary(boxes, kdata)
    assert box == [0.0, 0.0, 96.0, 96.0] and kpts[0] == [0.0, 0.0, 0.9]
