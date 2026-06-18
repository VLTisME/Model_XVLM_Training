"""`--set` override parsing — especially booleans (regression: `flag=false` used to become the
TRUTHY string "false", silently leaving the flag ON)."""
from star.config import load_config, parse_overrides


def test_parse_overrides_booleans_are_real_bools():
    ov = parse_overrides(["data.lhp_enabled=false", "model.pose_enabled=true",
                          "model.lora_enabled=False", "model.lora_freeze_text=TRUE"])
    assert ov["data"]["lhp_enabled"] is False
    assert ov["model"]["pose_enabled"] is True
    assert ov["model"]["lora_enabled"] is False     # case-insensitive
    assert ov["model"]["lora_freeze_text"] is True


def test_parse_overrides_none_int_float_str():
    ov = parse_overrides(["model.checkpoint=null", "optim.epochs=6",
                          "loss.lambda_smooth_ap=0.2", "data.group_by=pair"])
    assert ov["model"]["checkpoint"] is None
    assert ov["optim"]["epochs"] == 6 and isinstance(ov["optim"]["epochs"], int)
    assert abs(ov["loss"]["lambda_smooth_ap"] - 0.2) < 1e-9
    assert ov["data"]["group_by"] == "pair"          # bare word stays a string


def test_override_false_actually_disables_flag_end_to_end():
    cfg = load_config("configs/star_v3_10k_kaggle.yaml",
                      parse_overrides(["data.lhp_enabled=false", "model.lora_enabled=false"]))
    assert cfg.data.lhp_enabled is False and not cfg.data.lhp_enabled
    assert cfg.model.lora_enabled is False and not cfg.model.lora_enabled


def test_pair_mixed_first_rerun_overrides():
    cfg = load_config(
        "configs/star_v3_10k_kaggle.yaml",
        parse_overrides([
            "data.group_by=pair_mixed",
            "data.pair_hard_pairs=4",
            "data.lhp_enabled=false",
            "model.pose_enabled=true",
            "loss.lambda_itm=2.0",
            "loss.lambda_smooth_ap=0.2",
            "model.lora_freeze_text=true",
        ]),
    )
    assert cfg.data.group_by == "pair_mixed"
    assert cfg.data.pair_hard_pairs == 4
    assert cfg.data.lhp_enabled is False
    assert cfg.model.pose_enabled is True
    assert cfg.loss.lambda_itm == 2.0
    assert cfg.loss.lambda_smooth_ap == 0.2
    assert cfg.model.lora_freeze_text is True
