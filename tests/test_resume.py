"""Resume-from-checkpoint: a new Trainer restores step/best_metric and computes the start epoch,
so training can continue across Kaggle commits."""
from star.config import Config
from star.engine import Trainer
from star.models import STARModel
from star.utils.checkpoint import save_checkpoint


class _FakeLoader:                       # Trainer.__init__ only needs len(); resume doesn't iterate
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def _cfg(tmp):
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.checkpoint = None
    cfg.model.embed_dim = 64
    cfg.optim.epochs = 10
    cfg.train.grad_accum = 2
    cfg.train.out_dir = str(tmp)
    return cfg


def test_resume_restores_step_best_and_start_epoch(tmp_path):
    cfg = _cfg(tmp_path)
    t1 = Trainer(STARModel(cfg), cfg, _FakeLoader(20), None, "cpu")   # 20//2 = 10 steps/epoch
    assert t1.steps_per_epoch == 10 and t1.start_epoch == 0 and t1.max_seconds is None

    # checkpoint as if 35 optimizer-steps + best 0.5 had been done (= 3.5 epochs in)
    save_checkpoint(str(tmp_path / "last.pth"), t1.model, t1.optimizer, t1.scheduler,
                    step=35, best_metric=0.5, extra={"cfg": {}})

    t2 = Trainer(STARModel(cfg), cfg, _FakeLoader(20), None, "cpu")
    start = t2.resume_from(str(tmp_path / "last.pth"))
    assert t2.step == 35 and abs(t2.best_metric - 0.5) < 1e-9
    assert start == 3 and t2.start_epoch == 3            # 35 // 10 -> resume at epoch 3
