"""Loss correctness + key invariants."""
import torch

from star.losses import ITCLoss, ITMLoss, SmoothAPLoss, build_itm_pairs
from star.losses.smooth_ap import smooth_ap_from_sim


def test_itc_loss_nonnegative_and_lower_when_aligned():
    torch.manual_seed(0)
    d = 32
    feats = torch.randn(8, d)
    itc = ITCLoss()
    aligned = itc(feats, feats.clone())           # perfect diagonal alignment
    misaligned = itc(feats, torch.randn(8, d))
    assert aligned >= 0 and misaligned >= 0
    assert aligned < misaligned                   # aligned pairs => smaller loss


def test_itc_temp_is_clamped():
    itc = ITCLoss(temp_init=0.07)
    with torch.no_grad():
        itc.temp.fill_(5.0)                # out of range
    itc(torch.randn(3, 8), torch.randn(3, 8))
    assert 0.001 <= itc.temperature <= 0.5  # forward must clamp into the X-VLM range


def test_itc_reduces_to_infonce_for_unique_ids():
    # Faithfulness: with all-unique ids the soft-target loss must equal symmetric InfoNCE.
    import torch.nn.functional as F
    torch.manual_seed(0)
    img = F.normalize(torch.randn(5, 16), dim=-1)
    txt = F.normalize(torch.randn(5, 16), dim=-1)
    itc = ITCLoss()
    with torch.no_grad():
        itc.temp.fill_(0.05)
    loss = itc(img, txt)                                  # ids default -> arange (unique)
    sim = img @ txt.t() / 0.05
    tgt = torch.arange(5)
    ref = 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt))
    assert torch.allclose(loss, ref, atol=1e-5)


def test_itc_ids_change_the_loss():
    # Sanity that identity targets actually take effect (unique vs all-same differ).
    import torch.nn.functional as F
    torch.manual_seed(1)
    img = F.normalize(torch.randn(4, 16), dim=-1)
    txt = F.normalize(torch.randn(4, 16), dim=-1)
    itc = ITCLoss()
    l_unique = itc(img, txt, ids=torch.tensor([0, 1, 2, 3]))
    l_same = itc(img, txt, ids=torch.tensor([0, 0, 0, 0]))
    assert not torch.allclose(l_unique, l_same)


def test_smooth_ap_zero_when_perfectly_ranked():
    # single positive per query placed clearly on top -> AP ~ 1 -> loss ~ 0
    sim = torch.tensor([[5.0, 0.0, -5.0], [5.0, 0.0, -5.0]])
    rel = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    loss = smooth_ap_from_sim(sim, rel, tau=0.01)
    assert loss.item() < 1e-2


def test_smooth_ap_higher_when_positive_at_bottom():
    good = smooth_ap_from_sim(torch.tensor([[5.0, 0.0, -5.0]]),
                              torch.tensor([[1.0, 0.0, 0.0]]), tau=0.01)
    bad = smooth_ap_from_sim(torch.tensor([[-5.0, 0.0, 5.0]]),
                             torch.tensor([[1.0, 0.0, 0.0]]), tau=0.01)
    assert bad > good


def test_smooth_ap_module():
    feats = torch.randn(6, 16)
    loss = SmoothAPLoss()(feats, feats.clone())
    assert torch.isfinite(loss) and loss.item() < 0.2


def test_build_itm_pairs_shapes_and_labels():
    n = 5
    sim = torch.randn(n, n)
    pairs = build_itm_pairs(sim)
    assert pairs["img_idx"].numel() == 3 * n
    # first n are positives
    assert pairs["label"][:n].tolist() == [1] * n
    assert pairs["label"][n:].tolist() == [0] * (2 * n)
    # positive pairs are diagonal
    assert torch.equal(pairs["img_idx"][:n], pairs["txt_idx"][:n])


def test_itm_loss_runs():
    logits = torch.randn(12, 2)
    labels = torch.randint(0, 2, (12,))
    assert torch.isfinite(ITMLoss()(logits, labels))
