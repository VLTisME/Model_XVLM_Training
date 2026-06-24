import torch
import torch.nn.functional as F

from star.pe import CrossBatchMemory, pe_xbm_loss


def test_xbm_enqueue_wrap_and_loss():
    torch.manual_seed(0)
    queue = CrossBatchMemory(capacity=4, dim=8)
    image = F.normalize(torch.randn(4, 8), dim=-1)
    text = image.clone()
    ids = torch.arange(4)
    hashes = torch.arange(10, 14)

    loss_empty, diagnostics = pe_xbm_loss(
        image, text, ids, hashes, torch.tensor(0.07), queue
    )
    assert torch.isfinite(loss_empty)
    assert diagnostics["queue_size"].item() == 0

    queue.enqueue(image, text, ids, hashes)
    assert int(queue.filled) == 4
    loss_full, diagnostics = pe_xbm_loss(
        image, text, ids, hashes, torch.tensor(0.07), queue
    )
    assert torch.isfinite(loss_full)
    assert diagnostics["queue_fill"].item() == 1.0

    queue.enqueue(image[:2], text[:2], ids[:2], hashes[:2])
    assert int(queue.filled) == 4
