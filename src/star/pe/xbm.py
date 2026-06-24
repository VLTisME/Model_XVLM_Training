from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CrossBatchMemory(nn.Module):
    def __init__(self, capacity: int, dim: int):
        super().__init__()
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.register_buffer("image", torch.zeros(capacity, dim, dtype=torch.float16))
        self.register_buffer("text", torch.zeros(capacity, dim, dtype=torch.float16))
        self.register_buffer("instance_id", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("caption_hash", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("age", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))
        self.register_buffer("clock", torch.zeros((), dtype=torch.long))

    def reset(self) -> None:
        self.pointer.zero_()
        self.filled.zero_()
        self.clock.zero_()
        self.instance_id.fill_(-1)
        self.caption_hash.fill_(-1)
        self.age.zero_()

    @torch.no_grad()
    def enqueue(
        self,
        image: Tensor,
        text: Tensor,
        instance_id: Tensor,
        caption_hash: Tensor,
    ) -> None:
        image = F.normalize(image.detach(), dim=-1).to(self.image.dtype)
        text = F.normalize(text.detach(), dim=-1).to(self.text.dtype)
        count = image.size(0)
        if count >= self.capacity:
            image = image[-self.capacity :]
            text = text[-self.capacity :]
            instance_id = instance_id[-self.capacity :]
            caption_hash = caption_hash[-self.capacity :]
            count = self.capacity
        positions = (torch.arange(count, device=image.device) + self.pointer) % self.capacity
        self.image[positions] = image
        self.text[positions] = text
        self.instance_id[positions] = instance_id
        self.caption_hash[positions] = caption_hash
        self.age[positions] = self.clock
        self.pointer.copy_((self.pointer + count) % self.capacity)
        self.filled.copy_(torch.minimum(self.filled + count, self.filled.new_tensor(self.capacity)))
        self.clock.add_(1)

    def view(self) -> dict[str, Tensor]:
        count = int(self.filled)
        return {
            "image": self.image[:count].float(),
            "text": self.text[:count].float(),
            "instance_id": self.instance_id[:count],
            "caption_hash": self.caption_hash[:count],
            "age": self.clock - self.age[:count],
        }


def pe_xbm_loss(
    image: Tensor,
    text: Tensor,
    instance_id: Tensor,
    caption_hash: Tensor,
    temperature: Tensor,
    queue: CrossBatchMemory,
) -> tuple[Tensor, dict[str, Tensor]]:
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    memory = queue.view()
    n = image.size(0)
    targets = torch.arange(n, device=image.device)

    all_text = torch.cat([text, memory["text"].to(image.device)], dim=0)
    all_image = torch.cat([image, memory["image"].to(image.device)], dim=0)
    logits_i2t = image @ all_text.t() / temperature
    logits_t2i = text @ all_image.t() / temperature

    diag = torch.eye(n, dtype=torch.bool, device=image.device)
    current_forbidden = (
        (instance_id[:, None] == instance_id[None, :])
        | (caption_hash[:, None] == caption_hash[None, :])
    ) & ~diag
    logits_i2t[:, :n] = logits_i2t[:, :n].masked_fill(current_forbidden, -1e4)
    logits_t2i[:, :n] = logits_t2i[:, :n].masked_fill(current_forbidden, -1e4)

    if memory["text"].numel():
        same_instance = instance_id[:, None] == memory["instance_id"].to(image.device)[None, :]
        same_caption = caption_hash[:, None] == memory["caption_hash"].to(image.device)[None, :]
        forbidden = same_instance | same_caption
        logits_i2t[:, n:] = logits_i2t[:, n:].masked_fill(forbidden, -1e4)
        logits_t2i[:, n:] = logits_t2i[:, n:].masked_fill(forbidden, -1e4)

    loss = 0.5 * (
        F.cross_entropy(logits_i2t, targets) + F.cross_entropy(logits_t2i, targets)
    )
    diagnostics = {
        "positive_similarity": (image * text).sum(dim=1).mean().detach(),
        "queue_size": image.new_tensor(float(int(queue.filled))),
        "queue_fill": image.new_tensor(float(int(queue.filled)) / max(queue.capacity, 1)),
        "queue_age": (
            memory["age"].float().mean().to(image.device)
            if memory["age"].numel()
            else image.new_zeros(())
        ),
        "queue_negative_similarity": (
            (image @ memory["text"].to(image.device).t()).max(dim=1).values.mean()
            if memory["text"].numel()
            else image.new_zeros(())
        ),
    }
    return loss, diagnostics
