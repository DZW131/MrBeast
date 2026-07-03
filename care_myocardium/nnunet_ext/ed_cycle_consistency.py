from __future__ import annotations

import torch
import torch.nn.functional as F


def _primary_output(output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _motion_features_from_cine(frames: torch.Tensor) -> torch.Tensor:
    ref = frames[:, :1]
    diffs = frames - ref
    absdiff = diffs.abs()
    max_idx = absdiff.argmax(dim=1, keepdim=True)
    signed_at_max = diffs.gather(1, max_idx)
    denom = max(frames.shape[1] - 1, 1)
    return torch.cat(
        [
            frames.std(dim=1, keepdim=True, unbiased=False),
            absdiff.max(dim=1, keepdim=True).values,
            absdiff.mean(dim=1, keepdim=True),
            signed_at_max,
            max_idx.to(dtype=frames.dtype) / float(denom),
        ],
        dim=1,
    )


def build_last_ed_reference_view(data: torch.Tensor, num_frames: int = 30) -> torch.Tensor:
    """Reverse the cine sequence and rebuild ED-referenced motion proxy channels.

    Dataset608 stores cine frames first, followed by five deterministic motion
    proxy channels relative to channel 0. Reversing frames makes the final cine
    phase the reference ED view, then recomputes those five proxy channels. Any
    extra channels after the first five motion proxies are kept as zeros because
    they are reference-frame specific and should not leak stale first-ED cues.
    """

    if data.ndim < 3:
        raise ValueError(f"Expected nnU-Net tensor with shape (B, C, ...), got {tuple(data.shape)}")
    if data.shape[1] < num_frames + 5:
        raise ValueError(
            f"Need at least {num_frames + 5} channels for {num_frames} cine frames plus motion proxies, "
            f"got {data.shape[1]}"
        )
    reversed_frames = torch.flip(data[:, :num_frames], dims=(1,))
    rebuilt_motion = _motion_features_from_cine(reversed_frames)
    if data.shape[1] == num_frames + 5:
        return torch.cat([reversed_frames, rebuilt_motion], dim=1)
    extra = torch.zeros_like(data[:, num_frames + 5 :])
    return torch.cat([reversed_frames, rebuilt_motion, extra], dim=1)


def masked_soft_kl_divergence(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    mask: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits, dim=1)
    teacher_log_probs = teacher_probs.clamp_min(1e-8).log()
    per_class = teacher_probs * (teacher_log_probs - log_probs)
    if class_weights is not None:
        per_class = per_class * class_weights.view(1, -1, *([1] * (teacher_probs.ndim - 2)))
    per_voxel = per_class.sum(dim=1, keepdim=True)
    denom = mask.sum().clamp_min(1.0)
    return (per_voxel * mask).sum() / denom


def ed_cycle_consistency_loss(
    first_ed_output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    last_ed_output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    confidence_threshold: float = 0.6,
    scar_weight: float = 2.0,
) -> torch.Tensor:
    """Bidirectional consistency between first-ED and last-ED cine views."""

    first_logits = _primary_output(first_ed_output)
    last_logits = _primary_output(last_ed_output)
    first_teacher = F.softmax(first_logits.detach(), dim=1)
    last_teacher = F.softmax(last_logits.detach(), dim=1)

    first_conf = first_teacher.max(dim=1, keepdim=True).values
    last_conf = last_teacher.max(dim=1, keepdim=True).values
    first_mask = (first_conf >= confidence_threshold).to(dtype=first_logits.dtype)
    last_mask = (last_conf >= confidence_threshold).to(dtype=last_logits.dtype)

    num_classes = first_logits.shape[1]
    class_weights = torch.ones(num_classes, dtype=first_logits.dtype, device=first_logits.device)
    if num_classes > 3:
        class_weights[3] = float(scar_weight)

    first_to_last = masked_soft_kl_divergence(last_logits, first_teacher, first_mask, class_weights)
    last_to_first = masked_soft_kl_divergence(first_logits, last_teacher, last_mask, class_weights)
    return 0.5 * (first_to_last + last_to_first)
