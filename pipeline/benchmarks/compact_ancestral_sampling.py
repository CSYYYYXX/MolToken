"""Unwarped causal sampling that removes completed rows from the KV cache.

One unpadded prompt is shared by all requests. No length/validity filtering,
retry, forced EOS, vocabulary masking, or sampling-temperature change.
Batch compaction changes RNG consumption, not the ancestral distribution.
"""
import torch


def _draw_token(logits, slots, step):
    # slots/step let tests replay model-produced paths without gold references.
    return torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(1)


@torch.inference_mode()
def compact_generate(model, prompt_ids, batch_size, *, eos_token_id,
                     max_new_tokens=512, compact_every=16, shared_prefill=False):
    if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1
            or batch_size < 1 or max_new_tokens < 1 or compact_every < 1):
        raise ValueError("expected one unpadded prompt and positive sampling settings")
    from transformers import DynamicCache

    device = prompt_ids.device
    slots = torch.arange(batch_size, device=device)
    output = torch.full((batch_size, max_new_tokens), eos_token_id,
                        dtype=torch.long, device=device)
    nll = torch.zeros(batch_size, dtype=torch.float32, device=device)
    counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    cache = DynamicCache(config=model.config)
    prefill_ids = prompt_ids if shared_prefill else prompt_ids.repeat(batch_size, 1)
    result = model(input_ids=prefill_ids, past_key_values=cache,
                   use_cache=True, logits_to_keep=1)
    logits = result.logits[:, -1, :].float()
    cache = result.past_key_values
    if shared_prefill:
        logits = logits.expand(batch_size, -1)
        cache.batch_repeat_interleave(batch_size)
    del result, prefill_ids
    row_steps = 0
    active_steps = 0
    compactions = 0
    for step in range(max_new_tokens):
        token = _draw_token(logits, slots, step)
        selected_nll = logits.logsumexp(-1) - logits.gather(1, token[:, None]).squeeze(1)
        token = torch.where(active, token, eos_token_id)
        output[slots, step] = token
        nll[slots] += torch.where(active, selected_nll, 0.)
        counts[slots] += active.long()
        row_steps += slots.numel()
        active_steps += active.sum()
        active &= token.ne(eos_token_id)
        del logits, selected_nll
        if not active.any() or step + 1 == max_new_tokens:
            break
        if (step + 1) % compact_every == 0 and not active.all():
            keep = active.nonzero(as_tuple=True)[0]
            cache.batch_select_indices(keep)
            slots, token, active = slots[keep], token[keep], active[keep]
            compactions += 1
        result = model(input_ids=token[:, None], past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        logits = result.logits[:, -1, :].float()
        cache = result.past_key_values
        del result
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite generated-path likelihood")
    sizes = counts.tolist()
    rows = [row[:size] for row, size in zip(output.tolist(), sizes, strict=True)]
    return rows, nll.tolist(), sizes, {
        "decode_steps": step + 1, "row_steps": row_steps,
        "active_token_steps": int(active_steps), "compactions": compactions,
        "prefill_rows": 1 if shared_prefill else batch_size, "compact_every": compact_every,
    }
