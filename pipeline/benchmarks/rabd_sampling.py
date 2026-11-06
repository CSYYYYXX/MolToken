"""Likelihood recording and prompt isolation for RAbD graph-token sampling."""
from copy import deepcopy

import torch
from transformers import LogitsProcessor


def prompt_only(record):
    # Never expose target IDs or reference/meta fields to the collator/model.
    return {"id": record["id"], "task": record["task"],
            "target_type": record["target_type"],
            "condition": deepcopy(record["condition"]), "target_bpe_ids": [0]}


class SequenceNLLRecorder(LogitsProcessor):
    """Record selected-token NLL without retaining vocabulary logits per step.

    Used only with unwarped ancestral sampling (T=1, no top-k/p). This processor
    never changes logits. The next invocation reveals the preceding draw;
    finish() records the last draw. Include EOS, ignore post-EOS padding.
    """
    def __init__(self, eos_token_id):
        self.eos = eos_token_id
        self.previous = None
        self.nll = self.active = self.count = None

    def _consume(self, token):
        selected = self.previous.gather(1, token[:, None]).squeeze(1)
        self.nll += torch.where(self.active, -selected, 0.)
        self.count += self.active.long()
        self.active &= token.ne(self.eos)
        self.previous = None

    def __call__(self, input_ids, scores):
        if self.previous is None:
            self.nll = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.float32)
            self.count = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.long)
            self.active = torch.ones(scores.shape[0], device=scores.device, dtype=torch.bool)
        else:
            self._consume(input_ids[:, -1])
        self.previous = scores.float().log_softmax(-1)
        return scores

    def finish(self, generated_ids):
        if self.previous is None or generated_ids.shape[1] == 0:
            raise ValueError("no generated token to finish likelihood recording")
        self._consume(generated_ids[:, -1])
        if not torch.isfinite(self.nll).all():
            raise ValueError("nonfinite generated-sequence likelihood")
        return self.nll.tolist(), self.count.tolist()


@torch.inference_mode()
def scored_generate(qwen, prompt_embeds, batch_size, *, max_context=512):
    """One fixed condition per batch; unconstrained ancestral EOS generation."""
    from transformers import GenerationConfig, LogitsProcessorList
    from pipeline.finetune.axis_b_data import EOS_TOKEN_ID, PAD_TOKEN_ID

    length = prompt_embeds.shape[1]
    if not 0 < length < max_context or prompt_embeds.shape[0] != 1:
        raise ValueError("expected one nonempty prompt within context")
    embeds = prompt_embeds.expand(batch_size, -1, -1)
    mask = torch.ones((batch_size, length), dtype=torch.long, device=embeds.device)
    recorder = SequenceNLLRecorder(EOS_TOKEN_ID)
    # Construct an explicit config: no inherited repetition/length penalties,
    # token suppression, top-k truncation, or minimum-length intervention.
    config = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
                              max_new_tokens=max_context - length, num_beams=1,
                              eos_token_id=EOS_TOKEN_ID, pad_token_id=PAD_TOKEN_ID,
                              use_cache=True)
    generated = qwen.generate(inputs_embeds=embeds, attention_mask=mask,
                              generation_config=config, use_model_defaults=False,
                              use_cache=True,
                              logits_processor=LogitsProcessorList([recorder]))
    nll, counts = recorder.finish(generated)
    return generated.tolist(), nll, counts
