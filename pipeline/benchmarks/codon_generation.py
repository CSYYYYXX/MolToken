"""Codon one-output greedy decoding and raw-slot accounting."""
from __future__ import annotations

from copy import deepcopy

TASK = "codontransformer_protein_to_dna"
PROTOCOL = "axis-codon-figure4-input-only-greedy-v1-20260923"


def prompt_only(request):
    return {"id": request["id"], "task": TASK, "target_type": "dna",
            "condition": deepcopy(request["condition"]), "target_bpe_ids": [0]}


def validate_checkpoint(config):
    metadata = config["metadata"]
    if (metadata["stage"] != "full" or metadata["checkpoint_role"] != "best" or
            metadata["lineage_id"] != TASK or metadata["max_context"] != 512 or
            metadata["molecule_condition_mode"] != "direct_qwen"):
        raise ValueError("Codon evaluation requires the matching Full/best, direct_qwen, context512")


def generate_requests(model, collator, requests, device):
    """One greedy output per request; group equal prompts to avoid pad queries.

    Never expose reference proteins/DNA, target IDs or reporting metadata to
    the model. The protein condition was encoded in the input-only preparer.
    """
    import torch
    from transformers import GenerationConfig
    from cli.evaluate.sample_axis_b_sft import model_batch_from_records, _generation_groups
    from cli.evaluate.sample_tcr_translate_released_top20_auxiliary import convert_model_sequence
    from pipeline.finetune.axis_b_data import EOS_TOKEN_ID, PAD_TOKEN_ID

    with torch.inference_mode():
        build, collated = model_batch_from_records([prompt_only(r) for r in requests], collator, device)
        if bool(collated["molecule_truncated"].any()):
            raise ValueError("sampling collator truncated a prepared protein condition")
        inputs = model.build_qwen_inputs(**build)
        lengths = inputs["attention_mask"].sum(dim=1).tolist()
        for request, length in zip(requests, lengths, strict=True):
            if (length != request["meta"]["prompt_tokens"] or
                    length + request["max_new_tokens"] != 512):
                raise ValueError(f"prepared/actual prompt budget differs for {request['id']}")
        outputs = [None] * len(requests)
        for length, indices in _generation_groups(inputs["attention_mask"]):
            selector = torch.tensor(indices, dtype=torch.long, device=device)
            embeds = inputs["inputs_embeds"].index_select(0, selector)[:, :length]
            mask = inputs["attention_mask"].index_select(0, selector)[:, :length]
            config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                                      max_new_tokens=512-length, eos_token_id=EOS_TOKEN_ID,
                                      pad_token_id=PAD_TOKEN_ID, use_cache=True)
            generated = model.qwen.generate(inputs_embeds=embeds, attention_mask=mask,
                                            generation_config=config, use_model_defaults=False,
                                            use_cache=True)
            if generated.shape[0] != len(indices):
                raise ValueError("greedy generation did not return one result per input")
            for index, model_ids in zip(indices, generated.tolist(), strict=True):
                raw, eos, invalid = convert_model_sequence(model_ids)
                outputs[index] = {"id": requests[index]["id"], "task": TASK, "protocol": PROTOCOL,
                                  "model_token_ids": model_ids, "generated_bpe_ids": raw,
                                  "ended_with_eos": eos, "invalid_model_id": invalid,
                                  "prompt_tokens": length, "max_new_tokens": 512-length}
    if any(row is None for row in outputs):
        raise ValueError("sampling omitted requested slots")
    return outputs
