"""Axis-B conditional generation model built on the FDDFS flagship Qwen3.

Frozen condition encoders retain their sequence outputs.  Independent
projectors map them into Qwen space, calibrate per-token norms to the native
embedding scale, and a bounded contiguous pooling step produces multiple
ordered soft tokens.  Task descriptions live in the normal text condition
channel, so adding a new task does not require a Qwen vocab resize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .axis_b_constants import (
    BASE_MODEL_VOCAB_SIZE,
    MODEL_VOCAB_SIZE,
    QWEN_AXIS_B_MARKER,
    RESERVED_MASK_TOKEN_ID,
    TARGET_TYPE_TOKEN_IDS,
    axis_b_marker_is_compatible,
)

MOLECULE_CONDITION_MODES = frozenset(("direct_qwen", "molbert"))


class ConditionProjection(nn.Module):
    """Map one frozen-encoder representation into the Qwen hidden space."""

    def __init__(
        self,
        encoder_dim: int,
        hidden_size: int,
        *,
        initial_output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(encoder_dim), int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
        )
        # Keep projected prefixes on a Qwen-like norm scale.  The gain remains
        # trainable so alignment can adapt magnitude without exposing the
        # backbone to an unbounded randomly initialized prefix.
        initial_output_scale = max(float(initial_output_scale), 1e-6)
        self.register_buffer(
            "initial_output_scale",
            torch.tensor(initial_output_scale),
            persistent=True,
        )
        self.output_scale = nn.Parameter(torch.tensor(initial_output_scale))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool valid sequence positions without producing NaNs for empties."""

    # Accumulate in fp32 even when the frozen encoder runs in bf16; long
    # protein/text sequences otherwise lose noticeable pooling precision.
    values = hidden_states.float()
    weights = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return total / denominator


class AxisBForwardOutput(NamedTuple):
    """Small stable wrapper around the Hugging Face causal-LM output."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    loss_tokens: int
    qwen_output: Any


class AxisBConditionalModel(nn.Module):
    """Frozen condition encoders plus trainable projections and Qwen3."""

    def __init__(
        self,
        *,
        qwen: nn.Module,
        text_encoder: nn.Module,
        molecule_encoder: nn.Module | None = None,
        encoder_dim: int = 768,
        normalize_text: bool = True,
        normalize_molecule: bool = True,
        # Keep the full encoder sequence by default.  The encoders are capped
        # at 384 (text) and 512 (FDDFS/protein); Axis-B's operational Qwen
        # window is 512, so the collator reserves the remainder for targets.
        condition_token_budget: int = 512,
        molecule_condition_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.text_encoder = text_encoder
        if molecule_condition_mode is None:
            molecule_condition_mode = "molbert" if molecule_encoder is not None else "direct_qwen"
        if molecule_condition_mode not in MOLECULE_CONDITION_MODES:
            raise ValueError(
                f"molecule_condition_mode must be one of {sorted(MOLECULE_CONDITION_MODES)}"
            )
        if molecule_condition_mode == "molbert" and molecule_encoder is None:
            raise ValueError("molbert molecule condition mode requires molecule_encoder")
        if molecule_condition_mode == "direct_qwen" and molecule_encoder is not None:
            raise ValueError("direct_qwen molecule condition mode must not load molecule_encoder")
        self.molecule_condition_mode = str(molecule_condition_mode)
        self.molecule_encoder = molecule_encoder
        hidden_size = int(qwen.config.hidden_size)
        embedding_weight = qwen.get_input_embeddings().weight.detach().float()
        qwen_embedding_norm = max(float(embedding_weight.norm(dim=-1).mean().item()), 1e-6)
        self.text_projection = ConditionProjection(
            encoder_dim, hidden_size, initial_output_scale=qwen_embedding_norm
        )
        self.molecule_projection: ConditionProjection | None = None
        self.normalize_text = bool(normalize_text)
        self.normalize_molecule = bool(normalize_molecule)
        if self.molecule_condition_mode == "molbert":
            self.molecule_projection = ConditionProjection(
                encoder_dim, hidden_size, initial_output_scale=qwen_embedding_norm
            )
        if int(condition_token_budget) <= 0:
            raise ValueError("condition_token_budget must be positive")
        self.condition_token_budget = int(condition_token_budget)
        self._stage = "unconfigured"
        self._embedding_grad_handle: Any | None = None
        self.freeze_condition_encoders()

    @property
    def stage(self) -> str:
        return self._stage

    def freeze_condition_encoders(self) -> None:
        encoders = [self.text_encoder]
        if self.molecule_encoder is not None:
            encoders.append(self.molecule_encoder)
        for encoder in encoders:
            encoder.requires_grad_(False)
            encoder.eval()

    def train(self, mode: bool = True) -> "AxisBConditionalModel":
        super().train(mode)
        # The encoders are frozen for both training stages; keep dropout off.
        self.text_encoder.eval()
        if self.molecule_encoder is not None:
            self.molecule_encoder.eval()
        return self

    def _remove_embedding_hook(self) -> None:
        if self._embedding_grad_handle is not None:
            self._embedding_grad_handle.remove()
            self._embedding_grad_handle = None

    def configure_training_stage(self, stage: str) -> None:
        """Configure the LLaVA-style alignment or full-finetuning stage.

        During alignment the tied embedding/output matrix must remain frozen
        except for the five target-type rows.  A gradient hook masks all other
        rows.  The optimizer must put this matrix in a zero-weight-decay group.
        """

        if stage not in {"alignment", "full"}:
            raise ValueError(f"unknown training stage: {stage!r}")
        self._remove_embedding_hook()
        self.freeze_condition_encoders()

        self.qwen.requires_grad_(stage == "full")
        self.text_projection.requires_grad_(True)
        if self.molecule_projection is not None:
            self.molecule_projection.requires_grad_(True)

        if stage == "alignment":
            embeddings = self.qwen.get_input_embeddings().weight
            embeddings.requires_grad_(True)
            trainable_rows = torch.zeros(
                embeddings.shape[0], dtype=torch.bool, device=embeddings.device
            )
            for token_id in TARGET_TYPE_TOKEN_IDS.values():
                trainable_rows[int(token_id)] = True

            def mask_frozen_rows(gradient: torch.Tensor) -> torch.Tensor:
                return gradient * trainable_rows.to(dtype=gradient.dtype).unsqueeze(1)

            self._embedding_grad_handle = embeddings.register_hook(mask_frozen_rows)

        self._stage = stage

    def trainable_parameter_report(self) -> dict[str, int | str]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "stage": self.stage,
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "frozen_parameters": int(total - trainable),
        }

    def _project_condition_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        present: torch.Tensor,
        projection: ConditionProjection,
        *,
        normalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a masked encoder sequence into a bounded soft-token prefix.

        Contiguous adaptive pooling keeps token order and local structure while
        bounding Qwen memory for long text/protein conditions.  Inputs shorter
        than the budget are passed through position-by-position.
        """

        batch_size = int(hidden_states.shape[0])
        encoder_width = int(hidden_states.shape[1])
        hidden_size = int(self.qwen.config.hidden_size)
        max_slots = max(1, min(encoder_width, self.condition_token_budget))
        lengths = attention_mask.to(torch.long).sum(dim=1)
        lengths = torch.minimum(
            lengths,
            torch.full_like(lengths, self.condition_token_budget),
        )
        lengths = lengths * present.to(torch.long)
        projected_prefix = torch.zeros(
            (batch_size, max_slots, hidden_size),
            dtype=projection.net[0].weight.dtype,
            device=hidden_states.device,
        )
        if not bool(present.any()):
            return projected_prefix, lengths

        values = hidden_states.float()
        if normalize:
            values = F.normalize(values, p=2, dim=-1)
        projected = projection(values.to(projection.net[0].weight.dtype))
        # Both condition modalities are now calibrated to the pretrained
        # token-embedding manifold.  In particular, raw molBERT hidden states
        # must not inject a much larger RMS than Qwen token embeddings.
        projected = F.normalize(projected.float(), p=2, dim=-1)
        minimum_scale = projection.initial_output_scale.float() * 0.25
        projected = projected * projection.output_scale.float().clamp_min(minimum_scale)
        for row in range(batch_size):
            if not bool(present[row]):
                continue
            valid = projected[row][attention_mask[row].to(torch.bool)]
            if valid.numel() == 0:
                continue
            slots = min(int(valid.shape[0]), self.condition_token_budget)
            if int(valid.shape[0]) > slots:
                # [L,H] -> adaptive pool over L while retaining order.
                compact = F.adaptive_avg_pool1d(
                    valid.transpose(0, 1).unsqueeze(0), slots
                ).squeeze(0).transpose(0, 1)
            else:
                compact = valid[:slots]
            projected_prefix[row, :slots] = compact
        return projected_prefix, lengths

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = output.last_hidden_state
        return self._project_condition_sequence(
            hidden,
            attention_mask,
            present,
            self.text_projection,
            normalize=self.normalize_text,
        )

    def _encode_molecule(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.molecule_condition_mode != "molbert" or self.molecule_encoder is None:
            raise RuntimeError("_encode_molecule is only available in legacy molbert mode")
        with torch.no_grad():
            output = self.molecule_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = output.last_hidden_state
        return self._project_condition_sequence(
            hidden,
            attention_mask,
            present,
            self.molecule_projection,
            normalize=self.normalize_molecule,
        )

    def build_qwen_inputs(
        self,
        *,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        decoder_labels: torch.Tensor | None,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_present: torch.Tensor,
        molecule_input_ids: torch.Tensor,
        molecule_attention_mask: torch.Tensor,
        molecule_present: torch.Tensor,
        prefix_lengths: torch.Tensor | None = None,
        text_prefix_lengths: torch.Tensor | None = None,
        molecule_prefix_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape [batch, length]")
        batch_size = int(decoder_input_ids.shape[0])
        for name, value in {
            "text_present": text_present,
            "molecule_present": molecule_present,
        }.items():
            if value.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [{batch_size}]")

        text_prefix, text_lengths = self._encode_text(
            text_input_ids, text_attention_mask, text_present
        )
        if self.molecule_condition_mode == "direct_qwen":
            molecule_input_ids = molecule_input_ids.to(dtype=torch.long)
            vocab_size = int(self.qwen.get_input_embeddings().weight.shape[0])
            if bool((molecule_input_ids < 0).any()) or bool((molecule_input_ids >= vocab_size).any()):
                raise ValueError("direct molecule condition contains an out-of-range Qwen token id")
            molecule_prefix = self.qwen.get_input_embeddings()(molecule_input_ids)
            molecule_lengths = molecule_attention_mask.to(torch.long).sum(dim=1)
            molecule_lengths = molecule_lengths * molecule_present.to(torch.long)
        else:
            molecule_prefix, molecule_lengths = self._encode_molecule(
                molecule_input_ids, molecule_attention_mask, molecule_present
            )
        for name, explicit, natural in (
            ("text_prefix_lengths", text_prefix_lengths, text_lengths),
            ("molecule_prefix_lengths", molecule_prefix_lengths, molecule_lengths),
        ):
            if explicit is None:
                continue
            explicit = explicit.to(device=decoder_input_ids.device, dtype=torch.long)
            if explicit.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [{batch_size}]")
            if bool((explicit < 0).any()) or bool((explicit > natural).any()):
                raise ValueError(f"{name} exceeds the encoded condition length")
            if name == "text_prefix_lengths":
                text_lengths = explicit
            else:
                molecule_lengths = explicit
        token_embeddings = self.qwen.get_input_embeddings()(decoder_input_ids)
        expected_prefix_lengths = text_lengths + molecule_lengths
        if prefix_lengths is None:
            prefix_lengths = expected_prefix_lengths
        else:
            prefix_lengths = prefix_lengths.to(
                device=decoder_input_ids.device, dtype=torch.long
            )
            if not torch.equal(prefix_lengths, expected_prefix_lengths):
                raise ValueError("prefix_lengths disagrees with condition presence flags")

        decoder_lengths = decoder_attention_mask.to(torch.long).sum(dim=1)
        full_lengths = prefix_lengths + decoder_lengths
        max_full_length = int(full_lengths.max().item())
        hidden_size = int(token_embeddings.shape[-1])
        inputs_embeds = torch.zeros(
            (batch_size, max_full_length, hidden_size),
            dtype=token_embeddings.dtype,
            device=token_embeddings.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_full_length),
            dtype=decoder_attention_mask.dtype,
            device=decoder_attention_mask.device,
        )
        labels: torch.Tensor | None = None
        if decoder_labels is not None:
            labels = torch.full(
                (batch_size, max_full_length),
                -100,
                dtype=decoder_labels.dtype,
                device=decoder_labels.device,
            )

        # Text is first and molecule is second when both are present.  Direct
        # molecule mode contributes the original ordered Qwen token span;
        # legacy mode contributes projected soft tokens.
        for row in range(batch_size):
            cursor = 0
            text_length = int(text_lengths[row].item())
            molecule_length = int(molecule_lengths[row].item())
            if text_length:
                inputs_embeds[row, cursor : cursor + text_length] = text_prefix[
                    row, :text_length
                ].to(token_embeddings.dtype)
                attention_mask[row, cursor : cursor + text_length] = 1
                cursor += text_length
            if molecule_length:
                inputs_embeds[row, cursor : cursor + molecule_length] = molecule_prefix[
                    row, :molecule_length
                ].to(token_embeddings.dtype)
                attention_mask[row, cursor : cursor + molecule_length] = 1
                cursor += molecule_length
            decoder_length = int(decoder_lengths[row].item())
            inputs_embeds[row, cursor : cursor + decoder_length] = token_embeddings[
                row, :decoder_length
            ]
            attention_mask[row, cursor : cursor + decoder_length] = 1
            if labels is not None:
                labels[row, cursor : cursor + decoder_length] = decoder_labels[
                    row, :decoder_length
                ]

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "prefix_lengths": prefix_lengths,
            "full_lengths": full_lengths,
            "text_prefix_lengths": text_lengths,
            "molecule_prefix_lengths": molecule_lengths,
        }

    def forward(self, **batch: torch.Tensor) -> AxisBForwardOutput:
        qwen_inputs = self.build_qwen_inputs(**batch)
        qwen_call = {
            key: value
            for key, value in qwen_inputs.items()
            if key in {"inputs_embeds", "attention_mask", "labels"}
        }
        output = self.qwen(**qwen_call, use_cache=False, return_dict=True)
        labels = qwen_inputs["labels"]
        loss_tokens = 0
        if labels is not None:
            # Qwen shifts labels internally, so the first position is excluded.
            loss_tokens = int((labels[:, 1:] != -100).sum().item())
        return AxisBForwardOutput(
            loss=output.loss,
            logits=output.logits,
            loss_tokens=loss_tokens,
            qwen_output=output,
        )

    def save_axis_b_pretrained(
        self,
        output_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save trainable model state without duplicating frozen encoders."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # The Qwen sub-checkpoint must carry enough immutable interface
        # identity for a 17,213-row resume to be distinguishable from an
        # arbitrary resized checkpoint before any weight tensor is accepted.
        self.qwen.config.axis_b_checkpoint_interface = json.loads(
            json.dumps(QWEN_AXIS_B_MARKER)
        )
        self.qwen.save_pretrained(
            output_dir / "qwen3", safe_serialization=True, max_shard_size="5GB"
        )
        projection_state = {
            f"text_projection.{key}": value.detach().cpu().contiguous()
            for key, value in self.text_projection.state_dict().items()
        }
        if self.molecule_projection is not None:
            projection_state.update(
                {
                    f"molecule_projection.{key}": value.detach().cpu().contiguous()
                    for key, value in self.molecule_projection.state_dict().items()
                }
            )
        try:
            from safetensors.torch import save_file

            save_file(projection_state, output_dir / "condition_projectors.safetensors")
        except ImportError:
            torch.save(projection_state, output_dir / "condition_projectors.pt")

        text_max_length = (metadata or {}).get("text_max_length")
        if (
            text_max_length is not None
            and int(text_max_length) <= int(self.condition_token_budget)
        ):
            text_reduction = (
                "tokenizer_truncate_then_one_to_one_per_token_projection_no_pooling"
            )
        else:
            text_reduction = (
                "input_l2_normalize_then_token_projection_then_per_token_l2_normalize_"
                "with_qwen_rms_gain_then_ordered_adaptive_pool_if_over_budget"
            )

        payload = {
            "format_version": 2,
            "stage": self.stage,
            "model_vocab_size": MODEL_VOCAB_SIZE,
            "reserved_mask_token_id": RESERVED_MASK_TOKEN_ID,
            "target_type_token_ids": TARGET_TYPE_TOKEN_IDS,
            "operational_context": int((metadata or {}).get("max_context", 512)),
            "molecule_condition_mode": self.molecule_condition_mode,
            "molecule_component_separator": {
                "raw_bpe_id": 2,
                "model_id": 5,
                "name": "FDDFS_<SEP>",
            },
            "condition_pooling": {
                "text": text_reduction,
            },
            "target_type_initialization": "mean_resizing_from_existing_qwen_embedding_distribution",
            "condition_token_budget": self.condition_token_budget,
        }
        if metadata:
            payload["metadata"] = metadata
        (output_dir / "axis_b_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def resize_qwen_vocab_for_axis_b(qwen: nn.Module) -> None:
    """Append the six fixed rows or validate an already-complete checkpoint.

    A formal base must have exactly 17,207 rows.  An already-17,213-row model
    is accepted only when its Qwen config carries a semantically compatible
    Axis-B marker. Transformers must support and
    actually receive ``mean_resizing=True``; silently falling back to ordinary
    random resizing would violate the frozen initialization contract.
    """

    old_size = int(qwen.get_input_embeddings().weight.shape[0])
    declared_size = int(getattr(qwen.config, "vocab_size", -1))
    if declared_size != old_size:
        raise ValueError(
            "Qwen config/embedding vocabulary mismatch: "
            f"config={declared_size} embeddings={old_size}"
        )
    if getattr(qwen.config, "tie_word_embeddings", None) is not True:
        raise ValueError("Axis-B requires tie_word_embeddings=true")
    if old_size == BASE_MODEL_VOCAB_SIZE:
        if getattr(qwen.config, "axis_b_checkpoint_interface", None) is not None:
            raise ValueError("17,207-row base unexpectedly claims an Axis-B checkpoint")
        try:
            qwen.resize_token_embeddings(MODEL_VOCAB_SIZE, mean_resizing=True)
        except TypeError as exc:
            raise RuntimeError(
                "installed Transformers cannot honor the required "
                "resize_token_embeddings(mean_resizing=True) contract"
            ) from exc
    elif old_size == MODEL_VOCAB_SIZE:
        marker = getattr(qwen.config, "axis_b_checkpoint_interface", None)
        if not axis_b_marker_is_compatible(marker):
            raise ValueError(
                "17,213-row Qwen is not a complete legal Axis-B checkpoint"
            )
    else:
        raise ValueError(
            "formal Axis-B Qwen vocab must be exactly 17,207 (base) or "
            f"17,213 (validated resume), got {old_size}"
        )
    qwen.config.vocab_size = MODEL_VOCAB_SIZE
    qwen.config.axis_b_checkpoint_interface = json.loads(
        json.dumps(QWEN_AXIS_B_MARKER)
    )
    if int(qwen.get_input_embeddings().weight.shape[0]) != MODEL_VOCAB_SIZE:
        raise AssertionError("Qwen embedding resize did not take effect")
    qwen.tie_weights()
    output_embeddings = qwen.get_output_embeddings()
    if (
        output_embeddings is None
        or int(output_embeddings.weight.shape[0]) != MODEL_VOCAB_SIZE
    ):
        raise AssertionError("Qwen tied output embedding resize did not take effect")


def load_projection_state(model: AxisBConditionalModel, checkpoint_dir: Path) -> None:
    """Load projectors from an Axis-B checkpoint directory."""

    checkpoint_dir = Path(checkpoint_dir)
    safe_path = checkpoint_dir / "condition_projectors.safetensors"
    torch_path = checkpoint_dir / "condition_projectors.pt"
    if safe_path.is_file():
        from safetensors.torch import load_file

        state = load_file(safe_path, device="cpu")
    elif torch_path.is_file():
        state = torch.load(torch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"missing condition projector weights under {checkpoint_dir}"
        )

    text_prefix = "text_projection."
    model.text_projection.load_state_dict(
        {key[len(text_prefix) :]: value for key, value in state.items() if key.startswith(text_prefix)},
        strict=False,
    )
    if model.molecule_projection is not None:
        molecule_prefix = "molecule_projection."
        model.molecule_projection.load_state_dict(
            {
                key[len(molecule_prefix) :]: value
                for key, value in state.items()
                if key.startswith(molecule_prefix)
            },
            strict=False,
        )
