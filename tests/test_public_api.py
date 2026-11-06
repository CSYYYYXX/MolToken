"""Standalone release contracts on synthetic inputs and tiny models."""
import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from moltoken import MolTokenizer, load_pretrained
from moltoken.checkpoints import checkpoint_path_identity
from moltoken.encode_corpus import encode_corpus
from pipeline.finetune.fddfs_encode import build_graph


@pytest.fixture(scope="module")
def tokenizer():
    return MolTokenizer()


@pytest.mark.parametrize("domain,sequence", [
    ("small_molecule", "CCO"), ("small_molecule", "c1ccccc1"),
    ("peptide", "AGK"), ("protein", "ACDEFG"), ("dna", "ACGT"), ("rna", "ACGU"),
])
def test_five_domain_round_trip(tokenizer, domain, sequence):
    raw = tokenizer.encode(sequence, domain)
    framed = tokenizer.encode(sequence, domain, model_ids=True)
    assert framed == [1, *(i+3 for i in raw), 2]
    decoded = tokenizer.decode_graph(raw)
    kwargs = {"smiles": sequence} if domain == "small_molecule" else {"sequence": sequence}
    original = build_graph(target_type=domain, sample_id="test", **kwargs)
    # DFS relabels atoms. Compare attributed graph isomorphism, not array order.
    import networkx as nx
    def as_graph(g):
        out = nx.DiGraph()
        for i, features in enumerate(g["node_features"]):
            out.add_node(i, features=tuple(int(x) for x in features))
        for u, v, features in zip(*g["edges"], g["edge_features"], strict=True):
            out.add_edge(int(u), int(v), features=tuple(int(x) for x in features))
        return out
    assert nx.is_isomorphic(as_graph(decoded), as_graph(original),
                            node_match=lambda a,b: a == b, edge_match=lambda a,b: a == b)
    assert tokenizer.decode_tokens(framed, model_ids=True) == tokenizer.decode_tokens(raw)
    assert tokenizer.raw_vocab_size == 17204
    assert tokenizer.vocab_size == 17207
    if domain != "small_molecule":
        assert tokenizer.decode_sequence(raw, domain) == sequence


def test_invalid_ids_fail(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.decode_tokens([17204])
    with pytest.raises(ValueError):
        tokenizer.decode_tokens([1, 94, 2, 96], model_ids=True)


def tiny_model():
    config = Qwen3Config(vocab_size=17207, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=16, tie_word_embeddings=True, max_position_embeddings=512,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2, rope_theta=1e6)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize("layout", ["hf_safe", "hf_sharded", "packed_bin"])
def test_checkpoint_layout_equivalence(tmp_path, layout):
    torch.manual_seed(17)
    original = tiny_model()
    ids = torch.tensor([[94, 1971, 2]])
    with torch.no_grad():
        expected = original(ids).logits
    if layout.startswith("hf"):
        original.save_pretrained(tmp_path, safe_serialization=True,
                                  max_shard_size="1MB" if layout == "hf_sharded" else "1GB")
    else:
        original.config.save_pretrained(tmp_path)
        state = {k.removeprefix("model."): v for k,v in original.state_dict().items()}
        torch.save(state, tmp_path / "pytorch_model.bin")
    loaded = load_pretrained(tmp_path)
    with torch.no_grad():
        observed = loaded(ids).logits
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert loaded.get_input_embeddings().weight is loaded.get_output_embeddings().weight


def test_missing_weight_is_not_randomly_initialized(tmp_path):
    model = tiny_model()
    model.config.save_pretrained(tmp_path)
    state = model.state_dict()
    del state["model.layers.0.self_attn.q_proj.weight"]
    torch.save(state, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="missing model weights"):
        load_pretrained(tmp_path)


def test_legacy_v1_mean_pool_is_not_v2_token_projection(tmp_path):
    from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizerFast
    from safetensors.torch import save_file
    from moltoken import load_sft
    from pipeline.finetune.axis_b_model import ConditionProjection, masked_mean_pool
    from torch.nn import functional as F
    torch.manual_seed(4)
    encoder = tmp_path / 'text'
    molecule = tmp_path / 'molecule'
    ckpt = tmp_path / 'checkpoint'
    encoder.mkdir()
    ckpt.mkdir()
    (encoder/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\n')
    BertTokenizerFast(vocab_file=str(encoder/'vocab.txt')).save_pretrained(encoder)
    config = BertConfig(vocab_size=20, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2)
    BertModel(config).save_pretrained(encoder)
    BertForMaskedLM(config).save_pretrained(molecule)
    tiny_model().save_pretrained(ckpt/'qwen3')
    projections = {name: ConditionProjection(16, 32) for name in
                   ('text_projection', 'molecule_projection')}
    weights = {f'{name}.{k}': v.contiguous() for name, projection in projections.items()
               for k,v in projection.state_dict().items() if k.startswith('net.')}
    save_file(weights, ckpt/'condition_projectors.safetensors')
    cfg = {'format_version': 1, 'max_soft_prefix_slots': 2,
           'condition_pooling': {'text': 'attention_mask_mean_then_l2_normalize',
                                 'molecule': 'attention_mask_mean'}}
    (ckpt/'axis_b_config.json').write_text(json.dumps(cfg))
    model, _ = load_sft(ckpt, text_encoder=encoder, molecule_encoder=molecule)
    hidden = torch.randn(2, 3, 16)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    present = torch.tensor([True, False])
    for name, normalize in [('text_projection', True), ('molecule_projection', False)]:
        observed, lengths = model._project_condition_sequence(
            hidden, mask, present, getattr(model, name), normalize=normalize)
        pooled = masked_mean_pool(hidden, mask)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        expected = projections[name](pooled).unsqueeze(1) * present[:, None, None]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert lengths.tolist() == [1, 0]
    saved = tmp_path/'saved'
    model.save_axis_b_pretrained(saved)
    reloaded, _ = load_sft(saved, text_encoder=encoder, molecule_encoder=molecule)
    assert reloaded.condition_token_budget == 1


def test_path_relocation_is_explicit(tmp_path, monkeypatch):
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"/old/data": str(tmp_path / "data")}))
    monkeypatch.setenv("MOLTOKEN_PATH_MAP", str(mapping))
    assert checkpoint_path_identity("/old/data/a") == str(tmp_path / "data/a")
    assert checkpoint_path_identity("/old/database/a") == "/old/database/a"


def test_compact_file_uses_int32_and_preserves_records(tmp_path):
    from cli.data.build_compact_int_index import read_header, HEADER_STRUCT
    import struct
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"id": "x", "domain": "small_molecule", "sequence": "CCO"}) + "\n")
    target = tmp_path / "encoded.bin"
    result = encode_corpus(source, target)
    with target.open("rb") as handle:
        assert read_header(handle, target) == (1, 2, 1968)
        assert struct.unpack("<I", handle.read(4))[0] == 2
        assert struct.unpack("<2i", handle.read(8)) == (91, 1968)
        assert not handle.read(1)
    assert result["records"] == 1
