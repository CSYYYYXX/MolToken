import unittest
from unittest.mock import patch

from cli.data import materialize_rabd_cdrh3_sft as module


class RAbDMaterializationTests(unittest.TestCase):
    def test_extracts_only_flanks_as_molecular_condition(self):
        row = module.extract_pair({"seq": "ACDEWGKLMN", "cdr": "0000330000", "pdb": "toy"}, "train", 1)
        self.assertEqual((row["left"], row["target"], row["right"]), ("ACDE", "WG", "KLMN"))
        self.assertEqual(row["requested_length"], 2)

    def test_regular_official_crop_is_first_130_residues(self):
        seq = "A" * 160
        labels = "0" * 100 + "3" * 15 + "0" * 45
        cropped, cdr, start, end = module.official_crop(seq, labels)
        self.assertEqual((len(cropped), len(cdr), start, end), (130, 130, 0, 130))
        self.assertEqual(cdr.count("3"), 15)

    def test_shifted_official_crop_preserves_target_and_right_flank(self):
        seq = "A" * 160
        labels = "0" * 125 + "3" * 11 + "0" * 24
        row = module.extract_pair({"seq": seq, "cdr": labels, "pdb": "long"}, "test", 1)
        self.assertEqual((row["crop_start"], row["crop_end_exclusive"]), (15, 145))
        self.assertEqual((len(row["left"]), len(row["target"]), len(row["right"])), (110, 11, 9))

    def test_upstream_missing_cdr_exclusion(self):
        for cdr in (None, "000"):
            self.assertIsNone(module.extract_pair({"seq": "ACD", "cdr": cdr, "pdb": "none"}, "train", 1))

    def test_mismatched_annotations_raise(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            module.official_crop("ACDE", "030")

    def test_discontinuous_cdr_raises(self):
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            module.extract_pair({"seq": "ACDEF", "cdr": "03030", "pdb": "bad"}, "train", 1)

    def test_partial_chain_preserves_empty_flank(self):
        row = module.extract_pair({"seq": "ACDEF", "cdr": "00033", "pdb": "partial"}, "train", 1)
        self.assertEqual((row["left"], row["target"], row["right"]), ("ACD", "EF", ""))
        self.assertEqual(module.encode_component(""), ({"fddfs_tokens": 0}, [], []))

    def test_record_has_only_numeric_text_and_separate_flanks(self):
        class Codec:
            def bpe_ids(self, tokens, label):
                self.joined = list(tokens)
                return list(range(len(tokens)))
        class Decoder:
            def decode_raw_ids(self, ids):
                return ids, codec.joined
        def component(sequence):
            return {"fddfs_tokens": len(sequence)}, list(sequence), [10] * len(sequence)
        codec = Codec()
        row = module.extract_pair({"seq": "ACDEWGKLMN", "cdr": "0000330000", "pdb": "toy"}, "test", 1)
        with patch.object(module, "CODEC", codec), patch.object(module, "DECODER", Decoder()), \
             patch.object(module, "TEXT_LENGTHS", {2: 3}), patch.object(module, "encode_component", component):
            record = module.encode_pair(row)
        self.assertEqual(codec.joined, [*"ACDE", "<SEP>", *"KLMN"])
        self.assertEqual(record["condition"]["text"], "2")
        self.assertEqual(set(record["condition"]), {"type", "text", "molecule_bpe_ids"})
        self.assertEqual(record["meta"]["full_context_tokens"], 9 + 2 + 3 + 3)
        self.assertEqual(record["target_bpe_ids"], [10, 10])


if __name__ == "__main__":
    unittest.main()
