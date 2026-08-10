import tempfile
import unittest
from pathlib import Path

import numpy as np

from protein_llm_embeddings import (
    ProteinLLMEmbeddingSource,
    build_variant_sequence,
    read_fasta_sequence,
)


class DeterministicEmbeddingSource(ProteinLLMEmbeddingSource):
    def _embed_uncached(self, sequences):
        return [
            np.asarray(
                [[ord(residue), position] for position, residue in enumerate(sequence, start=1)],
                dtype=np.float32,
            )
            for sequence in sequences
        ]


class ProteinLLMEmbeddingTests(unittest.TestCase):
    def test_read_fasta_and_build_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wt.fasta"
            path.write_text(">chain A\nACD\nEF\n", encoding="utf-8")
            self.assertEqual(read_fasta_sequence(path), "ACDEF")
        self.assertEqual(
            build_variant_sequence("ACDEF", {2: "Y", 4: "W"}, {2: "C", 4: "E"}),
            "AYDWF",
        )

    def test_position_arrays_preserve_order_and_use_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DeterministicEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C", 4: "E"},
                cache_dir=directory,
            )
            pockets = [{2: "Y", 4: "W"}, {2: "C", 4: "E"}, {2: "Y", 4: "W"}]
            arrays = source.position_arrays(pockets, positions=(2, 4))
            np.testing.assert_array_equal(arrays[2][:, 0], [ord("Y"), ord("C"), ord("Y")])
            np.testing.assert_array_equal(arrays[4][:, 0], [ord("W"), ord("E"), ord("W")])

            source._embed_uncached = lambda *_: self.fail("cache should avoid recomputation")
            cached_arrays = source.position_arrays(pockets, positions=(2, 4))
            np.testing.assert_array_equal(cached_arrays[2], arrays[2])

    def test_global_and_window_delta_representations(self):
        with tempfile.TemporaryDirectory() as directory:
            pocket = {2: "Y", 4: "W"}
            global_source = DeterministicEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C", 4: "E"},
                cache_dir=directory,
                representation="global_delta",
                name="esm2_global_delta",
            )
            global_features = global_source.feature_matrix([pocket, {2: "C", 4: "E"}])
            self.assertEqual(global_features.shape, (2, 10))
            np.testing.assert_array_equal(global_features[1], np.zeros(10))
            self.assertGreater(np.linalg.norm(global_features[0]), 0)

            window_source = DeterministicEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C", 4: "E"},
                cache_dir=directory,
                representation="window_delta",
                window_radius=1,
                name="esm2_window_delta",
            )
            windows = window_source.position_arrays([pocket], positions=(2, 4))
            self.assertEqual(windows[2].shape, (1, 6))
            self.assertEqual(windows[4].shape, (1, 6))

    def test_wild_type_numbering_mismatch_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "WT mismatch"):
            ProteinLLMEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "D"},
                cache_dir="unused",
            )


if __name__ == "__main__":
    unittest.main()
