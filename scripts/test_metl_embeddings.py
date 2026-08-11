import tempfile
import unittest
from pathlib import Path

import numpy as np

from metl_embeddings import METLEmbeddingSource


class DeterministicMETLSource(METLEmbeddingSource):
    def _embed_uncached(self, sequences):
        return [
            np.asarray(
                [[ord(residue), position] for position, residue in enumerate(sequence, start=1)],
                dtype=np.float32,
            )
            for sequence in sequences
        ]


class METLEmbeddingTests(unittest.TestCase):
    def test_position_arrays_preserve_order_and_use_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source = DeterministicMETLSource(
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

    def test_3d_model_requires_existing_pdb(self):
        with self.assertRaisesRegex(ValueError, "requires the NylC wild-type PDB"):
            METLEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C"},
                model_id="metl-g-20m-3d",
            )
        with self.assertRaises(FileNotFoundError):
            METLEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C"},
                model_id="metl-g-20m-3d",
                pdb_path=Path("missing.pdb"),
            )

    def test_rejects_published_local_models_for_nylc(self):
        with self.assertRaisesRegex(ValueError, "METL-Global"):
            METLEmbeddingSource(
                wt_sequence="ACDEF",
                wt_pocket={2: "C"},
                model_id="metl-l-2m-1d-gfp",
            )


if __name__ == "__main__":
    unittest.main()
