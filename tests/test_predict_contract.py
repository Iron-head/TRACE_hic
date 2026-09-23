"""Small tests for the public prediction input and checkpoint boundary."""

from pathlib import Path

import numpy as np
import pytest
import torch

from trace_hic.inference.predict import load_checkpoint, sequence_context
from trace_hic.model.enformer_hic_standalone import (
    ENFORMER_CONTEXT_BP,
    ENFORMER_FLANK_BP,
    HIC_WINDOW_BP,
)


def test_sequence_context_pads_left_flank_and_preserves_base_order(tmp_path: Path):
    fasta = tmp_path / "chrTest.fa"
    fasta.write_text(">chrTest\n" + "ACGTN" * ((HIC_WINDOW_BP + 10) // 5) + "\n")
    encoded = sequence_context(fasta, 0)
    assert encoded.shape == (ENFORMER_CONTEXT_BP, 4)
    assert not encoded[:ENFORMER_FLANK_BP].any()
    np.testing.assert_array_equal(
        encoded[ENFORMER_FLANK_BP : ENFORMER_FLANK_BP + 5],
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]),
    )
    with pytest.raises(ValueError, match="exceeds FASTA length"):
        sequence_context(fasta, 20)


def test_stage2_loader_rejects_wrong_checkpoint_schema(tmp_path: Path):
    path = tmp_path / "wrong.ckpt"
    torch.save({"balanced_clip_minmax_multicell_checkpoint_schema": "other"}, path)
    with pytest.raises(ValueError, match="balanced, clipped"):
        load_checkpoint(path)
