# TRACE_hic

TRACE_hic predicts Hi-C from DNA sequence and a cell's RNA context. Its
DNA representation model is SUCCEED, which is frozen while the shared
multi-cell TRACE_hic prediction head is trained. This repository trains the
head across multiple human cell types and predicts one 2,097,152-bp hg38
window at a time. The head is trained with balanced, clipped
10-kb contacts, resized to 256 × 256, transformed by `log1p`, and normalized
per window over valid pixels. The output is a 256 × 256 matrix in `[0, 1]`.
It is **not** an estimate of raw contact counts.

## Install

Use Python 3.10+ and a PyTorch build appropriate for your CUDA installation.

```bash
pip install -e .
```

Model weights are not in Git. Put the two trusted files at `weights/stage1.pt`
and `weights/stage2.ckpt`; see [artifacts.json](artifacts.json) for their hashes.
The model contract and 64,217-gene vocabulary are in `assets/model_contract/`.

## Inputs

Training expects a data root with this layout:

```text
DATA_ROOT/
  dna_sequence/chr1.fa.gz ... chrX.fa.gz
  GM12878/RNA/context.npz
  GM12878/hic_matrix_balanced_clip93/metadata.json
  GM12878/hic_matrix_balanced_clip93/chr1.npz ... chrX.npz
  IMR90/...  H1-hESC/...  K562/...
```

Prepare each RNA context from TPM tables with `gene_id` and `TPM` columns:

```bash
trace-hic-rna-context --rna-tsv sample1.tsv sample2.tsv \
  --celltype GM12878 --rna-protocol total_rna \
  --output DATA_ROOT/GM12878/RNA/context.npz
```

To export training labels from a 10-kb Cooler with balancing weights, first
determine a cell-specific 93rd-percentile clipping threshold using your chosen
training protocol, then run:

```bash
trace-hic-prepare-labels --cool INPUT.cool \
  --clip-threshold THRESHOLD --output-dir DATA_ROOT/GM12878/hic_matrix_balanced_clip93
```

The exporter records the threshold and preprocessing operations in
`metadata.json`; the trainer checks them. Supply the same genome assembly,
reference DNA, gap regions, and clipping protocol across cells.

## Train the Hi-C head

```bash
trace-hic-train --data-root DATA_ROOT \
  --celltypes GM12878,IMR90,H1-hESC,K562 \
  --save-path results/multicell_run \
  --devices 4 --batch-size-per-device 6 --succeed-tile-batch-size 2 \
  --precision bf16-mixed
```

The original run used seed 2179, `max_epochs=80`, four devices, batch size 6
per device, and the four cells shown above. Chromosomes 10 and 15 and X are
excluded from training; chromosome 10 is validation. The trainer selects
checkpoints by macro validation loss across cells.
The default centromere/telomere and gap BEDs are in `assets/regions/` and can
be overridden with the corresponding CLI flags.

## Predict without experimental Hi-C

```bash
trace-hic-predict \
  --context-npz DATA_ROOT/HepG2/RNA/context.npz \
  --fasta DATA_ROOT/dna_sequence/chr15.fa.gz \
  --chromosome chr15 --start 44000000 \
  --output outputs/hepg2_chr15_44000000.npz
```

The output NPZ contains `prediction` and `metadata_json`. Start is zero based;
the predicted interval is `[start, start + 2097152)`. The reference FASTA may
be gzip compressed. Windows outside the chromosome are rejected. RNA can be
from a training cell or an unseen cell. Experimental Hi-C is not read during
prediction. Use `--precision float32` for CPU execution; a GPU is recommended.

## Provenance and publication

The Hi-C components were extracted from the local SUCCEED project. The
SUCCEED backbone implementation was extracted from the local Stage-I source
to remove runtime dependencies on a sibling checkout. The exact trained
gene vocabulary and model contract are included. Checkpoint formats and RNA
provenance hashes are validated when loading the model. Historical identifiers
inside the model contract and checkpoint are preserved for weight compatibility.

The source projects did not contain a top-level license covering these
extracted files. Confirm redistribution permission and choose a repository
license before making this repository public. No reference genome, raw RNA,
Hi-C data, or trained weights are tracked by Git.
