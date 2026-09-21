# NuCLR

Implements [NuCLR](https://arxiv.org/abs/2512.01199) based on the [official implementation](https://github.com/nerdslab/nuclr).

How to run it, and how a checkpoint becomes unit embeddings, is in
`src/pretrain/README.md`.

## Differences from official implementation

### 1. No probe-level data file splitting

**Original:** The official implementation first splits the data files into probe-level files.
This is done as a way to avoid performing across-probe contrast in the loss.

**This:** Here we do not split the files. We handle the within-probe-only contrast
by passing ``probe_id`` to the loss which applies appropriate masks such that across-probe contrast
is not performed.

**Effect:** This will change the composition of batches during training, leading to slight differences
in training dynamics. This should only have a minor effect.
