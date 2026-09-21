# CEBRA

Implements [CEBRA](https://doi.org/10.1038/s41586-023-06031-6) as a TS1 single-session
baseline: the encoder is fit in-run on the training split, then an MLP reads out the task.

## Install

`cebra` constrains the rest of the environment (numpy below 2.0 on Windows), so it is
kept out of the `train` extra and installed alongside it:

```bash
uv pip install -e ".[train,cebra]"
```

## Usage

Run from the repo root:

```bash
python src/ts1/train.py trainer=cebra data_root=<data-root> \
    recording_id=<session_id> task=<task>
```

Only single-session inference is supported, so there is no pretrained variant.
