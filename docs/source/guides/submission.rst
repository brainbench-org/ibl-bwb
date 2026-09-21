.. _submission_guide:

Building and Submitting a Model
================================

This guide walks through the full path from a model to a leaderboard submission:
pretraining, evaluation and saving predictions, scaling up to a full submission, and
the submission itself. Throughout this guide, we maintain a worked example with
NDT-stitch using the released checkpoint.

.. contents:: On this page
   :local:
   :depth: 2

How to submit a model, end to end
----------------------------------

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: 1. Pretrain
      :link: submission-pretrain
      :link-type: ref

      Train a shared model once for all the tasks, or directly use a released checkpoint.

   .. grid-item-card:: 2. Evaluate and save predictions
      :link: submission-evaluate
      :link-type: ref

      Evaluate the model on the suite you're targeting and save the predictions. TS1 and TS2 write
      prediction files as part of the same run; TS3 scores unit embeddings.

   .. grid-item-card:: 3. Scale up
      :link: submission-scale
      :link-type: ref

      A submission needs at least three seeds and predictions over all eval recordings. You can
      either do this manually, or fan it out using the repo's Ray-based scripts.

   .. grid-item-card:: 4. Submit
      :link: submission-submit
      :link-type: ref

      Upload your prediction files to the leaderboard. Scoring runs server-side, you just need to
      provide the predictions.

Prerequisites for the worked example (NDT-stitch)
-------------------------------------------------

NDT-stitch has a trainer config on all three suites, which makes it a convenient example
to follow through this whole guide. Download its released checkpoint once and reuse it
throughout:

.. code-block:: bash

   mkdir -p ckpt
   curl -L https://huggingface.co/nerdslab/ibl-bwb-ndt_stitch/resolve/main/ndt_stitch.pt \
       -o ckpt/ndt_stitch.pt

See :ref:`pretrain_released_ckpts` for the full checkpoint list and the
``huggingface_hub`` alternative. If you're bringing your own model instead, see
:ref:`pretraining_guide` and the suite guides (:ref:`ts1_guide`, :ref:`ts2_guide`,
:ref:`ts3_guide`) for the model interface.

.. _submission-pretrain:

1. Pretrain
-----------------------

See :ref:`pretraining_guide` for the full pretraining walkthrough: dataset classes, the trainer
contract, and how to launch a run. The output is a checkpoint that the next step loads with ``ckpt.load_from``.
For our example, we will use the released checkpoint for NDT-stitch, which we 
downloaded in the prerequisites section.

.. _submission-evaluate:

2. Evaluate and save predictions
---------------------------------

TS1 and TS2
~~~~~~~~~~~

Training and testing are one call: ``train()`` runs the standardized test protocol
automatically at the end, and writes prediction files when ``save_preds.enable=true``:

.. code-block:: bash

   python src/ts1/train.py trainer=<trainer> data_root=<data_root> \
       save_preds.enable=true save_preds.label=<submission-id>

``save_preds.label`` is your submission id, which you can choose as you like. Files land at
``predictions/<label>/<task>/<recording_id>/seed_<seed>.safetensors``, under
``BWB_PREDICTIONS_DIR`` (default ``predictions/``).

**Example: NDT-stitch on TS1**, finetuning the checkpoint on one task and recording:

.. code-block:: bash

   python src/ts1/train.py trainer=ndt_stitch_finetune data_root=<data_root> \
       task=wheel_speed recording_id=<recording_id> \
       ckpt.load_from=$PWD/ckpt/ndt_stitch.pt \
       save_preds.enable=true save_preds.label=ndt_stitch-demo

The same command works on TS2 with ``trainer=ndt_stitch_finetune`` under
``src/ts2/train.py`` and ``task=co_smoothing`` or ``task=forecasting`` in place of
``task=wheel_speed``. See :ref:`ts1_guide` and :ref:`ts2_guide` for the full baseline
tables.

TS3
~~~

TS3 evaluates extracted embeddings, which can be produced by the pretrained model in
different ways depending on how the model defines where embeddings come from.
A transductive model like NDT-stitch goes through three steps:

1. **Calibrate** the pretrained checkpoint on one eval recording:

   .. code-block:: bash

      python src/ts3/scripts/calibrate.py trainer=ndt_stitch_calibrate data_root=<data_root> \
          +recording_id=<recording_id> ckpt.load_from=$PWD/ckpt/ndt_stitch.pt

2. **Extract** the calibrated checkpoints and write unit embeddings:

   .. code-block:: bash

      python src/ts3/extract.py extractor=ndt_stitch data_root=<data_root> \
          extractor.pretrain_ckpt=$PWD/ckpt/ndt_stitch.pt \
          extractor.seed=<seed> extractor.ckpt_dir=<calibration_output_dir>

3. **Probe** fit a probe on the embeddings and write the submission:

   .. code-block:: bash

      python src/ts3/eval.py probe=linear data_root=<data_root> emb_path=<embeddings_file> \
          save_preds.enable=true save_preds.label=<submission-id>

The seed for the submission comes from the embeddings file, which ``extract.py``
records; pass ``seed=<seed>`` only for an embeddings file you built some other way. See
:ref:`ts3_guide` for the inductive/transductive distinction and non-transductive models
(NuCLR, NEMO), which skip calibration.

.. _submission-scale:

3. Scale to a full submission
------------------------------

A submission needs at least three seeds, run over every eval recording.

Multiple seeds
~~~~~~~~~~~~~~

``train_seeds.py`` trains one config once per seed, sequentially in the same process,
and prints a mean ± std ± SEM summary at the end:

.. code-block:: bash

   python src/ts1/scripts/train_seeds.py trainer=ndt_stitch_finetune data_root=<data_root> \
       task=wheel_speed recording_id=<recording_id> \
       ckpt.load_from=$PWD/ckpt/ndt_stitch.pt \
       save_preds.enable=true save_preds.label=ndt_stitch-demo

The benchmark's default is five seeds (43-47); override with ``+eval_seeds=[<a>,<b>,<c>]``
for exactly three. We require that the chosen set of seeds are contiguous. ``eval_seeds``
isn't predefined in the base config, so it needs the ``+`` prefix Hydra requires for new
keys. TS2 has the same script at ``src/ts2/scripts/train_seeds.py``. Remember to set
``save_preds.enable=true`` and ``save_preds.label=<submission-id>`` to save the predictions.

Every eval recording
~~~~~~~~~~~~~~~~~~~~~

A submission covers every session in ``src/ibl_bwb_eval/data/eval_recording_ids.txt``. The
scripts in the next section do this for you. To do it by hand, reuse the same helper they call,
:func:`core.sweep.resolve_recording_ids`, rather than reading the file yourself:

.. code-block:: python

   # Save next to train_seeds.py, e.g. src/ts1/scripts/loop_recordings.py, so
   # config_path resolves the same way it does there.
   from copy import deepcopy

   import hydra
   from omegaconf import DictConfig, open_dict

   from core.launch import train
   from core.sweep import resolve_recording_ids


   @hydra.main(version_base="1.2", config_path="../configs", config_name="train.yaml")
   def main(cfg: DictConfig):
       for recording_id in resolve_recording_ids(cfg):
           cfg_r = deepcopy(cfg)
           with open_dict(cfg_r):
               cfg_r.recording_id = recording_id
           train(cfg_r, 0, 1)


   if __name__ == "__main__":
       main()

Leaving ``recording_id`` unset on the command line makes ``resolve_recording_ids`` fall back
to every eval recording. Passing one still works, for a quick single-session check.

Both at once, in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~

``finetune.py`` fans every (recording, task, seed) triple out over `Ray <https://www.ray.io/>`_,
one job per triple, on whichever GPUs and CPUs are visible on the machine you run it from.
Leaving ``task`` unset runs every task in the suite. It also doubles as a hyperparameter sweep
when you set ``sweep.<key>`` to more than one value. Leaving ``sweep`` unset runs the config you give it as-is:

.. code-block:: bash

   python src/ts1/scripts/finetuning/finetune.py trainer=ndt_stitch_finetune data_root=<data_root> \
       ckpt.load_from=$PWD/ckpt/ndt_stitch.pt +eval_seeds=[43,44,45] \
       +ray.gpu=0.25 +ray.cpu=4 \
       save_preds.enable=true save_preds.label=ndt_stitch-demo

``ray.gpu`` and ``ray.cpu`` are fractional resources requested per job (default
``0.125``/``3.5``, packing about eight jobs onto one GPU); raise them if a job needs more
of a GPU to itself. TS2 has the same script at ``src/ts2/scripts/finetuning/finetune.py``.

For TS3, where calibration is required, we can fan out in the same way using Ray. Currently,
TS3 only contains a single task (Cosmos unit-level brain region classification), hence we only
have the (recording, seed) axis.

.. code-block:: bash

   python src/ts3/scripts/calibrate.py trainer=ndt_stitch_calibrate data_root=<data_root> \
       ckpt.load_from=$PWD/ckpt/ndt_stitch.pt +eval_seeds=[43,44,45] \
       +ray.gpu=0.25 +ray.cpu=4

Extract and probe still run once each, over whatever the calibration sweep wrote. For TS3,
there will be one set of predictions for each of single- and multi-unit probes. Unlike TS1 and TS2,
its files also carry no recording-id segment, because its units are scored as one pooled set across sessions.

.. _submission-submit:

4. Submit
---------

Once you have the predictions, zip up the one label you're submitting. The zip's root
must contain ``<task>/`` directly, i.e., zip from inside ``predictions/``, not the parent:

.. code-block:: bash

   cd predictions && zip -r ../ndt_stitch-demo.zip ndt_stitch-demo

Then upload it at `bwb.iblcore.org <https://bwb.iblcore.org>`_, which walks through:

1. **Sign in** using your Google, GitHub, Microsoft, ORCID, or Hugging Face account.
2. **Create or join a team.** A team owns your models and submissions; anyone on it can
   see and edit them.
3. **Create a model** defining a architecture and training recipe, with its metadata:
   links (paper, code, weights), architecture (parameter count, context length), and
   pretraining details (on what modalities, on what data).
4. **Create a submission** under that model: upload the zip, wait for it to validate,
   then declare which tasks it covers. Each task gets its own methodology metadata:
   training paradigm, supervision regime, calibration, finetuning strategy, any extra
   input modality. Submitting then launches scoring automatically and, if made public,
   the leaderboard will update with the new submission.

The site's forms carry the full description of every field. A submission stays editable
while the file uploads and validates, so there's no need to have every answer ready
before you start. For support on the leaderboard website, visit the Issues page for the
`leaderboard repository <https://github.com/int-brain-lab/app-brain-wide-bench/issues>`_.
