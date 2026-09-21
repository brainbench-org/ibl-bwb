.. _ts3_guide:

Task Suite 3: Brain Region Prediction
=======================================

TS3 evaluates neuron-level representations by predicting the Cosmos-level
anatomical region (10 classes, Allen CCF) of individual units. Regions span
coarse but functionally meaningful areas such as Isocortex, Hippocampus, and
Cerebellum. Both single-unit and multi-unit settings are covered.

All evaluations are zero-shot on held-out animals. Two regimes are included:
**transductive zero-shot** (adaptation via non-region supervision is allowed)
and **inductive zero-shot** (no adaptation). Performance is reported as
macro-averaged F1 to account for class imbalance.

.. image:: ../_static/ts3_overview.svg
   :class: dark-light
   :width: 100%
   :alt: TS3 overview

.. contents:: On this page
   :local:
   :depth: 2

Overview of TS3
---------------

.. note::

   "Probe" carries two meanings in this guide. A **recording probe** is the Neuropixels
   insertion a unit sits on: what ``probe_qc`` filters and what the multi-unit readout
   pools within. A **probe** on its own, as in ``probe=linear``, is the classifier
   ``ts3/eval.py`` fits on frozen embeddings to predict the brain region.

Every unit is assigned one of ten Cosmos-level regions, and a submission is scored twice:

.. list-table::
   :widths: 20 58 22
   :header-rows: 1

   * - Setting
     - What is scored
     - Primary metric
   * - Single-unit
     - the probabilities a model gives each unit on its own
     - Macro F1
   * - Multi-unit
     - the same probabilities averaged over a unit's five nearest neighbours on the same
       recording probe within 60 µm, itself included
     - Macro F1

The ten classes are ``CB``, ``CNU``, ``CTXsp``, ``HB``, ``HPF``, ``HY``, ``Isocortex``,
``MB``, ``OLF`` and ``TH``. They are imbalanced, which is why the metric is
macro-averaged rather than a plain accuracy.

Evaluation pipeline
~~~~~~~~~~~~~~~~~~~

The probe is fit on the pretrain units and their region labels, and scored on the eval
units, which come from animals held out of pretraining. The eval labels are only ever
scored against, never fit to.

Which regime a model belongs to is a property of the model, not a setting you pick:

- **Inductive.** The unit's embedding is a function of the unit's own data, so one encoder
  answers for pretrain and eval units alike. NuCLR, NEMO and the ISI baseline.
- **Transductive.** The embedding is a free parameter that exists only for units a run has
  seen, so the eval units are answered by per-session checkpoints adapted on non-region
  supervision. NDT Stitch and POYO.

.. _ts3_unit_qc:


Two layers of unit QC
~~~~~~~~~~~~~~~~~~~~~

Unit QC is applied at two independent points, and the two do not enforce the same
rule. A unit count that looks wrong is usually the two being read as one.

**Build time** bakes the three filters from :ref:`dataset_builds` into the
``.h5``. It defines the stored population, and it is what ``unit_filtering`` records.

**Load time** is applied by the consuming task suite on top of whatever the build
contains. It is task specific and is never recorded in the file.

Mostly, load time just re-verifies the build. These three criteria are checked at
both layers with identical thresholds, so they never change the population:

.. list-table::
   :widths: 34 33 33
   :header-rows: 1

   * - Criterion
     - Build time
     - Re-applied by TS3 scoring
   * - ``qc_neural`` (recording probe QC)
     - ``== PASS``, eval only
     - ``== PASS`` on eval
   * - ``firing_rate``
     - ``> 1.0`` Hz
     - ``> 1.0`` Hz
   * - ``unit_qc`` label
     - ``== 1.0``
     - ``== 1.0``

Only three criteria are genuinely load time. No build applies them, so no
``--unit-filter`` value can satisfy them:

.. list-table::
   :widths: 28 30 42
   :header-rows: 1

   * - Criterion
     - Rule
     - Applied by
   * - ``qc_neural`` on pretrain
     - ``!= FAIL``, so ``WARNING`` is kept
     - the TS3 dataset mask and scoring, identically
   * - ``qc_neural_alignment``
     - ``== PASS``
     - TS3 scoring
   * - Brain region
     - not ``void`` or ``root``
     - TS3 scoring

Two consequences are worth stating plainly:

* Because ``probe_qc`` is not applied to pretrain sessions, a pretrain build
  labeled ``selected_units`` still contains units on recording probes whose
  ``qc_neural`` is ``WARNING`` or ``FAIL``. TS3 masks the ``FAIL`` ones at load time and
  trains on the ``WARNING`` ones, which are about 30% of pretrain recording probes. TS3
  also drops the 32 pretrain sessions whose every recording probe is ``FAIL``.
* Because the last two criteria are invisible to the build, ``selected_units`` does
  not mean "the population TS3 scores on"; that population is pinned by
  ``_EXPECTED_MD_SIZE`` in ``src/ts3/protocol.py``.

How to use the TS3 benchmark
----------------------------

TS3 scores unit embeddings rather than a trainer, so what you supply depends on the shape
of your model. There are two ways to plug in:

1. **Supply an Extractor**: turn your model into one embedding per unit, let
   ``ts3/extract.py`` write them and ``ts3/eval.py`` fit and score the probe. This is the
   path for anything pretrained.
2. **Bring your own trainer**: a model supervised on the region labels skips embeddings and
   the probe entirely, and reports F1 from its own run, as LOLCAT does.

Key classes
~~~~~~~~~~~

- :class:`~ts3.IBLBrainWideBenchTS3`: whole sessions after the suite's unit QC, which is
  fixed rather than a parameter.
- :class:`~ts3.Extractor`: the interface an embedding model implements, and what
  ``extract.py`` drives.
- :class:`~ts3.models.transductive.TransductiveExtractor`: base for extractors whose
  embeddings exist only for units a run has seen.
- :class:`~ts3.probes.Probe`, with :class:`~ts3.probes.LinearProbe` and
  :class:`~ts3.probes.MLPProbe`: the heads ``eval.py`` fits on frozen embeddings.
- :class:`~ts3.models.supervised.LOLCATTrainer`: the worked example of a model that skips
  the probe and scores from its own run.

Model families
~~~~~~~~~~~~~~

NuCLR and NEMO pretrain from ``src/pretrain/train.py``, alongside the TS1 and
TS2 pretraining models; LOLCAT and the ISI baseline stay in ``src/ts3``. Turning any
of them into unit embeddings is one step, ``src/ts3/extract.py``, and which extractor
it runs is grouped by regime under ``src/ts3/models/``: ``inductive/`` for the models
whose unit embedding is a function of the unit's own data, ``transductive/`` for those
whose unit embedding is a free parameter that exists only for units a run has seen.

**NuCLR** learns a shared embedding space from unit features using contrastive
learning. Pretraining is handled by :class:`~pretrain.models.NuCLRPretrain`; the
loss is :class:`~pretrain.models.NuCLRLoss`.

**NEMO** fuses waveform and autocorrelogram modalities via a CLIP-style objective.
:class:`~pretrain.models.WVFEncoder` and :class:`~pretrain.models.ACGEncoder`
produce per-modality embeddings that are aligned by ``core.nn.loss.CLIPLoss``
through a :class:`~pretrain.models.LinearProjector`.

**LOLCAT** is supervised rather than pretrained. An MLP embeds each of a unit's
per-trial ISI histograms, :class:`~ts3.models.supervised.MultiHeadGlobalAttention` pools the trials
into one unit-level vector, and a linear head predicts the region, so it reports
F1 directly instead of going through a probe.
:class:`~ts3.models.supervised.LossFeedbackSampler` retunes the per-class oversampling factors
between epochs from the train/val loss gap.

**ISI histograms** is a training-free baseline. ``extractor=isi`` writes one
log-spaced, L1-normalised :func:`~ts3.compute_isi_histogram` per unit and uses it
directly as the embedding.

How do I evaluate my model on TS3?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extractor interface
^^^^^^^^^^^^^^^^^^^

The sampled-window flow of :ref:`pretraining_sample_flow` does not apply here. ``extract.py``
builds one extractor and asks it for the pretrain units, then the eval units. The two halves
of the file are comparable only because the same object answered both calls, so the regime is
an argument rather than a field:

.. code-block:: python

   from core.dataset import BenchmarkRegime
   from ts3 import Extractor

   class MyExtractor(Extractor):
       @property
       def name(self) -> str:
           """Names the embeddings file, so two runs do not overwrite each other."""

       @property
       def run_seed(self) -> int | None:
           """The seed behind these embeddings, which a submission is filed under."""

       def encode(self, regime: BenchmarkRegime):
           """Return (N, D) embeddings and the (N,) uids they belong to."""

``setup`` binds the run's data root, device and logger, and is where a checkpoint is loaded
once rather than per regime. Units your extractor cannot represent may be left out: the
probe joins on uids and drops anything TS3 does not score, but insists the eval side is
complete.

Which checkpoint answers the eval call is the entire inductive/transductive difference, and
it is settled inside ``encode``, not by the caller.

Writing embeddings
^^^^^^^^^^^^^^^^^^

``extract.py`` writes one ``.pt`` per run into ``embs_dir``, named after the extractor
unless ``name=`` overrides it. The file holds both regimes' embeddings with their uids, and
the ``run_seed`` the extractor reported. Naming by run keeps several epochs of one model, or
several finetuning seeds, from overwriting each other.

Probing
^^^^^^^

After pretraining, a probe is fit on top of frozen embeddings by
``ts3/eval.py`` with ``probe=linear`` or ``probe=mlp``, and scored with macro-averaged
F1. LOLCAT skips this step. NuCLR and Nemo also probe themselves while pretraining, but
that monitor belongs to each of them (``pretrain/models/*/monitor.py``) and reads
pretrain units only, so it never touches the units scored here.

.. _ts3_released_embeddings:

The released NuCLR and NEMO repositories ship the unit embeddings each of their
checkpoints produced, as ``<model>_embeddings_<seed>.pt``, so
``ts3/eval.py emb_path=<FILE>`` probes them directly and ``extract.py`` need not run
again. :ref:`pretrain_released_ckpts` covers the download.

Bringing your own trainer
^^^^^^^^^^^^^^^^^^^^^^^^^

A model supervised on the region labels has no embedding step and no probe. It trains and
scores in one call through ``src/ts3/train.py``, reports macro F1 directly, and writes its
own submission. :class:`~ts3.models.supervised.LOLCATTrainer` is the worked example, and a
trainer of your own owes what :ref:`pretraining <pretraining_guide>` describes.

What is fixed
^^^^^^^^^^^^^

.. warning::

   ``src/ts3/protocol.py`` holds the unit table every TS3 metric is computed against, and
   ``_EXPECTED_MD_SIZE`` pins its size: 3,222 eval units and 41,834 pretrain units for
   ``unit_cosmos``. ``ts3/eval.py`` is the only scoring path, fitting the probe on pretrain
   labels and scoring eval ones, and both regimes must come from the ``selected_units``
   build. None of that is yours to change. The extractor, the embeddings it writes and which
   probe scores them are.

Launching training and evaluation
---------------------------------

TS3 scores a probe on frozen unit embeddings, so most baselines are two calls, both driven
by `Hydra <https://hydra.cc>`_: write the embeddings with ``src/ts3/extract.py``, then fit
and score the probe with ``src/ts3/eval.py``.

An inductive extractor reads one pretrain checkpoint (:ref:`pretrain_released_ckpts`) and
encodes the pretrain and eval units with it:

.. code-block:: bash

   python src/ts3/extract.py extractor=nuclr data_root=<data_root> extractor.ckpt=/path/to/ckpt.pt

``extractor=isi`` is training free and takes no checkpoint at all.

A transductive extractor requires three calls rather than two, because it also needs a
calibration checkpoint for every eval recording. Calibration continues the model's own
pretraining objective on one eval recording, initialized from the shared pretrain
checkpoint, and writes a session-local ``best.pt`` whose unit table or stitcher becomes
the eval part of the embeddings file.

Run one calibration per (recording, seed), **into a directory used by that sweep and
nothing else**. ``src/ts3/scripts/calibrate.py`` does the whole grid as one Ray sweep,
every eval recording by every seed:

.. code-block:: bash

   python src/ts3/scripts/calibrate.py trainer=poyo_plus_calibrate \
       ckpt.load_from=/path/to/pretrain.pt ckpt.dir=/path/to/calibrations \
       +ray.gpu=0.125 +ray.cpu=3

``src/ts3/train.py`` runs a single one instead, for a single recording:

.. code-block:: bash

   python src/ts3/train.py trainer=poyo_plus_calibrate \
       ckpt.load_from=/path/to/pretrain.pt ckpt.dir=/path/to/calibrations \
       recording_ids=[<eid>]

Extraction then reads the pretrain checkpoint for the pretrain part and **scans**
``extractor.ckpt_dir`` for the eval part, keeping every ``best.pt`` whose saved config
names one recording, that seed, and the same model:

.. code-block:: bash

   python src/ts3/extract.py extractor=poyo_plus \
       extractor.pretrain_ckpt=/path/to/pretrain.pt extractor.seed=<seed> \
       extractor.ckpt_dir=/path/to/calibrations

.. warning::

   ``extractor.ckpt_dir`` scans the ckpt tree recursively. The scan keeps any ``best.pt`` whose
   config names one recording, a seed, and the same model. Pointing extraction at a shared root
   therefore picks up runs you did not mean. Two matches for one (seed, recording) raises an error,
   and a missing one fails the scan.

   Therefore, ``ckpt_dir`` is mandatory (no defualt), and calibration for a given model
   should be given a fresh directory. For example::

      $BWB_CKPT_DIR/ts3-calibrations/poyo_plus/

   Pass that one path as ``ckpt.dir`` when calibrating and as ``extractor.ckpt_dir`` when
   extracting. ``filters`` narrows the scan on any config field if checkpoints must share
   a directory.

The other transductive models follow the same three-step pattern: swap ``poyo_plus`` for
``possm``, ``ndt_stitch``, or ``mtm`` in every command.

Either way the embeddings land in ``embs_dir`` (``BWB_EMBS_DIR``, default ``embs/``). Pass
that file to the probe, which reports macro F1 single-unit and multi-unit:

.. code-block:: bash

   python src/ts3/eval.py probe=linear data_root=<data_root> emb_path=/path/to/embeddings.pt

The linear probe is CPU; ``probe=mlp`` searches ``probe.num_trials`` configurations on a GPU
instead. LOLCAT is supervised on the region labels and has no embedding step, so it trains and runs
the test protocol in one call, like a TS1 or TS2 baseline:

.. code-block:: bash

   python src/ts3/train.py trainer=lolcat data_root=<data_root>

``save_preds`` writes the prediction file the `leaderboard <https://bwb.iblcore.org>`_ scores. For an embedding
model it comes from the probe:

.. code-block:: bash

   python src/ts3/eval.py probe=linear emb_path=/path/to/embeddings.pt \
       save_preds.enable=true save_preds.label=<submission-id>

The submission is filed under the seed ``extract.py`` recorded in the file; add ``seed=``
only for an embeddings file you built yourself, which records none. LOLCAT writes from its
own run instead, with the same two flags on ``src/ts3/train.py``.

Key config options
~~~~~~~~~~~~~~~~~~

Defaults live in ``src/ts3/configs/extract.yaml`` and ``src/ts3/configs/eval.yaml``.

**Extraction.** ``name`` overrides the file name, which otherwise comes from the extractor,
and ``embs_dir`` says where it lands (``BWB_EMBS_DIR`` in your :ref:`.env <setup_env_file>`,
default ``embs/``). A transductive extractor also reads ``extractor.pretrain_ckpt``, the
shared checkpoint behind the pretrain part, ``extractor.seed``, which selects the
calibration run, and ``extractor.ckpt_dir``, the sweep directory scanned for those
calibrations. All three are mandatory: there is no sensible default for which sweep to
read.

**Probing.** ``task`` is ``unit_cosmos``, the only one defined today. ``probe.num_trials``
sets how many configurations ``probe=mlp`` searches. ``seed`` names the submission rather
than seeding anything, and is needed only for an embeddings file that records none.

Both entry points read ``data_root`` from ``BWB_DATA_ROOT_SELECTED_UNITS``: TS3 refuses any
other build.

Reference baselines
--------------------

The training runs behind these baselines are accessible on `W&B
<https://wandb.ai/ibl-bwb/projects>`_, linked from each row below. TS3 scores a probe
rather than the encoder, so a project holds the probe runs behind the reported numbers,
not the pretraining they read from.

Inductive
~~~~~~~~~

The unit's embedding is a function of its own data, so one encoder answers for both
regimes. Run with ``src/ts3/extract.py``, then score with ``src/ts3/eval.py``.

.. list-table::
   :widths: 16 26 50 8
   :header-rows: 1

   * - Model
     - Config
     - Extractor class
     - W&B
   * - ISI histograms
     - ``extractor=isi``
     - :class:`~ts3.models.inductive.ISIExtractor`, training-free
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-isi-probe
          :alt: W&B project ts3-isi-probe
          :width: 22px
   * - `NEMO <https://openreview.net/forum?id=10JOlFIPjt>`__
     - ``extractor=nemo``
     - :class:`~ts3.models.inductive.NEMOExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-nemo-probe
          :alt: W&B project ts3-nemo-probe
          :width: 22px
   * - `NuCLR <https://openreview.net/forum?id=zt3RKc6VBp>`__
     - ``extractor=nuclr``
     - :class:`~ts3.models.inductive.NuCLRExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-nuclr-probe
          :alt: W&B project ts3-nuclr-probe
          :width: 22px

Transductive
~~~~~~~~~~~~

The embedding exists only for units a run has seen, so each also needs the per-session
checkpoints in ``extractor.ckpt_dir`` for its seed. Calibration happens first
(:ref:`submission_guide`), then extraction reads its checkpoints; each model lists both
stages as separate rows below.

.. list-table::
   :widths: 14 12 20 34 20
   :header-rows: 1

   * - Model
     - Stage
     - Config
     - Class
     - W&B
   * - `NDT Stitch <https://www.biorxiv.org/content/early/2021/07/23/2021.01.16.426955>`__
     - Calibrate
     - ``trainer=ndt_stitch_calibrate``
     - :class:`~ts3.models.transductive.NDTStitchCalibrateTrainer`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-ndt_stitch-calibrate
          :alt: W&B project ts3-ndt_stitch-calibrate
          :width: 22px
   * -
     - Probe
     - ``extractor=ndt_stitch``
     - :class:`~ts3.models.transductive.NDTStitchExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-ndt_stitch-probe
          :alt: W&B project ts3-ndt_stitch-probe
          :width: 22px
   * - MtM
     - Calibrate
     - ``trainer=mtm_calibrate``
     - :class:`~ts3.models.transductive.MtMCalibrateTrainer`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-mtm-calibrate
          :alt: W&B project ts3-mtm-calibrate
          :width: 22px
   * -
     - Probe
     - ``extractor=mtm``
     - :class:`~ts3.models.transductive.MtMExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-mtm-probe
          :alt: W&B project ts3-mtm-probe
          :width: 22px
   * - `POYO+ <https://openreview.net/forum?id=IuU0wcO0mo>`__
     - Calibrate
     - ``trainer=poyo_plus_calibrate``
     - :class:`~ts3.models.transductive.POYOPlusCalibrateTrainer`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-poyo_plus-calibrate
          :alt: W&B project ts3-poyo_plus-calibrate
          :width: 22px
   * -
     - Probe
     - ``extractor=poyo_plus``
     - :class:`~ts3.models.transductive.POYOPlusExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-poyo_plus-probe
          :alt: W&B project ts3-poyo_plus-probe
          :width: 22px
   * - POSSM
     - Calibrate
     - ``trainer=possm_calibrate``
     - :class:`~ts3.models.transductive.POSSMCalibrateTrainer`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-possm-calibrate
          :alt: W&B project ts3-possm-calibrate
          :width: 22px
   * -
     - Probe
     - ``extractor=possm``
     - :class:`~ts3.models.transductive.POSSMExtractor`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-possm-probe
          :alt: W&B project ts3-possm-probe
          :width: 22px

Supervised
~~~~~~~~~~

Trained on the region labels and scored without a probe, in one call to
``src/ts3/train.py``.

.. list-table::
   :widths: 16 26 50 8
   :header-rows: 1

   * - Model
     - Config
     - Trainer class
     - W&B
   * - `LOLCAT <https://doi.org/10.1016/j.celrep.2023.112318>`__
     - ``trainer=lolcat``
     - :class:`~ts3.models.supervised.LOLCATTrainer`
     - .. image:: https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg
          :class: dark-light
          :target: https://wandb.ai/ibl-bwb/ts3-lolcat-supervised
          :alt: W&B project ts3-lolcat-supervised
          :width: 22px
