.. _setup_guide:

Setup
=====

Setting up is three things: an environment created with ``uv``, a ``.env``
holding the paths and switches every entry point reads, and a data root pointing
at the build you downloaded in the :ref:`dataset quick start <dataset_quickstart>`.

Nothing here needs credentials. The brainsets on S3 are public, and W&B logging is
off until you turn it on.


Requirements
------------

- Python 3.10 or newer.
- `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, which creates
  the environment and installs the dependencies.
- A CUDA GPU for anything past the smallest baselines.


.. _setup_install:

Installing
----------

The training code is meant to be edited, not imported from a Python library (at
least for now), so you work inside the clone and install it editable. To version
and push your own progress, fork the repository first and clone that instead. Make
sure `uv <https://docs.astral.sh/uv/getting-started/installation/>`__ is installed.

.. code-block:: bash

   # or your own fork: git clone https://github.com/<you>/ibl-bwb.git
   git clone https://github.com/brainbench-org/ibl-bwb.git
   cd ibl-bwb

   uv venv .venv -p python3.10
   source .venv/bin/activate
   uv pip install -e ".[train]"


.. _setup_env_file:

Configuring .env
----------------

Environment variables and paths for the benchmark are set in a `.env
<https://github.com/theskumar/python-dotenv#getting-started>`__ at the repo root,
which is gitignored. Edit it and the change applies on the next run: every
training entry point loads the file, so there is nothing to re-source.

Copy the annotated template:

.. code-block:: bash

   cp .env.example .env

Every variable in it starts commented out, so uncomment and fill only the ones you
need.

The ones you are most likely to edit:

.. code-block:: bash
   :caption: .env

   BWB_CKPT_DIR=./ckpt                # checkpoints, and what a relative ckpt.load_from resolves against
   BWB_PREDICTIONS_DIR=./predictions  # saved predictions, for scoring
   BWB_EMBS_DIR=./embs                # TS3 unit embeddings
   BWB_CACHE_DIR=./cache              # TS3 waveform/ACG cache, built on first use
   WANDB_MODE=disabled                # online, offline or disabled (the default)

The four directories take relative paths, resolved against wherever you launch
from, and the values above are what you get if you leave them unset. Set them if
you launch from outside the repo root.

Model settings such as hyperparameters live elsewhere, in the `Hydra
<https://hydra.cc/docs/intro/>`__ ``.yaml`` configs (`Where the code lives`_).
There you will often see the pattern ``${oc.env:VAR,default}``, which is what
pulls a variable out of this file: the command line beats ``.env``, which beats
the default. Leave a variable commented out rather than empty, since an exported
empty string beats the default.


.. _setup_data_root:
.. _dataset_data_root:

Setting the data root
---------------------

``data_root`` is where a build you downloaded (:ref:`dataset quick start
<dataset_quickstart>`) lives: everything above the brainset directory itself.

.. code-block:: text

   /abs/path/to/data/all_units/ibl_brain_wide_bench_2026/   <- the build you downloaded
   /abs/path/to/data/all_units                              <- data_root

The different builds carry that same directory name ``ibl_brain_wide_bench_2026``,
so they cannot share a root.

Set the ones you have downloaded in your ``.env``, as absolute paths:

.. code-block:: bash
   :caption: .env

   BWB_DATA_ROOT_ALL_UNITS=/abs/path/to/data/all_units            # TS1
   BWB_DATA_ROOT_SELECTED_UNITS=/abs/path/to/data/selected_units  # TS2, TS3
   BWB_DATA_ROOT_PRETRAIN=/abs/path/to/data/all_units             # pretraining, either build

Each falls back to ``BWB_DATA_ROOT`` when unset, so one root covers a single
build. The exception is ``trainer=nuclr_pretrain`` and ``nemo_pretrain``: they
score TS3 as they train, so they read ``BWB_DATA_ROOT_SELECTED_UNITS`` rather
than ``BWB_DATA_ROOT_PRETRAIN``, and need their pretraining sessions under the
``selected_units`` root next to the eval ones.


.. _codebase_guide:

Where the code lives
--------------------

The benchmark codebase is organized as follows, colored by role:

.. raw:: html

   <p><span class="bwb-slate bwb-b">shared machinery</span> &nbsp;
   <span class="bwb-blue bwb-b">pretraining</span> &nbsp;
   <span class="bwb-pink bwb-b">the three evaluation suites</span> &nbsp;
   <span class="bwb-violet bwb-b">the scoring contract</span></p>

.. raw:: html

   <pre class="bwb-tree">
   <span class="bwb-ink bwb-b">src/</span>
   <span class="bwb-mute">├── </span><span class="bwb-slate bwb-b">core/</span>           <span class="bwb-mute"># shared machinery: trainer, datasets, samplers, checkpointing</span>
   <span class="bwb-mute">├── </span><span class="bwb-blue bwb-b">pretrain/</span>       <span class="bwb-mute"># pretraining: train.py and one directory per model</span>
   <span class="bwb-mute">├── </span><span class="bwb-pink bwb-b">ts1/</span>            <span class="bwb-mute"># decoding behavior</span>
   <span class="bwb-mute">├── </span><span class="bwb-pink bwb-b">ts2/</span>            <span class="bwb-mute"># neural activity</span>
   <span class="bwb-mute">├── </span><span class="bwb-pink bwb-b">ts3/</span>            <span class="bwb-mute"># brain region</span>
   <span class="bwb-mute">├── </span><span class="bwb-violet bwb-b">ibl_bwb_eval/</span>   <span class="bwb-mute"># the published contract: tasks, metrics, scoring</span>
   <span class="bwb-mute">├── </span><span class="bwb-slate">hydra_plugins/</span>  <span class="bwb-mute"># config auto-discovery</span>
   <span class="bwb-mute">└── </span><span class="bwb-slate">tests/</span>
   </pre>

Take NDT Stitch as an example: pretrained and then evaluated, it shows how a model
is organized in the codebase.

.. raw:: html

   <pre class="bwb-tree">
   <span class="bwb-blue bwb-b">src/pretrain/models/ndt_stitch/</span>               <span class="bwb-mute"># pretraining</span>
   <span class="bwb-mute">├── </span><span class="bwb-blue">ndt_stitch.py</span>                             <span class="bwb-mute"># the model class</span>
   <span class="bwb-mute">├── </span><span class="bwb-blue">ndt_stitch_pretrain.py</span>                    <span class="bwb-mute"># its trainer</span>
   <span class="bwb-mute">└── </span><span class="bwb-blue">configs/</span>
   <span class="bwb-mute">    ├── </span><span class="bwb-blue">model/ndt_stitch_10M.yaml</span>             <span class="bwb-mute"># reused by every suite</span>
   <span class="bwb-mute">    └── </span><span class="bwb-blue">trainer/ndt_stitch_pretrain.yaml</span>
   
   <span class="bwb-pink bwb-b">src/ts1/models/pretrained/ndt_stitch/</span>         <span class="bwb-mute"># the same model, on TS1</span>
   <span class="bwb-mute">├── </span><span class="bwb-pink">ndt_stitch_eval_trainer.py</span>                <span class="bwb-mute"># a trainer only</span>
   <span class="bwb-mute">└── </span><span class="bwb-pink">configs/trainer/ndt_stitch_finetune.yaml</span>
   </pre>

Each entry point is a ``train.py`` you call with Hydra overrides.

.. raw:: html

   <div class="highlight"><pre class="bwb-tree">
   <span class="bwb-ink">python </span><span class="bwb-blue bwb-b">src/pretrain</span><span class="bwb-ink">/train.py trainer=ndt_stitch_pretrain</span>
   </pre></div>

.. raw:: html

   <ul style="line-height:1.8">
   <li><code class="bwb-tree-chip">trainer=ndt_stitch_pretrain</code> names the trainer config
   <code class="bwb-tree-chip"><span class="bwb-blue">pretrain/models/ndt_stitch/</span>configs/trainer/ndt_stitch_pretrain.yaml</code>.</li>
   <li>That file asks for <code class="bwb-tree-chip">/model: ndt_stitch_10M</code>, which is the model config
   <code class="bwb-tree-chip"><span class="bwb-blue">pretrain/models/ndt_stitch/</span>configs/model/ndt_stitch_10M.yaml</code>.</li>
   </ul>

.. raw:: html

   <div class="highlight"><pre class="bwb-tree">
   <span class="bwb-ink">python </span><span class="bwb-pink bwb-b">src/ts1</span><span class="bwb-ink">/train.py trainer=ndt_stitch_finetune ckpt.load_from=&lt;checkpoint&gt;.pt</span>
   </pre></div>

.. raw:: html

   <ul style="line-height:1.8">
   <li>TS1 works the same way, from
   <code class="bwb-tree-chip"><span class="bwb-pink">ts1/models/pretrained/ndt_stitch/</span>configs/trainer/ndt_stitch_finetune.yaml</code>,
   except that its <code class="bwb-tree-chip">/model</code> is still the pretraining one: a suite can use the pretraining model config, and adds only a trainer.</li>
   </ul>
   <p>Configs are found by name, wherever they sit, because
   <code class="bwb-tree-chip"><span class="bwb-slate">src/hydra_plugins/</span></code> discovers every
   <code class="bwb-tree-chip">configs/</code> in the tree. That is why adding a model edits no central file.</p>

Three things to know before editing:

- **Every model is one directory.** As in the NDT Stitch example above, the model
  class, its trainer and their configs sit together, and nothing outside that
  directory knows about them.
- **The task suites are self-contained, on purpose.** ``ts1/``, ``ts2/`` and ``ts3/``
  repeat each other so that someone who cares about one suite can read one directory.
- **Some parts are fixed, so that two submissions stay comparable.** ``ibl_bwb_eval/``
  holds the shared contract: tasks, metrics, evaluation seeds and the prediction
  format. Do not edit it, nor the code that evaluates the held-out test split: the
  suites' test mixins, and ``ts3/protocol.py``, the unit table every TS3 metric is
  computed against.


Weights & Biases
----------------

`Weights & Biases <https://docs.wandb.ai/quickstart/>`__ tracks a run's metrics and
config in a dashboard you can compare runs in.

It is off until you turn it on in the ``.env``, and every run logs from then on.

.. code-block:: bash
   :caption: .env

   WANDB_MODE=online         # online, offline or disabled (the default, records nothing)
   WANDB_API_KEY=<your key>  # online needs it, from https://wandb.ai/authorize
   WANDB_ENTITY=<team>       # default: your own entity
   WANDB_DIR=wandb/          # where offline keeps runs

``offline`` is the middle ground if you want your own metrics but no account: it
keeps the full run locally, ready to push later with ``wandb sync
wandb/offline-run-*``.

The project a run logs to is set in the configs, named after the entry point and
the model: ``pretrain-ndt2``, ``ts1-poyo``, ``ts2-lfads``, ``ts3-lolcat``.


Checking the install
--------------------

One short run exercises the whole path, from the loader to the metrics:

.. code-block:: bash

   python src/ts1/train.py trainer=mlp task=choice debug=true

``debug=true`` cuts the run to one batch per split and one epoch, with W&B,
checkpointing and dataloader workers off.


Optional extras
---------------

**xformers, for POYO+.** POYO+ attends over chained tokens with a stock-torch
kernel by default (``model.attn_impl=nested``). ``xformers`` provides a leaner
one -- chiefly a much lower peak memory, so a larger batch fits on one GPU, plus
a smaller speedup -- but pins a wheel to an exact torch build, so it is kept out
of ``train``:

.. code-block:: bash

   uv pip install -e ".[train,xformers]"

Then pass ``model.attn_impl=xformers``. Both backends compute the same attention,
so checkpoints move between them freely.

**Ray, for tuning.** ``tune.py`` and ``tune.sh`` need a running cluster. Start a
local one with ``ray start --head``, or point ``RAY_ADDRESS`` at an existing one.
``BWB_RAY_RESULTS_DIR`` and ``BWB_RAY_OUTPUTS_DIR`` hold trial storage and result
CSVs, and the first has to be on a shared filesystem for a multi-node cluster.
``RAY_TMPDIR`` moves Ray's scratch space off ``/tmp`` when that is small.
