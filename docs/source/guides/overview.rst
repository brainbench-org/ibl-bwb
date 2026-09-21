.. _codebase_overview:

.. role:: raw-html(raw)
   :format: html

Codebase overview
=================

The repository is built from six pieces, each owning one job. One is fixed by the
benchmark, so that a reported number means the same thing in every submission: the
evaluation contract, and the standardized test path that reads it. The rest are yours to
write or extend. This page says which piece is responsible for what, so you know where to
look before reading :ref:`pretraining_guide`, :ref:`ts1_guide`, :ref:`ts2_guide` or
:ref:`ts3_guide`.

The six map onto ``src/`` like this:

.. raw:: html

   <pre class="bwb-tree">
   <span class="bwb-ink bwb-b">src/</span>
   <span class="bwb-mute">├── </span><span class="bwb-orange bwb-b">ibl_bwb_eval/</span>    <span class="bwb-mute"># the evaluation contract: tasks, metrics, protocol, submission format</span>
   <span class="bwb-mute">├── </span><span class="bwb-ink bwb-b">core/</span>            <span class="bwb-mute"># shared machinery, no suite of its own</span>
   <span class="bwb-mute">│   ├── </span><span class="bwb-emerald">dataset.py</span>   <span class="bwb-mute"># IBLBrainWideBench2026, the base every dataset extends</span>
   <span class="bwb-mute">│   ├── </span><span class="bwb-pink">trainer.py</span>   <span class="bwb-mute"># BaseTrainer: loop, logging, checkpointing, DDP, patience</span>
   <span class="bwb-mute">│   ├── </span><span class="bwb-blue">model.py</span>     <span class="bwb-mute"># BaseModel, the model interface</span>
   <span class="bwb-mute">│   └── </span><span class="bwb-slate">transforms/, samplers/, nn/, finetuning/, ...</span>
   <span class="bwb-mute">├── </span><span class="bwb-ink bwb-b">pretrain/</span>        <span class="bwb-mute"># one directory per model, each owning its trainer</span>
   <span class="bwb-mute">├── </span><span class="bwb-ink bwb-b">ts1/</span>             <span class="bwb-mute"># dataset, task trainer, test mixin, model trainers</span>
   <span class="bwb-mute">├── </span><span class="bwb-ink bwb-b">ts2/</span>             <span class="bwb-mute"># same layout as ts1</span>
   <span class="bwb-mute">└── </span><span class="bwb-ink bwb-b">ts3/</span>             <span class="bwb-mute"># extractors and probes instead of a task trainer</span>
   </pre>

Who owns what
-------------

:raw-html:`<strong class="bwb-orange">Evaluation contract</strong>` (:mod:`ibl_bwb_eval`).
**Fixed**, and imported by every other piece while importing none of them.

- **the eval protocol**: the eval sessions, the eval seeds, the selection metric.
- **the task vocabulary** of each suite, and **one readout spec** per task.
- **the submission format**.

:raw-html:`<strong class="bwb-emerald">Dataset</strong>`
(:class:`~core.dataset.IBLBrainWideBench2026`, one subclass per suite, instantiated per
task, split and session).

- **which recordings and which units** the ``regime`` (pretrain or eval) allows.
- **the splits** (train, val, and test), selected by ``split=``.
- **the intervals a sampler may draw from**, ``get_sampling_intervals``.
- **the evaluation target**, ``_get_target``, taken before any augmentation to avoid
  leakage.
- **the conditioning every sample gets**, ``dataset_transform``, and **the order the rest
  run in**, ``__getitem__``. Both are in `How a sample reaches the model`_.

TS3 is the exception: it scores units rather than windows, so it serves whole sessions with
no split, no task and no target.

:raw-html:`<strong class="bwb-pink">Base Trainer</strong>`
(:class:`~core.trainer.BaseTrainer`). What depends on neither the task nor the model. Every
trainer subclasses it, in all three suites and in pretraining.

- **the epoch loop**, ``train()``: when val runs, when test runs, when to stop.
- **everything around it**: logging, checkpointing, DDP, early stopping.

:raw-html:`<strong class="bwb-pink">Task trainers</strong>` (TS1 and TS2, one each:
:class:`~ts1.TS1EvalTrainer`, :class:`~ts2.TS2EvalTrainer`).

- **the task, its readout spec, and the loss** that matches it.
- **the loaders, the optimizer and the finetuning strategy**.
- **the epoch internals**: ``train_epoch``, ``val_epoch``, ``loss`` and ``predict``.
- **the wiring between model and dataset**, ``link_model``, the one place the two meet.

Testing is not theirs: :class:`~ts1.TS1TestMixin` and :class:`~ts2.TS2TestMixin` fix the
test split, the intervals, the metrics and the prediction file, so every submission is
scored the same way.

:raw-html:`<strong class="bwb-blue">Model</strong>` (:class:`~core.model.BaseModel`, in its
own directory alongside its trainer and configs).

- **the architecture** and its ``forward``.
- **how a sample becomes model inputs**, ``input_fn``: one ``Data`` sample to the tensors
  this architecture expects.
- **whatever has to be built from the dataset**, ``link_datasets``: a unit or session
  vocabulary, say.
- **the readout head**, ``configure_readout``, matching the task's readout spec, which is
  what makes one architecture evaluable on several tasks.
- **how pretrained weights land in it**, ``load_ckpt``.

:raw-html:`<strong class="bwb-pink">Model trainers</strong>` (next to the model they belong
to). Where a model states what it needs differently, and nothing else.

- **on the eval side**, a subclass of the suite's trainer, overriding only what differs.
- **on the pretrain side**, the objective itself, subclassing
  :class:`~core.trainer.BaseTrainer` directly. Its model code still satisfies the interface
  above, so the same class can be evaluated downstream by a suite's trainer.

.. _pretraining_sample_flow:

How a sample reaches the model
------------------------------

Every suite and every pretraining run reads the benchmark through one base dataset class,
:class:`~core.dataset.IBLBrainWideBench2026`, and each sample it hands the loader goes
through the same steps, in the order ``__getitem__`` fixes. Two of them are extension
points: ``dataset_transform``, owned by the dataset class, and ``transform``, supplied from
the config.

.. raw:: html

   <pre class="bwb-tree">
   <span class="bwb-mute">  sampler picks a recording and a window</span>
   <span class="bwb-mute">        │</span>
   <span class="bwb-mute">        ▼</span>
   <span class="bwb-emerald bwb-b">  data.slice(start, end)</span>          <span class="bwb-mute"># the window, straight off the .h5</span>
   <span class="bwb-mute">        │</span>
   <span class="bwb-mute">        ▼</span>
   <span class="bwb-emerald bwb-b">  dataset_transform(sample)</span>       <span class="bwb-mute"># dataset-owned: normalize, add, drop signals</span>
   <span class="bwb-mute">        │</span>
   <span class="bwb-mute">        ├──▶ </span><span class="bwb-emerald bwb-b">_get_target(sample)</span>       <span class="bwb-mute"># TS1/TS2 snapshot the label here, from clean data</span>
   <span class="bwb-mute">        ▼</span>
   <span class="bwb-violet bwb-b">  transform(sample)</span>               <span class="bwb-mute"># &lt;split&gt;_transforms from the config: augmentation</span>
   <span class="bwb-mute">        │</span>
   <span class="bwb-mute">        ▼</span>
   <span class="bwb-blue bwb-b">  model.input_fn(sample)</span>          <span class="bwb-mute"># last element of transform: sample -&gt; model inputs</span>
   <span class="bwb-mute">        │</span>
   <span class="bwb-mute">        ▼</span>
   <span class="bwb-slate bwb-b">  collate_fn</span>                      <span class="bwb-mute"># pads and stacks the batch</span>
   </pre>

``data.slice``
~~~~~~~~~~~~~~

A ``Data`` object is one whole recording, read lazily from the ``.h5``: spikes as an
irregular time series, behavior as a regular one, trials and split domains as intervals,
and non-temporal attributes such as units and session metadata.

``slice(start, end)`` cuts every time-based attribute down to the window, copies the rest
through, and re-zeroes timestamps to the window start.

``dataset_transform``
~~~~~~~~~~~~~~~~~~~~~

Owned by the dataset class, applied to every sample. It is where a dataset conditions
the signals it promises to carry: :class:`~ts1.IBLBrainWideBenchTS1` z-scores or bins
the behavioral target, :class:`~ts2.IBLBrainWideBenchTS2` drops the hold-out mask
belonging to the split it is not serving. The base implementation returns the sample
untouched.

It runs on a whole recording as readily as on a slice, so it can be called outside the
sampler:

.. code-block:: python

   data = dataset.dataset_transform(dataset.get_recording(recording_id))

``transform``
~~~~~~~~~~~~~

Unlike ``dataset_transform``, this one comes from outside the dataset: pass it as the
``transform`` argument when you construct a dataset yourself, or let a trainer build it
from the config. In ``link_model`` the trainer instantiates the ``train_transforms``,
``val_transforms`` and ``test_transforms`` lists and composes them onto the dataset:

.. code-block:: python

   dataset.transform = Compose([dataset.transform, *split_transforms, model.input_fn])

The lists are empty by default; a model that wants augmentation sets them in its own
trainer config:

.. code-block:: yaml

   train_transforms:
     - _target_: core.transforms.UnitDropout
       min_units: 0.6

Augmentation belongs here rather than in ``dataset_transform`` because the target is
snapshotted between the two, so nothing listed here can leak into the label.

.. note::

   Transforms come from two packages, and your own are welcome too.
   ``torch_brain.transforms`` carries the generic ones (``UnitDropout``, ``UnitFilter``,
   ``RandomCrop``, ``Compose``), :mod:`core.transforms` the benchmark's own
   (:class:`~core.transforms.FilterUnits`, :class:`~core.transforms.AdditivePepperNoise`).
   Anything taking and returning a ``Data`` can be listed.

``model.input_fn``
~~~~~~~~~~~~~~~~~~

Up to here the sample is still a ``Data`` object carrying every signal the recording
holds. ``input_fn`` is where the model picks out the ones it needs and puts them in the
shape it expects, with any light preprocessing that takes, binning the spikes for
instance.

It returns a dict: the ``model_inputs`` entry is splatted into ``forward``, and any other
top-level key is the model's own to read back in its trainer. It runs per item on the
dataloader workers, so it returns one sample's tensors, which ``collate_fn`` then pads
and stacks into the batch.
