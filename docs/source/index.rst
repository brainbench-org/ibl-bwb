IBL BrainWideBench
==================

**IBL BrainWideBench** is a benchmark for evaluating neural decoding models on large-scale
brain-wide recordings from the International Brain Laboratory.

If you encounter any bugs or have feature requests, please submit them to our
`GitHub Issues page <https://github.com/brainbench-org/ibl-bwb/issues>`_.

.. raw:: html

   <p class="bwb-pill-row">
     <a class="bwb-pill" href="guides/setup.html"><i class="fa-solid fa-rocket"></i>Get started</a>
     <a class="bwb-pill bwb-pill-ghost" href="https://bwb.iblcore.org"><i class="fa-solid fa-ranking-star"></i>Leaderboard</a>
     <a class="bwb-pill bwb-pill-ghost" href="https://arxiv.org/abs/2609.22064"><i class="fa-solid fa-file-lines"></i>Paper</a>
     <a class="bwb-pill bwb-pill-ghost" href="https://huggingface.co/collections/nerdslab/ibl-bwb"><i class="fa-brands fa-hugging-face"></i>Checkpoints</a>
     <a class="bwb-pill bwb-pill-ghost" href="https://wandb.ai/ibl-bwb/projects"><svg class="bwb-brand" viewBox="0 0 24 24" aria-hidden="true"><path d="M2.48 0a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm19.04 0a1.55 1.55 0 1 0 0 3.101 1.55 1.55 0 0 0 0-3.101zM12 2.295a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zM2.48 5.272a2.48 2.48 0 1 0 0 4.96 2.48 2.48 0 0 0 0-4.96zm19.04 0a2.48 2.48 0 1 0 0 4.96 2.48 2.48 0 0 0 0-4.96zM12 8.496a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm-9.52 3.907a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1zm19.04 0a1.55 1.55 0 1 0 0 3.102 1.55 1.55 0 0 0 0-3.102zM12 13.767a2.48 2.48 0 1 0 0 4.962 2.48 2.48 0 0 0 0-4.962zm-9.52 3.907a2.48 2.48 0 1 0 .001 4.962 2.48 2.48 0 0 0 0-4.962zm19.04.93a1.55 1.55 0 1 0 0 3.102 1.55 1.55 0 0 0 0-3.101zM12 20.9a1.55 1.55 0 1 0 0 3.1 1.55 1.55 0 0 0 0-3.1Z"/></svg>Runs</a>
     <a class="bwb-pill bwb-pill-ghost" href="https://github.com/brainbench-org/ibl-bwb"><i class="fa-brands fa-github"></i>GitHub</a>
   </p>

----

.. _citation:

Citation
--------

If you use IBL BrainWideBench, please cite the paper:

.. literalinclude:: ../../CITATION.bib
   :language: bibtex
   :start-after: %%% paper
   :end-before: %%% dataset

The benchmark is built on the IBL Brain Wide Map, so please cite the dataset as well:

.. literalinclude:: ../../CITATION.bib
   :language: bibtex
   :start-after: %%% dataset

.. toctree::
   :maxdepth: 2
   :hidden:

   Dataset <guides/dataset>
   Setup <guides/setup>
   Codebase <guides/overview>
   Pretraining <guides/pretraining>
   TS1: Behavior <guides/ts1>
   TS2: Dynamics <guides/ts2>
   TS3: Anatomy <guides/ts3>
   Submission <guides/submission>
   API <generated/api/index>
   Refs <references>
