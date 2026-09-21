"""A split is half of an attribute name, so a dataset declares the ones its build carries.

The pipeline writes one ``f"{scope}_{split}_domain"`` per scope: train and val for
pretrain, all three for ts1 and ts2, none for whole-session sampling, whose split axis is
held-out animals. ``SPLITS`` states that per class, checked before a recording is opened.
``split="test"`` on a pretraining dataset used to die as ``AttributeError:
pretrain_test_domain`` inside ``get_trial_intervals``.
"""

import inspect
import re
from pathlib import Path

import pytest

from core.data import read_recording_ids
from core.dataset import IBLBrainWideBench2026, UnitQCPolicy, WholeSessionSpikeDataset
from pretrain.datasets.mask_modeling_spike import IBLBrainWideBenchMaskModelingSpikes
from pretrain.datasets.multi_task_behavior import IBLBrainWideBenchMultiTaskBehavior
from pretrain.datasets.single_task_behavior import IBLBrainWideBenchSingleTaskBehavior
from ts1.ts1_dataset import IBLBrainWideBenchTS1
from ts2.ts2_dataset import IBLBrainWideBenchTS2

PIPELINE = Path(__file__).parents[2] / "ibl_brain_wide_bench_2026" / "pipeline.py"

# Every dataset that samples inside a split, with the scope its domain attribute names.
CONSUMERS = [
    (IBLBrainWideBenchTS1, "ts1"),
    (IBLBrainWideBenchTS2, "ts2"),
    (IBLBrainWideBenchMaskModelingSpikes, "pretrain"),
    (IBLBrainWideBenchSingleTaskBehavior, "pretrain"),
    (IBLBrainWideBenchMultiTaskBehavior, "pretrain"),
]


@pytest.mark.skipif(not PIPELINE.is_file(), reason="pipeline package not checked out")
@pytest.mark.parametrize("cls,scope", CONSUMERS, ids=[c.__name__ for c, _ in CONSUMERS])
def test_declared_splits_are_the_ones_the_pipeline_writes(cls, scope):
    """And that the scope checked here is the one the class actually reads."""
    source = inspect.getsource(cls)
    assert re.search(rf'f"{scope}_{{self\.split}}_domain"', source), f"{cls.__name__} not {scope}"

    written = set(re.findall(rf"data\.{scope}_(\w+)_domain\s*=", PIPELINE.read_text()))
    assert set(cls.SPLITS) == written, cls.__name__


def test_no_dataset_defaults_its_split():
    """Naming the split is the caller's job, all the way down to the base class.

    It used to default to ``"train"``, which only the metadata-only readers ever hit, and
    for them it claimed a temporal restriction they do not apply. ``None`` says what such
    a read means; a sampling dataset says which split it samples.
    """
    for cls in [IBLBrainWideBench2026, *(c for c, _ in CONSUMERS)]:
        split = inspect.signature(cls.__init__).parameters["split"]
        assert split.default is inspect.Parameter.empty, f"{cls.__name__} defaults its split"


def test_whole_session_sampling_declares_no_split():
    """Its units are split by subject inside the probe, so no split names a domain."""
    assert WholeSessionSpikeDataset.SPLITS == (None,)
    assert None not in IBLBrainWideBenchTS1.SPLITS


def test_a_split_the_build_does_not_carry_is_refused_before_any_io():
    """The root does not exist, so only a check that runs before the build can raise this."""
    with pytest.raises(ValueError, match="does not define split"):
        IBLBrainWideBenchMaskModelingSpikes(root="/nonexistent", split="test")


def test_whole_session_sampling_narrows_but_never_widens():
    """A caller may pick sessions out of the regime's list, never add to it."""
    assert read_recording_ids("pretrain"), "the pretrain list is empty, this test is moot"

    with pytest.raises(ValueError, match="are not in the pretrain recordings"):
        WholeSessionSpikeDataset(
            root="/nonexistent",
            regime="pretrain",
            unit_qc=UnitQCPolicy(keep_qc_neural=("PASS", "WARNING")),
            recording_ids="not-a-session",
        )
