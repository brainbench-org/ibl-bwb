from collections.abc import Callable

from core.dataset import BenchmarkRegime, UnitQCPolicy, WholeSessionSpikeDataset

# The population TS3 scores from: FAIL probes out, WARNING kept. Scoring needs
# qc_neural == PASS, so the scored units are a subset of this and an extraction always
# covers them; a tighter policy here could cover less than TS3 scores.
TS3_UNIT_QC = UnitQCPolicy(keep_qc_neural=("PASS", "WARNING"))


class IBLBrainWideBenchTS3(WholeSessionSpikeDataset):
    r"""TS3's view of the benchmark: whole sessions after neural QC.

    Shares its units with unit-level pretraining: the units TS3 scores are the units those
    models saw. The class exists to state that: the QC policy is fixed rather than a
    parameter, so no consumer can evaluate on a population other than the suite's, and a
    contract failure says which suite asked.

    Args:
        root: The root directory of the dataset.
        regime: The regime of the dataset (pretrain, eval).
        dirname: The name of the dataset (and the directory containing its data).
        transform: The transform(s) to apply to the data.
    """

    def __init__(
        self,
        root: str,
        regime: BenchmarkRegime,
        dirname: str = "ibl_brain_wide_bench_2026",
        transform: Callable | None = None,
        **kwargs,
    ):

        super().__init__(
            root=root,
            regime=regime,
            dirname=dirname,
            transform=transform,
            unit_qc=TS3_UNIT_QC,
            contract="TS3 eval" if regime == "eval" else "TS3 pretrain",
            **kwargs,
        )
