class TrainingConstraintsError(Exception):
    """Exception raised during training/tuning when hyperparams are invalid for the data."""

    pass


class DatasetContractError(Exception):
    """Exception raised when the dataset build does not satisfy the task suite's contract."""

    pass
