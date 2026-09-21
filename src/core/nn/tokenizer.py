import numpy as np


class StringArrayTokenizer:
    def __init__(self, vocab, unknown_token: str = "NA"):

        vocab = self._normalize_values(vocab)

        if unknown_token in vocab:
            raise ValueError(f"{unknown_token} is a reserved token")

        self.unknown_token = unknown_token

        unique_vocab = sorted(set(vocab))
        self.vocab = [self.unknown_token, *unique_vocab]

        self.str_to_id = {name: i for i, name in enumerate(self.vocab)}

        # Efficiency consideration: Vectorize the dictionary lookup for tokenization
        self._vectorized_get = np.vectorize(
            lambda x: self.str_to_id.get(str(x), 0), otypes=[np.int64]
        )

    def _normalize_values(self, values):
        """Simplifies and speeds up normalization."""
        if isinstance(values, (bytes, np.bytes_)):
            return values.decode("utf-8")

        if isinstance(values, (str, int, float)):
            return str(values)

        # Convert lists/tuples to numpy arrays first for easier handling
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if isinstance(values, np.ndarray):
            # If it's an array of bytes, use numpy's highly optimized char decoder
            if values.dtype.kind == "S":
                return np.char.decode(values, "utf-8")
            # Otherwise, force it to a string array
            return values.astype(str)

        return str(values)

    def input_fn(self, values: np.ndarray | list | str):
        values = self._normalize_values(values)

        if isinstance(values, np.ndarray):
            # Return early if already integers
            if values.dtype.kind in {"i", "u"}:
                return values.astype(np.int64)

            # Use the vectorized dictionary lookup (faster than list comprehensions)
            return self._vectorized_get(values)

        # If it's a single value
        return np.int64(self.str_to_id.get(str(values), 0))
