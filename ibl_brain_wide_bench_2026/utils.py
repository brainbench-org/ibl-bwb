from typing import Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, cheby1, decimate, resample, sosfiltfilt
from torch_brain.data import Interval, IrregularTimeSeries, RegularTimeSeries

RESAMPLE_ANTIALIAS_ORDER = 8
RESAMPLE_MAX_RIPPLE_DB = 0.05
NYQUIST_SAFE_RATIO = 0.8
PAD_SEC = 1

DFLT_RESAMPLE_FTYPE = "butter"
DFLT_DECIMATE_FTYPE = "fir"

REGULARIZE_INTERP_METHOD = "linear"
RESAMPLE_INTERP_METHOD = "linear"


def _get_domain(timestamps: np.ndarray, fs: float, tol: float) -> Interval:
    """Find the domain of almost regular data: the intervals where it was recorded.

    Almost regular data is data where the timestamps are not exactly regular, but close to
    regular. This function supports data that might not be contiguous, and finds the domain
    by locating contiguous blocks of data where the difference between consecutive
    timestamps is less than (1 + tol) * (1/fs).

    Args:
        timestamps: The timestamps of the data.
        fs: The sampling frequency of the data, in Hz.
        tol: The tolerance for the difference between consecutive timestamps.

    Returns:
        An Interval object covering the contiguous data blocks.
    """
    if len(timestamps) == 0:
        return Interval(start=np.array([]), end=np.array([]))

    dt = np.diff(timestamps)
    gap_indices = np.where(dt > (1 + tol) / fs)[0]
    start_indices = np.concatenate(([0], gap_indices + 1))
    end_indices = np.concatenate((gap_indices, [len(timestamps) - 1]))
    start = timestamps[start_indices]
    end = timestamps[end_indices]

    return Interval(start, end)


def regularize_timeseries(
    irregular_timeseries: IrregularTimeSeries,
    gap_tol: float,
    target_fs: float | None = None,
    start_timestamp: float | None = None,
    end_timestamp: float | None = None,
) -> IrregularTimeSeries:
    r"""Convert almost regular data to a regular time series.

    Almost regular data is data where the timestamps are not exactly regular, but close to
    regular. This function supports data that might not be contiguous, and interpolates the
    data onto a uniform timestamp grid.

    .. note::
        This function currently returns an IrregularTimeSeries, but it should be a RegularTimeSeries.
        This will be updated once we add support for non-contiguous data in RegularTimeSeries.

    Args:
        irregular_timeseries: The IrregularTimeSeries to resample.
        gap_tol: The tolerance for the difference between consecutive timestamps.
        target_fs: Grid frequency in Hz. Defaults to the input's inferred frequency;
            when given, the input must be within 5% of it or ValueError is raised.
        start_timestamp: First grid timestamp. Requires end_timestamp.
        end_timestamp: Exclusive upper bound on the grid.

    Returns:
        An IrregularTimeSeries with the same keys, on a uniform grid at target_fs.
    """
    # the signal is almost regular, so we need to resample it to the target frequency
    # the frequency is not exactly the target frequency, so we need to allow for some tolerance
    timestamps = irregular_timeseries.timestamps
    raw_fs = 1.0 / np.median(np.diff(timestamps))

    if target_fs is not None:
        if np.abs(raw_fs - target_fs) / target_fs > 0.05:
            raise ValueError(
                f"The raw frequency {raw_fs} is not close to the target frequency {target_fs}"
            )
    else:
        target_fs = raw_fs

    # next, we might have some missing timestamps, so we will determine the domain of the signal
    domain = _get_domain(timestamps, fs=target_fs, tol=gap_tol)

    # regularize timestamps by using interpolate
    if start_timestamp is not None:
        assert end_timestamp is not None
        domain = domain.__and__(Interval(start_timestamp, end_timestamp))

    start_timestamp = start_timestamp or timestamps[0]
    end_timestamp = end_timestamp or timestamps[-1]
    # update the domain to the new start and end timestamps
    regular_timestamps = np.arange(
        start_timestamp, end_timestamp, 1.0 / target_fs, dtype=np.float64
    )

    out = {}

    for key in irregular_timeseries.keys():  # noqa: SIM118
        if key == "timestamps":
            continue
        signal = getattr(irregular_timeseries, key)

        data_interpolator = interp1d(
            timestamps,
            signal,
            kind=REGULARIZE_INTERP_METHOD,
            fill_value="extrapolate",
            bounds_error=False,
            axis=0,
        )
        out[key] = data_interpolator(regular_timestamps)

    out = IrregularTimeSeries(
        timestamps=regular_timestamps,
        **out,
        domain=domain,
    )

    return out


def resample_timeseries(
    source_timeseries: RegularTimeSeries | IrregularTimeSeries,
    target_fs: float,
    source_fs: float | None = None,
    filter_type: Literal["cheby", "butter"] = DFLT_RESAMPLE_FTYPE,
) -> RegularTimeSeries:
    if source_fs is None:
        source_fs = 1 / np.median(np.diff(source_timeseries.timestamps))

    n_pad_source = int(PAD_SEC * source_fs)

    # Adjust new_size to include the padding we are about to add
    total_len = len(source_timeseries) + (2 * n_pad_source)
    new_size = int(total_len * target_fs / source_fs)

    # We need a temporal grid that covers the padded area for the interpolator
    # We extend the raw timestamps for the filtering/resampling stage
    offset = n_pad_source / source_fs
    padded_timestamps = np.linspace(
        start=source_timeseries.timestamps[0] - offset,
        stop=source_timeseries.timestamps[-1] + offset,
        num=total_len,
    )

    Wn = NYQUIST_SAFE_RATIO * target_fs / source_fs

    if filter_type == "cheby":
        sos = cheby1(N=RESAMPLE_ANTIALIAS_ORDER, rp=RESAMPLE_MAX_RIPPLE_DB, Wn=Wn, output="sos")
    elif filter_type == "butter":
        sos = butter(N=RESAMPLE_ANTIALIAS_ORDER, Wn=Wn, output="sos")
    else:
        raise ValueError(f" filter_type {filter_type} is not implemented")

    dt = 1.0 / target_fs
    target_timestamps = np.arange(
        source_timeseries.timestamps[0],
        source_timeseries.timestamps[-1],
        dt,
        dtype=np.float64,
    )

    out = {}
    for key in source_timeseries.keys():  # noqa: SIM118
        if key == "timestamps":
            continue
        signal = getattr(source_timeseries, key)

        pad = ((n_pad_source, n_pad_source), (0, 0)) if signal.ndim > 1 else n_pad_source
        signal = np.pad(signal, pad_width=pad, mode="reflect")

        filtered_signal = sosfiltfilt(sos, signal, axis=0)
        resampled_signal, resampled_timestamps = resample(
            filtered_signal, new_size, t=padded_timestamps, axis=0
        )

        interpolator = interp1d(
            resampled_timestamps,
            resampled_signal,
            kind=RESAMPLE_INTERP_METHOD,
            fill_value="extrapolate",
            bounds_error=False,
            axis=0,
        )

        resampled_signal = interpolator(target_timestamps)

        resampled_signal = np.clip(
            resampled_signal, a_min=signal.min(axis=0), a_max=signal.max(axis=0)
        )

        out[key] = resampled_signal

    domain_start = source_timeseries.timestamps[0]
    target = RegularTimeSeries(
        sampling_rate=target_fs,
        domain_start=domain_start,
        **out,
    )

    return target


def decimate_signal(
    source_signal: np.ndarray,
    downsample_factors: list,
    source_fs: float,
    ftype: str = DFLT_DECIMATE_FTYPE,
):

    n_pad = int(PAD_SEC * source_fs)

    signal = np.pad(source_signal, pad_width=n_pad, mode="reflect")

    for factor in downsample_factors:
        signal = decimate(signal, factor, ftype=ftype)

    final_n_pad = n_pad // np.prod(downsample_factors)
    return signal[final_n_pad:-final_n_pad]
