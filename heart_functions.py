import numpy as np
from scipy.interpolate import interp1d
import neurokit2 as nk
from joblib import Parallel, delayed

#Compute CSI and CVI indices from RR intervals. Code source: https://github.com/diegocandiar/robust_hrv/blob/main/compute_CSI_CVI.m
def approximate_CSI_CVI(Fs, RR, t_RR, wind):
    """
    Parameters
    ----------
    Fs : float
        Sampling frequency for the time stamps
    RR : array_like
        Array of R-R intervals in (s).
    t_RR : array_like
        Corresponding time stamps (s) for the RR intervals.
    wind : float
        Window length (in same units as t_RR) for the time-varying calculation.

    Returns
    -------
    CSI_out : numpy.ndarray
        Interpolated CSI time series.
    CVI_out : numpy.ndarray
        Interpolated CVI time series.
    t_out : numpy.ndarray
        Time stamps corresponding to the output series.
    """
    # Sampling frequency
    Fs = Fs

    # Create time vector from first to last t_RR with spacing 1/Fs
    time = np.arange(t_RR[0], t_RR[-1] + 1/Fs, 1/Fs)

    # -----------------------------
    # First Poincare Plot Calculation
    # -----------------------------
    sd = np.diff(RR)
    # Use sample standard deviation (ddof=1) to mimic MATLAB's std behavior
    sd_std = np.std(sd, ddof=1)
    RR_std = np.std(RR, ddof=1)
    SD01 = np.sqrt(0.5 * (sd_std ** 2))
    SD02 = np.sqrt(2 * (RR_std ** 2) - (0.5 * (sd_std ** 2)))

    # -----------------------------
    # Time-varying SD Calculation
    # -----------------------------
    t1 = time[0]
    t2 = t1 + wind
    # Find indices where t_RR is greater than t2
    ixs = np.where(t_RR > t2)[0]
    nt = len(ixs) - 1  # number of segments (MATLAB: length(ixs)-1)

    # Preallocate arrays for SD1, SD2 and the center time t_C
    SD1 = np.zeros(nt)
    SD2 = np.zeros(nt)
    t_C = np.zeros(nt)

    for k in range(nt):
        i = ixs[k]
        t2_val = t_RR[i]
        t1_val = t_RR[i] - wind
        # Find indices for the current window [t1_val, t2_val]
        ix = np.where((t_RR >= t1_val) & (t_RR <= t2_val))[0]

        # Compute differences for the current window
        sd_local = np.diff(RR[ix])
        # Check to avoid issues with too few points
        local_sd_std = np.std(sd_local, ddof=1) if len(sd_local) > 1 else np.nan
        local_RR_std = np.std(RR[ix], ddof=1) if len(RR[ix]) > 1 else np.nan

        SD1[k] = np.sqrt(0.5 * (local_sd_std ** 2)) #approximate approach
        SD2[k] = np.sqrt(2 * (local_RR_std ** 2) - (0.5 * (local_sd_std ** 2)))
        t_C[k] = t2_val


    # Normalize the time-varying SDs to match the global values
    SD1 = SD1 - np.mean(SD1) + SD01
    SD2 = SD2 - np.mean(SD2) + SD02

    # -----------------------------
    # Compute CVI and CSI
    # -----------------------------
   
    CVI = SD1 * 10
    CSI = SD2 * 10

    # -----------------------------
    # Remove duplicates in t_C
    # -----------------------------
    t_C, unique_indices = np.unique(t_C, return_index=True)
    CVI = CVI[unique_indices]
    CSI = CSI[unique_indices]

    # -----------------------------
    # Interpolation to Create Output Time Series
    # -----------------------------
    
    t_out = np.arange(t_C[0], t_C[-1] + 1/Fs, 1/Fs)

    # Using cubic spline interpolation (equivalent to MATLAB's 'Spline')
    interp_CVI = interp1d(t_C, CVI, kind='cubic', fill_value="extrapolate")
    interp_CSI = interp1d(t_C, CSI, kind='cubic', fill_value="extrapolate")

    CVI_out = interp_CVI(t_out)
    CSI_out = interp_CSI(t_out)

    return CSI_out, CVI_out, t_out

#Compute at RMSSD continous over time_window.
def _compute_rmssd_at_index(i, rr_timestamps, half_window, min_beats):
    """
    Helper function to compute RMSSD at a single index using a sliding window.

    Parameters
    ----------
    i : int
        Index of the current beat (starting from 1).
    rr_timestamps : np.ndarray
        Array of R-peak timestamps (in seconds).
    half_window : float
        Half the window duration in seconds.
    min_beats : int
        Minimum number of beats required in the window.

    Returns
    -------
    rmssd_val : float
        Computed RMSSD (in milliseconds) at index i or np.nan if not enough data.
    """
    t = rr_timestamps[i]
    t_start = t - half_window
    t_end   = t + half_window
    # Find indices of beats within the time window
    idx = np.where((rr_timestamps >= t_start) & (rr_timestamps <= t_end))[0]
    if len(idx) >= min_beats:
        # Compute RR intervals in the window
        rr_window = np.diff(rr_timestamps[idx])
        if len(rr_window) > 1:
            diff_rr = np.diff(rr_window)
            rmssd_val = np.sqrt(np.mean(diff_rr ** 2)) * 1000  # convert to ms
            return rmssd_val
    return np.nan

def continuous_rmssd(rr_timestamps, window_sec=12, min_beats=8, parallel=False, n_jobs=-1):
    """
    Computes a continuous (dynamic) RMSSD measure over a sliding time window.
    
    The RMSSD timestamp for each computed value is assigned as the timestamp of the later beat (i.e. rr_timestamps[i]).

    Parameters
    ----------
    rr_timestamps : array-like
        Array of R-peak timestamps (in seconds).
    window_sec : float, optional
        Total duration of the sliding window in seconds (default 12, i.e. 6 s before and 6 s after the current beat).
    min_beats : int, optional
        Minimum number of beats required in the window (default 8).
    parallel : bool, optional
        If True, parallelizes the computation using joblib.
    n_jobs : int, optional
        Number of parallel jobs (default -1 uses all processors).

    Returns
    -------
    rmssd_values : np.ndarray
        Array of continuous RMSSD values (in milliseconds), one per beat starting from index 1.
    rmssd_timestamps : np.ndarray
        Array of corresponding timestamps, each equal to rr_timestamps[i] for i>=1.
    """
    rr_timestamps = np.asarray(rr_timestamps)
    half_window = window_sec / 2.0

    if parallel:
        rmssd_values = Parallel(n_jobs=n_jobs)(
            delayed(_compute_rmssd_at_index)(i, rr_timestamps, half_window, min_beats)
            for i in range(1, len(rr_timestamps))
        )
        rmssd_values = np.array(rmssd_values)
    else:
        rmssd_values = []
        for i in range(1, len(rr_timestamps)):
            rmssd_values.append(_compute_rmssd_at_index(i, rr_timestamps, half_window, min_beats))
        rmssd_values = np.array(rmssd_values)

    # Timestamps for RMSSD are the later beat's time (starting from index 1)
    rmssd_times = rr_timestamps[1:]
    
    # Interpolate over NaN values for a continuous signal
    x = np.arange(len(rmssd_values))
    not_nan = ~np.isnan(rmssd_values)
    if np.sum(not_nan) >= 2:  # Only interpolate if there are at least two non-NaN values
        f_interp = interp1d(x[not_nan], rmssd_values[not_nan], kind='linear', fill_value="extrapolate")
        rmssd_values = f_interp(x)
    
    return rmssd_values, rmssd_times

#Compute at LF HF bands
def separate_hr_bands(t, hr_data, fs):
    """
    Splits a heart-rate time series into LF and HF bands using band-pass filters.
    
    Parameters
    ----------
    t : array-like
        The time vector (in seconds) corresponding to hr_data.
    hr_data : array-like
        1D array of the heart-rate or IBI signal (must be uniformly sampled).
    fs : int or float
        Sampling frequency in Hz.
    
    Returns
    -------
    t : ndarray
        Time array in seconds (the same as the input).
    lf_signal : ndarray
        The HR signal filtered in the LF band (0.04 - 0.15 Hz).
    hf_signal : ndarray
        The HR signal filtered in the HF band (0.15 - 0.4 Hz).
    """
    # Use the provided time vector 't'
    
    lf_signal = nk.signal_filter(
        hr_data,
        sampling_rate=fs,
        lowcut=0.04,
        highcut=0.15,
        method="butterworth",
        order=2
    )

    hf_signal = nk.signal_filter(
        hr_data,
        sampling_rate=fs,
        lowcut=0.15,
        highcut=0.4,
        method="butterworth",
        order=2
    )

    return t, lf_signal, hf_signal

#Detection of r_peaks and IBI
def detect_rpeaks_and_compute_ibi(ecg_signal, fs=1000, refractory_period_bpm=220, adjust=True, search_window=100):
    """
    Detect R-peaks in an ECG signal using NeuroKit2 and compute RR intervals (IBI).

    Parameters
    ----------
    ecg_signal : array-like
        Raw ECG signal.
    fs : int or float
        Sampling frequency in Hz.
    refractory_period_bpm : float, optional
        Maximum expected heart rate to define refractory period (default 220 BPM).
    adjust : bool, optional
        If True, realigns outlier R-peaks based on local amplitude.
    search_window : int, optional
        Number of samples to search for a local max when adjusting peaks.

    Returns
    -------
    rpeaks : np.array
        Indices of detected R-peaks.
    rpeaks_times : np.array
        Timestamps (in seconds) corresponding to each R-peak.
    ibi : np.array
        RR intervals (IBI) in seconds.
    ibi_times : np.array
        Timestamps (in seconds) assigned to the last between two consecutive R-peaks.
    """
    # Process the ECG signal using NeuroKit2
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]
    
    # Apply refractory period: remove peaks that are too close to each other
    refractory_samples = int((60 / refractory_period_bpm) * fs)
    filtered_rpeaks = [rpeaks[0]] if len(rpeaks) > 0 else []
    for i in range(1, len(rpeaks)):
        if rpeaks[i] - filtered_rpeaks[-1] > refractory_samples:
            filtered_rpeaks.append(rpeaks[i])
    rpeaks = np.array(filtered_rpeaks)
    
    # Realign outlier peaks based on local amplitude
    if adjust:
        rpeaks = realign_outlier_peaks(ecg_signal, rpeaks, search_window=search_window)
    
    # Convert indices to timestamps in seconds
    rpeaks_times = rpeaks / fs

    # Compute RR intervals (IBI) in seconds
    ibi = np.diff(rpeaks_times)
    # Assign each IBI to the midpoint between two consecutive R-peaks
    #ibi_times = rpeaks_times[:-1] + ibi / 2 #midpoint between R-peaks as the representative IBI
    ibi_times = rpeaks_times[1:] 
    
    return rpeaks, rpeaks_times, ibi, ibi_times

def realign_outlier_peaks(ecg_signal, r_peaks, search_window=100, factor=0.7, min_diff=0.1):
    """
    Automatically realign outlier R-peaks based on local amplitude.

    Parameters
    ----------
    ecg_signal : array-like
        Raw ECG signal.
    r_peaks : array-like
        Indices of detected R-peaks.
    search_window : int, optional
        Number of samples to search around each peak.
    factor : float, optional
        Threshold to identify an outlier (peak < factor * median amplitude).
    min_diff : float, optional
        Minimum difference from median amplitude to consider a realignment.

    Returns
    -------
    new_r_peaks : np.array
        Adjusted R-peak indices.
    """
    r_peaks = np.array(r_peaks)
    amplitudes = ecg_signal[r_peaks]
    median_amp = np.median(amplitudes)

    # Identify outlier peaks with low amplitude
    outlier_mask = (amplitudes < factor * median_amp) & (np.abs(amplitudes - median_amp) > min_diff)
    new_r_peaks = r_peaks.copy()
    n = len(ecg_signal)
    
    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            peak_idx = r_peaks[i]
            original_amp = ecg_signal[peak_idx]
            start = max(0, peak_idx - search_window)
            end = min(n, peak_idx + search_window + 1)
            local_window = ecg_signal[start:end]
            local_max_idx = np.argmax(local_window)
            aligned_idx = start + local_max_idx
            if ecg_signal[aligned_idx] > original_amp:
                new_r_peaks[i] = aligned_idx
    new_r_peaks.sort()
    return new_r_peaks

def uniform_interpolation(metric_values, timestamps, new_fs=1, start_time=None, end_time=None, kind='linear'):
    """
    Interpolates a metric time series uniformly onto a new time axis.

    Parameters
    ----------
    metric_values : array-like
        The values of the metric (e.g., RMSSD) at the given timestamps.
    timestamps : array-like
        The original timestamps (in seconds) corresponding to the metric values.
    new_fs : float, optional
        Desired sampling frequency (Hz) for the uniform time series (default is 1 Hz, i.e. one sample per second).
    start_time : float, optional
        Start time for the uniform time axis. If None, it uses the minimum of timestamps.
    end_time : float, optional
        End time for the uniform time axis. If None, it uses the maximum of timestamps.
    kind : str, optional
        Type of interpolation to use (default is 'linear').

    Returns
    -------
    new_time : np.ndarray
        Uniformly spaced time axis (in seconds).
    new_metric : np.ndarray
        Interpolated metric values corresponding to new_time.
    """
    timestamps = np.asarray(timestamps)
    metric_values = np.asarray(metric_values)
    
    if start_time is None:
        start_time = timestamps.min()
    if end_time is None:
        end_time = timestamps.max()
    
    # Create a uniform time axis at the desired sampling rate
    new_time = np.arange(start_time, end_time, 1/new_fs)
    
    # Create an interpolation function
    f_interp = interp1d(timestamps, metric_values, kind=kind, fill_value="extrapolate")
    
    # Interpolate the metric onto the new time axis
    new_metric = f_interp(new_time)
    
    return new_time, new_metric


