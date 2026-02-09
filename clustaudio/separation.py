"""
Audio Separation and Denoising module.

This module provides advanced audio processing for:
- Foreground/background separation
- Noise reduction and denoising
- Signal enhancement for noisy communications (ATC, radio, etc.)
- Wind noise removal
- Interference suppression
- Voice/speech enhancement

Designed to work with VAD for comprehensive audio analysis.

Author: AI Assistant
"""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

# Try to import advanced denoising libraries
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logger.debug("noisereduce not available. Using librosa-based denoising. Install with: pip install noisereduce")

try:
    from scipy.signal import wiener
    WIENER_AVAILABLE = True
except ImportError:
    WIENER_AVAILABLE = False


def denoise_audio(audio, sr=22050, method='spectral', stationary=True, **kwargs):
    """Denoise audio using various methods.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    method : str
        Denoising method:
        - 'spectral': Spectral subtraction (default)
        - 'wiener': Wiener filtering
        - 'noisereduce': Advanced noise reduction (requires noisereduce)
        - 'median': Median filtering
        - 'bandpass': Band-pass filtering
    stationary : bool
        Whether noise is stationary (constant over time)
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    np.ndarray
        Denoised audio
        
    Examples
    --------
    >>> import separation
    >>> clean_audio = separation.denoise_audio(noisy_audio, sr=16000, method='spectral')
    """
    if method == 'noisereduce' and NOISEREDUCE_AVAILABLE:
        return _denoise_noisereduce(audio, sr, stationary=stationary, **kwargs)
    elif method == 'spectral':
        return _denoise_spectral_subtraction(audio, sr, stationary=stationary, **kwargs)
    elif method == 'wiener':
        return _denoise_wiener(audio, sr, **kwargs)
    elif method == 'median':
        return _denoise_median(audio, **kwargs)
    elif method == 'bandpass':
        return _denoise_bandpass(audio, sr, **kwargs)
    else:
        logger.warning(f"Unknown denoising method: {method}, returning original audio")
        return audio


def _denoise_noisereduce(audio, sr, stationary=True, prop_decrease=1.0, **kwargs):
    """Advanced noise reduction using noisereduce library."""
    if not NOISEREDUCE_AVAILABLE:
        logger.warning("noisereduce not available, falling back to spectral subtraction")
        return _denoise_spectral_subtraction(audio, sr, stationary=stationary)
    
    try:
        # Use noisereduce library
        denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            **kwargs
        )
        return denoised
    except Exception as e:
        logger.error(f"Noisereduce failed: {str(e)}, using spectral subtraction")
        return _denoise_spectral_subtraction(audio, sr, stationary=stationary)


def _denoise_spectral_subtraction(audio, sr, stationary=True, noise_duration=1.0, 
                                   alpha=2.0, beta=0.1):
    """Spectral subtraction denoising.
    
    Parameters
    ----------
    noise_duration : float
        Duration (seconds) at start assumed to be noise only
    alpha : float
        Over-subtraction factor (1-3, higher = more aggressive)
    beta : float
        Spectral floor (0-1, prevents over-subtraction artifacts)
    """
    # Compute STFT
    n_fft = 2048
    hop_length = n_fft // 4
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum from initial frames
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = min(noise_frames, magnitude.shape[1] // 4)  # Max 25% of signal
    
    if stationary:
        # Use initial frames for noise estimate
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    else:
        # Use minimum statistics for non-stationary noise
        window_size = max(noise_frames, 10)
        noise_spectrum = np.zeros_like(magnitude)
        for i in range(magnitude.shape[0]):
            noise_spectrum[i, :] = median_filter(magnitude[i, :], size=window_size, mode='nearest')
    
    # Spectral subtraction
    magnitude_denoised = magnitude - alpha * noise_spectrum
    
    # Apply spectral floor
    magnitude_denoised = np.maximum(magnitude_denoised, beta * magnitude)
    
    # Reconstruct audio
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)
    
    # Match length
    if len(audio_denoised) < len(audio):
        audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
    else:
        audio_denoised = audio_denoised[:len(audio)]
    
    return audio_denoised


def _denoise_wiener(audio, sr, noise_frames=None):
    """Wiener filtering for denoising."""
    if not WIENER_AVAILABLE:
        return _denoise_spectral_subtraction(audio, sr)
    
    # Apply Wiener filter
    denoised = wiener(audio)
    return denoised


def _denoise_median(audio, kernel_size=5):
    """Median filtering for impulse noise."""
    denoised = median_filter(audio, size=kernel_size, mode='reflect')
    return denoised


def _denoise_bandpass(audio, sr, lowcut=300, highcut=3400):
    """Band-pass filtering (useful for speech: 300-3400 Hz)."""
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth band-pass filter
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Apply filter
    denoised = signal.filtfilt(b, a, audio)
    
    return denoised


def separate_foreground_background(audio, sr=22050, method='spectral_masking',
                                   noise_profile_duration=1.0, **kwargs):
    """Separate foreground (speech/voice) from background (noise).
    
    This function separates audio into foreground and background components,
    useful for analyzing both signal and noise characteristics.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    method : str
        Separation method:
        - 'spectral_masking': Spectral masking based on energy
        - 'harmonic_percussive': Harmonic-percussive source separation
        - 'nmf': Non-negative matrix factorization
        - 'wiener': Wiener filtering approach
    noise_profile_duration : float
        Duration (seconds) for noise profile estimation
    **kwargs
        Method-specific parameters
        
    Returns
    -------
    dict
        Dictionary containing:
        - foreground: Foreground/signal audio
        - background: Background/noise audio
        - foreground_mask: Binary mask for foreground
        - snr_db: Signal-to-noise ratio in dB
        
    Examples
    --------
    >>> result = separation.separate_foreground_background(audio, method='spectral_masking')
    >>> foreground = result['foreground']
    >>> background = result['background']
    >>> print(f"SNR: {result['snr_db']:.2f} dB")
    """
    if method == 'spectral_masking':
        return _separate_spectral_masking(audio, sr, noise_profile_duration, **kwargs)
    elif method == 'harmonic_percussive':
        return _separate_harmonic_percussive(audio, sr, **kwargs)
    elif method == 'nmf':
        return _separate_nmf(audio, sr, **kwargs)
    elif method == 'wiener':
        return _separate_wiener_masking(audio, sr, noise_profile_duration, **kwargs)
    else:
        logger.warning(f"Unknown separation method: {method}")
        return {
            'foreground': audio,
            'background': np.zeros_like(audio),
            'foreground_mask': np.ones(len(audio), dtype=bool),
            'snr_db': 0.0
        }


def _separate_spectral_masking(audio, sr, noise_duration=1.0, 
                               threshold_db=10, smoothing_frames=5):
    """Separate using spectral masking based on SNR."""
    # Compute STFT
    n_fft = 2048
    hop_length = n_fft // 4
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = min(noise_frames, magnitude.shape[1] // 4)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Compute SNR for each time-frequency bin
    snr_db = 20 * np.log10((magnitude + 1e-10) / (noise_spectrum + 1e-10))
    
    # Create binary mask
    mask = (snr_db > threshold_db).astype(float)
    
    # Smooth mask over time
    if smoothing_frames > 0:
        for i in range(mask.shape[0]):
            mask[i, :] = median_filter(mask[i, :], size=smoothing_frames, mode='nearest')
    
    # Apply masks
    stft_foreground = stft * mask
    stft_background = stft * (1 - mask)
    
    # Reconstruct
    foreground = librosa.istft(stft_foreground, hop_length=hop_length)
    background = librosa.istft(stft_background, hop_length=hop_length)
    
    # Match lengths
    target_len = len(audio)
    foreground = _match_length(foreground, target_len)
    background = _match_length(background, target_len)
    
    # Create time-domain mask
    foreground_mask = np.abs(foreground) > np.abs(background)
    
    # Compute overall SNR
    signal_power = np.mean(foreground ** 2)
    noise_power = np.mean(background ** 2)
    snr_db_overall = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'foreground': foreground,
        'background': background,
        'foreground_mask': foreground_mask,
        'snr_db': snr_db_overall
    }


def _separate_harmonic_percussive(audio, sr, margin=1.0):
    """Separate using harmonic-percussive source separation."""
    # Use librosa's HPSS
    harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
    
    # Treat harmonic as foreground (voice/tonal), percussive as background (noise)
    foreground = harmonic
    background = percussive
    
    foreground_mask = np.abs(foreground) > np.abs(background)
    
    signal_power = np.mean(foreground ** 2)
    noise_power = np.mean(background ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'foreground': foreground,
        'background': background,
        'foreground_mask': foreground_mask,
        'snr_db': snr_db
    }


def _separate_nmf(audio, sr, n_components=4, n_foreground=2):
    """Separate using Non-negative Matrix Factorization."""
    from sklearn.decomposition import NMF
    
    # Compute spectrogram
    n_fft = 2048
    hop_length = n_fft // 4
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Apply NMF
    nmf_model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = nmf_model.fit_transform(magnitude)
    H = nmf_model.components_
    
    # Reconstruct foreground (first n_foreground components)
    magnitude_foreground = W[:, :n_foreground] @ H[:n_foreground, :]
    
    # Reconstruct background (remaining components)
    magnitude_background = W[:, n_foreground:] @ H[n_foreground:, :]
    
    # Reconstruct audio
    stft_foreground = magnitude_foreground * np.exp(1j * phase)
    stft_background = magnitude_background * np.exp(1j * phase)
    
    foreground = librosa.istft(stft_foreground, hop_length=hop_length)
    background = librosa.istft(stft_background, hop_length=hop_length)
    
    # Match lengths
    target_len = len(audio)
    foreground = _match_length(foreground, target_len)
    background = _match_length(background, target_len)
    
    foreground_mask = np.abs(foreground) > np.abs(background)
    
    signal_power = np.mean(foreground ** 2)
    noise_power = np.mean(background ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'foreground': foreground,
        'background': background,
        'foreground_mask': foreground_mask,
        'snr_db': snr_db
    }


def _separate_wiener_masking(audio, sr, noise_duration=1.0, alpha=2.0):
    """Wiener filter-based separation."""
    n_fft = 2048
    hop_length = n_fft // 4
    
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = min(noise_frames, magnitude.shape[1] // 4)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Wiener filter gain
    signal_power = magnitude ** 2
    noise_power = (noise_spectrum * alpha) ** 2
    wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
    
    # Apply gains
    magnitude_foreground = magnitude * wiener_gain
    magnitude_background = magnitude * (1 - wiener_gain)
    
    # Reconstruct
    stft_foreground = magnitude_foreground * np.exp(1j * phase)
    stft_background = magnitude_background * np.exp(1j * phase)
    
    foreground = librosa.istft(stft_foreground, hop_length=hop_length)
    background = librosa.istft(stft_background, hop_length=hop_length)
    
    target_len = len(audio)
    foreground = _match_length(foreground, target_len)
    background = _match_length(background, target_len)
    
    foreground_mask = np.abs(foreground) > np.abs(background)
    
    signal_power = np.mean(foreground ** 2)
    noise_power = np.mean(background ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'foreground': foreground,
        'background': background,
        'foreground_mask': foreground_mask,
        'snr_db': snr_db
    }


def _match_length(audio, target_length):
    """Match audio length to target."""
    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))
    else:
        return audio[:target_length]


def extract_foreground_background_segments(filepath, output_dir=None, sr=16000,
                                          method='spectral_masking',
                                          denoise=True, denoise_method='spectral',
                                          save_both=True, **kwargs):
    """Extract foreground and background from audio file and save separately.
    
    Similar to VAD but also extracts background noise for analysis.
    
    Parameters
    ----------
    filepath : str
        Path to audio file
    output_dir : str, optional
        Output directory. If None, creates 'filepath_separated/'
    sr : int
        Sample rate
    method : str
        Separation method
    denoise : bool
        Apply denoising before separation
    denoise_method : str
        Denoising method
    save_both : bool
        Save both foreground and background
    **kwargs
        Additional parameters
        
    Returns
    -------
    dict
        Dictionary containing:
        - pathname: Original file
        - pathname_foreground: Path to foreground audio
        - pathname_background: Path to background audio
        - snr_db: Signal-to-noise ratio
        - duration_foreground: Duration of foreground
        - duration_background: Duration of background
    """
    # Load audio
    audio, sr_loaded = librosa.load(filepath, sr=sr, mono=True)
    
    # Optional denoising
    if denoise:
        audio = denoise_audio(audio, sr=sr, method=denoise_method)
    
    # Separate
    result = separate_foreground_background(audio, sr=sr, method=method, **kwargs)
    
    # Setup output directory
    if output_dir is None:
        from pathlib import Path
        base_dir = os.path.dirname(filepath)
        base_name = Path(filepath).stem
        output_dir = os.path.join(base_dir, f'{base_name}_separated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save foreground
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    foreground_path = os.path.join(output_dir, f'{base_filename}_foreground.wav')
    sf.write(foreground_path, result['foreground'], sr)
    
    # Save background
    background_path = None
    if save_both:
        background_path = os.path.join(output_dir, f'{base_filename}_background.wav')
        sf.write(background_path, result['background'], sr)
    
    output = {
        'pathname': filepath,
        'filename': os.path.basename(filepath),
        'pathname_foreground': foreground_path,
        'pathname_background': background_path,
        'snr_db': result['snr_db'],
        'duration_foreground': len(result['foreground']) / sr,
        'duration_background': len(result['background']) / sr
    }
    
    logger.info(f"Separated {os.path.basename(filepath)}: SNR = {result['snr_db']:.2f} dB")
    
    return output


def batch_separate_foreground_background(pathnames, output_dir=None, sr=16000,
                                        method='spectral_masking', denoise=True,
                                        **kwargs):
    """Batch separate foreground/background from multiple files.
    
    Parameters
    ----------
    pathnames : list
        List of audio file paths
    output_dir : str, optional
        Base output directory
    sr : int
        Sample rate
    method : str
        Separation method
    denoise : bool
        Apply denoising
    **kwargs
        Additional parameters
        
    Returns
    -------
    list
        List of result dictionaries
    """
    results = []
    
    for pathname in pathnames:
        try:
            result = extract_foreground_background_segments(
                pathname,
                output_dir=output_dir,
                sr=sr,
                method=method,
                denoise=denoise,
                **kwargs
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pathname}: {str(e)}")
            results.append({
                'pathname': pathname,
                'filename': os.path.basename(pathname),
                'pathname_foreground': None,
                'pathname_background': None,
                'snr_db': None,
                'error': str(e)
            })
    
    return results


def enhance_atc_audio(audio, sr=16000, method='aggressive'):
    """Enhance noisy ATC (Air Traffic Control) communication audio.
    
    Specialized enhancement for:
    - Radio static and interference
    - Wind noise
    - Engine noise
    - Low SNR conditions
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate (typically 8000 or 16000 for radio)
    method : str
        Enhancement level:
        - 'light': Minimal processing
        - 'moderate': Balanced enhancement
        - 'aggressive': Maximum enhancement
        
    Returns
    -------
    np.ndarray
        Enhanced audio
    """
    if method == 'light':
        # Band-pass filter for speech
        audio = _denoise_bandpass(audio, sr, lowcut=300, highcut=3400)
        
    elif method == 'moderate':
        # Band-pass + spectral subtraction
        audio = _denoise_bandpass(audio, sr, lowcut=300, highcut=3400)
        audio = _denoise_spectral_subtraction(audio, sr, stationary=False, alpha=1.5)
        
    elif method == 'aggressive':
        # Full processing chain
        # 1. High-pass to remove low-frequency rumble
        audio = _denoise_bandpass(audio, sr, lowcut=200, highcut=4000)
        
        # 2. Aggressive spectral subtraction
        audio = _denoise_spectral_subtraction(audio, sr, stationary=False, alpha=3.0, beta=0.05)
        
        # 3. Optional: noisereduce if available
        if NOISEREDUCE_AVAILABLE:
            audio = _denoise_noisereduce(audio, sr, stationary=False, prop_decrease=0.9)
    
    return audio
