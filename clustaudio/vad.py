"""
Voice Activity Detection (VAD) module for audio segmentation.

This module provides functionality to:
- Detect speech/voice segments in audio
- Extract non-silent portions
- Split audio based on voice activity
- Save voice segments as separate files

Author: AI Assistant
"""

import os
import logging
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import webrtcvad for advanced VAD
try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logger.debug("webrtcvad not available. Using librosa-based VAD. Install with: pip install webrtcvad")


# %%
def detect_voice_segments(audio, sr=16000, 
                          method='energy',
                          energy_threshold=0.02,
                          min_silence_duration=0.3,
                          min_segment_duration=0.5,
                          frame_duration=0.02,
                          aggressiveness=2):
    """Detect voice/speech segments in audio.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate (webrtcvad requires 8000, 16000, 32000, or 48000)
    method : str
        VAD method: 'energy', 'webrtcvad', or 'librosa'
    energy_threshold : float
        Energy threshold for 'energy' method (0-1)
    min_silence_duration : float
        Minimum silence duration to split segments (seconds)
    min_segment_duration : float
        Minimum duration of valid segments (seconds)
    frame_duration : float
        Frame duration for webrtcvad (0.01, 0.02, or 0.03 seconds)
    aggressiveness : int
        WebRTC VAD aggressiveness (0-3, higher = more aggressive)
        
    Returns
    -------
    list
        List of tuples (start_time, end_time) for each voice segment
    """
    if isinstance(audio, list) and len(audio) > 1:
        logger.error('Segments can only be extracted one audio filepath at a time. <return>')
        return

    if method == 'webrtcvad' and WEBRTCVAD_AVAILABLE:
        segments = _detect_voice_webrtc(
            audio, sr, 
            frame_duration=frame_duration,
            aggressiveness=aggressiveness,
            min_silence_duration=min_silence_duration,
            min_segment_duration=min_segment_duration
        )
    elif method == 'librosa':
        segments = _detect_voice_librosa(
            audio, sr,
            min_silence_duration=min_silence_duration,
            min_segment_duration=min_segment_duration
        )
    else:
        # Energy-based method (simple but effective)
        segments = _detect_voice_energy(
            audio, sr,
            energy_threshold=energy_threshold,
            min_silence_duration=min_silence_duration,
            min_segment_duration=min_segment_duration
        )
    
    return segments


def _detect_voice_energy(audio, sr, energy_threshold=0.02, 
                        min_silence_duration=0.3,
                        min_segment_duration=0.5):
    """Simple energy-based VAD."""
    # Compute frame-level energy
    frame_length = int(0.02 * sr)  # 20ms frames
    hop_length = frame_length // 2
    
    # RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Normalize
    rms_norm = rms / (rms.max() + 1e-8)
    
    # Voice activity detection
    is_voice = rms_norm > energy_threshold
    
    # Convert frame indices to time
    frame_times = librosa.frames_to_time(np.arange(len(is_voice)), sr=sr, hop_length=hop_length)
    
    # Find segments
    segments = []
    in_segment = False
    segment_start = None
    
    min_silence_frames = int(min_silence_duration / (hop_length / sr))
    silence_counter = 0
    
    for i, (is_v, time) in enumerate(zip(is_voice, frame_times)):
        if is_v and not in_segment:
            # Start of new segment
            segment_start = time
            in_segment = True
            silence_counter = 0
        elif not is_v and in_segment:
            # Potential end of segment
            silence_counter += 1
            if silence_counter >= min_silence_frames:
                # End segment
                segment_end = time - min_silence_duration
                duration = segment_end - segment_start
                
                if duration >= min_segment_duration:
                    segments.append((segment_start, segment_end))
                
                in_segment = False
                segment_start = None
        elif is_v and in_segment:
            # Continue segment, reset silence counter
            silence_counter = 0
    
    # Handle final segment
    if in_segment and segment_start is not None:
        segment_end = frame_times[-1]
        duration = segment_end - segment_start
        if duration >= min_segment_duration:
            segments.append((segment_start, segment_end))
    
    return segments


def _detect_voice_librosa(audio, sr, min_silence_duration=0.3, min_segment_duration=0.5):
    """Use librosa's onset detection for VAD."""
    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Split audio at silent intervals
    intervals = librosa.effects.split(audio, top_db=20, frame_length=2048, hop_length=512)
    
    segments = []
    for start_sample, end_sample in intervals:
        start_time = start_sample / sr
        end_time = end_sample / sr
        duration = end_time - start_time
        
        if duration >= min_segment_duration:
            segments.append((start_time, end_time))
    
    return segments


def _detect_voice_webrtc(audio, sr, frame_duration=0.02, aggressiveness=2,
                        min_silence_duration=0.3, min_segment_duration=0.5):
    """Use WebRTC VAD for voice detection."""
    if not WEBRTCVAD_AVAILABLE:
        logger.warning("webrtcvad not available, falling back to energy-based VAD")
        return _detect_voice_energy(audio, sr, min_silence_duration=min_silence_duration)
    
    # WebRTC VAD requires specific sample rates
    if sr not in [8000, 16000, 32000, 48000]:
        # Resample
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32768).astype(np.int16)
    
    # Initialize VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # Frame parameters
    frame_duration_ms = int(frame_duration * 1000)
    frame_size = int(sr * frame_duration)
    
    # Process frames
    voiced_frames = []
    num_frames = len(audio_int16) // frame_size
    
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio_int16[start:end].tobytes()
        
        try:
            is_speech = vad.is_speech(frame, sr)
            voiced_frames.append(is_speech)
        except:
            voiced_frames.append(False)
    
    # Convert to segments
    segments = []
    in_segment = False
    segment_start = None
    silence_counter = 0
    min_silence_frames = int(min_silence_duration / frame_duration)
    
    for i, is_voice in enumerate(voiced_frames):
        time = i * frame_duration
        
        if is_voice and not in_segment:
            segment_start = time
            in_segment = True
            silence_counter = 0
        elif not is_voice and in_segment:
            silence_counter += 1
            if silence_counter >= min_silence_frames:
                segment_end = time - min_silence_duration
                duration = segment_end - segment_start
                
                if duration >= min_segment_duration:
                    segments.append((segment_start, segment_end))
                
                in_segment = False
                segment_start = None
        elif is_voice and in_segment:
            silence_counter = 0
    
    # Handle final segment
    if in_segment and segment_start is not None:
        segment_end = len(audio) / sr
        duration = segment_end - segment_start
        if duration >= min_segment_duration:
            segments.append((segment_start, segment_end))
    
    return segments


def extract_voice_segments(filepath, output_dir=None, sr=16000, method='energy', **vad_params):
    """Extract voice segments from an audio file and save them.
    
    Parameters
    ----------
    filepath : str
        Path to input audio file
    output_dir : str, optional
        Directory to save segments. If None, uses filepath directory + '_segments'
    sr : int
        Sample rate for processing
    method : str
        VAD method
    **vad_params
        Additional parameters for VAD
        
    Returns
    -------
    dict
        Dictionary containing:
        - pathname: Original file path
        - filename: Original filename
        - pathnames_segments: List of paths to saved segments
        - timestamps: List of (start, end) tuples
        - durations: List of segment durations
        - num_segments: Number of segments found
    """
    if isinstance(filepath, list) and len(filepath) > 1:
        logger.error('Segments can only be extracted one audio filepath at a time. <return>')
        return
    
    # Load audio
    audio, sr_loaded = librosa.load(filepath, sr=sr, mono=True)
    
    # Detect segments
    segments = detect_voice_segments(audio, sr=sr, method=method, **vad_params)
    
    # Setup output directory
    if output_dir is None:
        base_dir = os.path.dirname(filepath)
        base_name = Path(filepath).stem
        output_dir = os.path.join(base_dir, f'{base_name}_segments')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and save segments
    pathnames_segments = []
    durations = []
    
    base_filename = Path(filepath).stem
    
    for i, (start_time, end_time) in enumerate(segments):
        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment_audio = audio[start_sample:end_sample]
        
        # Save segment
        segment_filename = f'{base_filename}_segment_{i:03d}.wav'
        segment_path = os.path.join(output_dir, segment_filename)
        
        sf.write(segment_path, segment_audio, sr)
        
        pathnames_segments.append(segment_path)
        durations.append(end_time - start_time)
    
    result = {
        'pathname': filepath,
        'filename': os.path.basename(filepath),
        'pathnames_segments': pathnames_segments,
        'timestamps': segments,
        'durations': durations,
        'num_segments': len(segments)
    }
    
    logger.info(f"Extracted {len(segments)} voice segments from {os.path.basename(filepath)}")
    
    return result


def batch_extract_voice_segments(pathnames, output_dir=None, sr=16000, method='energy', **vad_params):
    """Extract voice segments from multiple audio files.
    
    Parameters
    ----------
    pathnames : list
        List of audio file paths
    output_dir : str, optional
        Base output directory
    sr : int
        Sample rate
    method : str
        VAD method
    **vad_params
        VAD parameters
        
    Returns
    -------
    list
        List of result dictionaries for each file
    """
    results = []
    
    for pathname in pathnames:
        try:
            result = extract_voice_segments(
                pathname, 
                output_dir=output_dir, 
                sr=sr, 
                method=method, 
                **vad_params
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pathname}: {str(e)}")
            results.append({
                'pathname': pathname,
                'filename': os.path.basename(pathname),
                'pathnames_segments': [],
                'timestamps': [],
                'durations': [],
                'num_segments': 0,
                'error': str(e)
            })
    
    return results
