"""
Audio Fingerprinting module for duplicate detection and similarity matching.

This module provides perceptual audio hashing for:
- Finding duplicate audio files
- Detecting similar audio (remixes, covers, different encodings)
- Fast similarity comparison

Author: AI Assistant
"""

import os
import logging
import numpy as np
import librosa
from scipy.spatial import distance
from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger(__name__)

# Try to import chromaprint/acoustid for production fingerprinting
try:
    import acoustid
    import chromaprint
    CHROMAPRINT_AVAILABLE = True
except ImportError:
    CHROMAPRINT_AVAILABLE = False
    logger.debug("chromaprint/acoustid not available. Using librosa-based fingerprinting. Install with: pip install pyacoustid")


def compute_fingerprint(audio=None, filepath=None, sr=11025, method='chroma_hash'):
    """Compute audio fingerprint.
    
    Parameters
    ----------
    audio : np.ndarray, optional
        Audio waveform
    filepath : str, optional
        Path to audio file (alternative to audio array)
    sr : int
        Sample rate (11025 recommended for fingerprinting)
    method : str
        Fingerprinting method:
        - 'chromaprint': Use Chromaprint (requires acoustid)
        - 'chroma_hash': Simple chroma-based hash (default)
        - 'spectral_hash': Spectral-based hash
        
    Returns
    -------
    np.ndarray or str
        Fingerprint hash
    """
    if method == 'chromaprint' and CHROMAPRINT_AVAILABLE:
        return _compute_chromaprint(filepath, sr=sr)
    elif method == 'chroma_hash':
        if audio is None and filepath is not None:
            audio, sr = librosa.load(filepath, sr=sr, mono=True)
        return _compute_chroma_hash(audio, sr=sr)
    elif method == 'spectral_hash':
        if audio is None and filepath is not None:
            audio, sr = librosa.load(filepath, sr=sr, mono=True)
        return _compute_spectral_hash(audio, sr=sr)
    else:
        logger.error(f"Unknown fingerprinting method: {method}")
        return None


def _compute_chromaprint(filepath, sr=11025):
    """Compute Chromaprint fingerprint."""
    if not CHROMAPRINT_AVAILABLE:
        logger.warning("Chromaprint not available, using chroma hash instead")
        audio, sr = librosa.load(filepath, sr=sr, mono=True)
        return _compute_chroma_hash(audio, sr=sr)
    
    try:
        # Compute fingerprint using acoustid
        duration, fingerprint = acoustid.fingerprint_file(filepath)
        
        # Convert to binary array for similarity comparison
        # Chromaprint returns encoded string, we'll use it as-is
        return fingerprint
        
    except Exception as e:
        logger.error(f"Chromaprint failed for {filepath}: {str(e)}")
        # Fallback to chroma hash
        audio, sr = librosa.load(filepath, sr=sr, mono=True)
        return _compute_chroma_hash(audio, sr=sr)


def _compute_chroma_hash(audio, sr=11025, n_chroma=12, n_segments=32):
    """Compute chroma-based perceptual hash.
    
    This creates a compact binary hash based on chroma features.
    Similar to image perceptual hashing but for audio.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    n_chroma : int
        Number of chroma bins
    n_segments : int
        Number of time segments for hash
        
    Returns
    -------
    np.ndarray
        Binary hash array
    """
    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, n_chroma=n_chroma)
    
    # Divide into segments
    segment_length = chroma.shape[1] // n_segments
    if segment_length == 0:
        segment_length = 1
    
    # Compute hash by comparing segments
    hash_bits = []
    
    for i in range(0, chroma.shape[1] - segment_length, segment_length):
        segment = chroma[:, i:i+segment_length]
        segment_mean = np.mean(segment, axis=1)
        
        # Binary encoding: 1 if above median, 0 otherwise
        median = np.median(segment_mean)
        bits = (segment_mean > median).astype(int)
        hash_bits.extend(bits)
    
    # Convert to numpy array
    fingerprint = np.array(hash_bits[:n_segments * n_chroma], dtype=np.uint8)
    
    return fingerprint


def _compute_spectral_hash(audio, sr=11025, n_mels=32, n_segments=32):
    """Compute spectral-based perceptual hash.
    
    Uses mel-spectrogram for fingerprinting.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio waveform
    sr : int
        Sample rate
    n_mels : int
        Number of mel bands
    n_segments : int
        Number of time segments
        
    Returns
    -------
    np.ndarray
        Binary hash array
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Divide into segments
    segment_length = mel_spec_db.shape[1] // n_segments
    if segment_length == 0:
        segment_length = 1
    
    hash_bits = []
    
    for i in range(0, mel_spec_db.shape[1] - segment_length, segment_length):
        segment = mel_spec_db[:, i:i+segment_length]
        segment_mean = np.mean(segment, axis=1)
        
        # Binary encoding
        median = np.median(segment_mean)
        bits = (segment_mean > median).astype(int)
        hash_bits.extend(bits)
    
    fingerprint = np.array(hash_bits[:n_segments * n_mels], dtype=np.uint8)
    
    return fingerprint


def compare_fingerprints(fp1, fp2, method='hamming'):
    """Compare two fingerprints for similarity.
    
    Parameters
    ----------
    fp1, fp2 : np.ndarray or str
        Fingerprints to compare
    method : str
        Distance metric:
        - 'hamming': Hamming distance (for binary hashes)
        - 'cosine': Cosine similarity
        - 'euclidean': Euclidean distance
        
    Returns
    -------
    float
        Similarity score (0 = identical, higher = more different)
    """
    # Handle chromaprint strings
    if isinstance(fp1, str) and isinstance(fp2, str):
        # Simple string comparison for chromaprint
        # In practice, chromaprint has its own comparison function
        return 0.0 if fp1 == fp2 else 1.0
    
    # Ensure arrays
    if not isinstance(fp1, np.ndarray):
        fp1 = np.array(fp1)
    if not isinstance(fp2, np.ndarray):
        fp2 = np.array(fp2)
    
    # Ensure same length
    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]
    
    if method == 'hamming':
        # Hamming distance (for binary hashes)
        return np.sum(fp1 != fp2) / len(fp1)
    elif method == 'cosine':
        # Cosine similarity
        return distance.cosine(fp1, fp2)
    elif method == 'euclidean':
        # Euclidean distance
        return distance.euclidean(fp1, fp2)
    else:
        logger.error(f"Unknown comparison method: {method}")
        return float('inf')


def find_duplicates(fingerprints, pathnames=None, threshold=0.1, method='hamming'):
    """Find duplicate or near-duplicate audio files.
    
    Parameters
    ----------
    fingerprints : list or np.ndarray
        List of fingerprints
    pathnames : list, optional
        List of file paths (for reference)
    threshold : float
        Similarity threshold (0-1, lower = more strict)
    method : str
        Comparison method
        
    Returns
    -------
    list
        List of duplicate groups. Each group is a list of indices.
    """
    if pathnames is None:
        pathnames = list(range(len(fingerprints)))
    
    n = len(fingerprints)
    duplicate_groups = []
    used_indices = set()
    
    for i in range(n):
        if i in used_indices:
            continue
        
        group = [i]
        
        for j in range(i + 1, n):
            if fingerprints[i] is not None and fingerprints[j] is not None:
                if j in used_indices:
                    continue
                
                # Compare fingerprints
                similarity = compare_fingerprints(fingerprints[i], fingerprints[j], method=method)
                
                if similarity <= threshold:
                    group.append(j)
                    used_indices.add(j)
        
        if len(group) > 1:
            duplicate_groups.append(group)
            used_indices.add(i)
    
    logger.info(f"Found {len(duplicate_groups)} groups of duplicates")
    
    return duplicate_groups


def batch_compute_fingerprints(pathnames, sr=11025, method='chroma_hash'):
    """Compute fingerprints for multiple audio files.
    
    Parameters
    ----------
    pathnames : list
        List of audio file paths
    sr : int
        Sample rate
    method : str
        Fingerprinting method
        
    Returns
    -------
    list
        List of fingerprints
    """
    fingerprints = []
    
    logger.info(f"Computing fingerprints for {len(pathnames)} files...")
    
    for i, pathname in enumerate(pathnames):
        try:
            fp = compute_fingerprint(filepath=pathname, sr=sr, method=method)
            fingerprints.append(fp)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(pathnames)} files")
                
        except Exception as e:
            logger.error(f"Failed to compute fingerprint for {pathname}: {str(e)}")
            # Add empty fingerprint
            fingerprints.append(None)
    
    logger.info(f"Fingerprinting complete")
    
    return fingerprints


def compute_fingerprint_matrix(fingerprints, method='hamming'):
    """Compute pairwise distance matrix for fingerprints.
    
    This is used for clustering and similarity search.
    
    Parameters
    ----------
    fingerprints : list
        List of fingerprints
        
    Returns
    -------
    np.ndarray
        Distance matrix (n_samples x n_samples)
    """
    n = len(fingerprints)
    dist_matrix = np.zeros((n, n))
    
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            if fingerprints[i] is not None and fingerprints[j] is not None:
                dist = compare_fingerprints(fingerprints[i], fingerprints[j], method=method)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
    
    return dist_matrix
