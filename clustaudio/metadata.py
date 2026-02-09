"""
Metadata extraction module for audio files.

This module handles extraction of metadata from audio files including:
- ID3 tags (MP3)
- Vorbis comments (OGG, FLAC)
- MP4 metadata (M4A, AAC)
- Basic file properties (duration, bitrate, sample rate)

Author: AI Assistant
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to import mutagen for comprehensive metadata support
try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.oggvorbis import OggVorbis
    from mutagen.mp4 import MP4
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logger.warning("mutagen not available. Metadata extraction will be limited. Install with: pip install mutagen")

# Fallback to eyed3 for MP3 files
try:
    import eyed3
    EYED3_AVAILABLE = True
except ImportError:
    EYED3_AVAILABLE = False

# Fallback to tinytag for basic metadata
try:
    from tinytag import TinyTag
    TINYTAG_AVAILABLE = True
except ImportError:
    TINYTAG_AVAILABLE = False


def extract_metadata(filepath, logger=None):
    """Extract metadata from a single audio file.
    
    Parameters
    ----------
    filepath : str
        Path to audio file
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    dict
        Dictionary containing metadata fields:
        - filename: File name
        - pathname: Full path
        - artist: Artist name
        - album: Album name
        - title: Track title
        - genre: Genre
        - year: Release year
        - track_number: Track number
        - duration: Duration in seconds
        - bitrate: Bitrate in kbps
        - sample_rate: Sample rate in Hz
        - channels: Number of channels
        - filesize: File size in bytes
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    metadata = {
        'pathname': filepath,
        'filename': os.path.basename(filepath),
        'artist': None,
        'album': None,
        'title': None,
        'genre': None,
        'year': None,
        'track_number': None,
        'duration': None,
        'bitrate': None,
        'sample_rate': None,
        'channels': None,
        'filesize': None
    }
    
    # Check file exists
    if not os.path.isfile(filepath):
        logger.warning(f"File not found: {filepath}")
        return metadata
    
    # Get file size
    metadata['filesize'] = os.path.getsize(filepath)
    
    # Try mutagen first (most comprehensive)
    if MUTAGEN_AVAILABLE:
        try:
            audio = MutagenFile(filepath)
            if audio is not None:
                # Extract common tags
                metadata['artist'] = _get_tag(audio, ['artist', 'TPE1', '\xa9ART'])
                metadata['album'] = _get_tag(audio, ['album', 'TALB', '\xa9alb'])
                metadata['title'] = _get_tag(audio, ['title', 'TIT2', '\xa9nam'])
                metadata['genre'] = _get_tag(audio, ['genre', 'TCON', '\xa9gen'])
                metadata['year'] = _get_tag(audio, ['date', 'year', 'TDRC', '\xa9day'])
                metadata['track_number'] = _get_tag(audio, ['tracknumber', 'TRCK', 'trkn'])
                
                # Audio properties
                if hasattr(audio.info, 'length'):
                    metadata['duration'] = audio.info.length
                if hasattr(audio.info, 'bitrate'):
                    metadata['bitrate'] = audio.info.bitrate / 1000  # Convert to kbps
                if hasattr(audio.info, 'sample_rate'):
                    metadata['sample_rate'] = audio.info.sample_rate
                if hasattr(audio.info, 'channels'):
                    metadata['channels'] = audio.info.channels
                
                return metadata
        except Exception as e:
            logger.debug(f"Mutagen extraction failed for {filepath}: {str(e)}")
    
    # Fallback to TinyTag
    if TINYTAG_AVAILABLE:
        try:
            tag = TinyTag.get(filepath)
            metadata['artist'] = tag.artist
            metadata['album'] = tag.album
            metadata['title'] = tag.title
            metadata['genre'] = tag.genre
            metadata['year'] = tag.year
            metadata['track_number'] = tag.track
            metadata['duration'] = tag.duration
            metadata['bitrate'] = tag.bitrate
            metadata['sample_rate'] = tag.samplerate
            metadata['channels'] = tag.channels
            
            return metadata
        except Exception as e:
            logger.debug(f"TinyTag extraction failed for {filepath}: {str(e)}")
    
    # Last resort: eyed3 for MP3 files
    if EYED3_AVAILABLE and filepath.lower().endswith('.mp3'):
        try:
            audiofile = eyed3.load(filepath)
            if audiofile and audiofile.tag:
                metadata['artist'] = audiofile.tag.artist
                metadata['album'] = audiofile.tag.album
                metadata['title'] = audiofile.tag.title
                metadata['genre'] = str(audiofile.tag.genre) if audiofile.tag.genre else None
                
                if audiofile.tag.recording_date:
                    metadata['year'] = audiofile.tag.recording_date.year
                
                if audiofile.tag.track_num:
                    metadata['track_number'] = audiofile.tag.track_num[0]
            
            if audiofile and audiofile.info:
                metadata['duration'] = audiofile.info.time_secs
                metadata['bitrate'] = audiofile.info.bit_rate[1] / 1000 if audiofile.info.bit_rate else None
                metadata['sample_rate'] = audiofile.info.sample_freq
                
            return metadata
        except Exception as e:
            logger.debug(f"eyed3 extraction failed for {filepath}: {str(e)}")
    
    logger.warning(f"Could not extract metadata from {filepath}. Install mutagen or tinytag for better support.")
    return metadata


def _get_tag(audio, tag_names):
    """Helper to get tag value from multiple possible tag names.
    
    Parameters
    ----------
    audio : mutagen.File
        Mutagen audio object
    tag_names : list
        List of possible tag names to try
        
    Returns
    -------
    str or None
        Tag value if found
    """
    for tag_name in tag_names:
        try:
            if tag_name in audio:
                value = audio[tag_name]
                # Handle different return types
                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])
                elif isinstance(value, str):
                    return value
                else:
                    return str(value)
        except:
            continue
    return None


def extract_metadata_batch(pathnames, logger=None, max_workers=4):
    """Extract metadata from multiple audio files.
    
    Parameters
    ----------
    pathnames : list
        List of file paths
    logger : logging.Logger, optional
        Logger instance
    max_workers : int
        Number of parallel workers (currently not used, sequential processing)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metadata for all files
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting metadata from {len(pathnames)} files...")
    
    metadata_list = []
    
    for pathname in pathnames:
        metadata = extract_metadata(pathname, logger=logger)
        metadata_list.append(metadata)
    
    df = pd.DataFrame(metadata_list)
    
    # Report statistics
    fields_with_data = {
        col: df[col].notna().sum() 
        for col in ['artist', 'album', 'genre', 'year', 'duration']
    }
    
    logger.info(f"Metadata extraction complete:")
    for field, count in fields_with_data.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {field}: {count}/{len(df)} ({pct:.1f}%)")
    
    return df


def cluster_by_metadata(df, cluster_by=['artist', 'album'], similarity_threshold=0.8):
    """Create cluster labels based on metadata fields.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metadata
    cluster_by : list
        List of metadata fields to use for clustering
    similarity_threshold : float
        Threshold for fuzzy string matching (0-1)
        
    Returns
    -------
    np.ndarray
        Cluster labels
    """
    # Simple exact matching for now
    labels = np.zeros(len(df), dtype=int)
    
    # Create composite key from selected fields
    keys = []
    for idx, row in df.iterrows():
        key_parts = []
        for field in cluster_by:
            value = row.get(field, None)
            if value is not None and value != '':
                key_parts.append(str(value).lower().strip())
            else:
                key_parts.append('__unknown__')
        keys.append('|'.join(key_parts))
    
    # Assign labels based on unique keys
    unique_keys = {}
    label_counter = 0
    
    for i, key in enumerate(keys):
        if key not in unique_keys:
            unique_keys[key] = label_counter
            label_counter += 1
        labels[i] = unique_keys[key]
    
    return labels


def create_metadata_features(df, encode_categorical=True):
    """Convert metadata to numerical features for clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metadata
    encode_categorical : bool
        Whether to encode categorical variables
        
    Returns
    -------
    np.ndarray
        Feature matrix
    """
    features = []
    
    # Numerical features
    numerical_cols = ['duration', 'bitrate', 'sample_rate', 'year']
    
    # for col in numerical_cols:
    #     if col in df.columns:
    #         values = df[col].fillna(df[col].median()).values
    #         # Normalize
    #         if values.std() > 0:
    #             values = (values - values.mean()) / values.std()
    #         features.append(values.reshape(-1, 1))

    for col in numerical_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
    
            median = values.median()
            if pd.isna(median):
                values = values.fillna(0)   # fallback for all-NaN columns
            else:
                values = values.fillna(median)
    
            std = values.std()
            if std > 0:
                values = (values - values.mean()) / std
    
            features.append(values.values.reshape(-1, 1))


    # Categorical encoding (simple one-hot for now)
    if encode_categorical:
        categorical_cols = ['artist', 'album', 'genre']
        
        for col in categorical_cols:
            if col in df.columns:
                # Get top N most common values
                top_values = df[col].value_counts().head(20).index
                
                for value in top_values:
                    feature = (df[col] == value).astype(int).values
                    features.append(feature.reshape(-1, 1))
    
    if len(features) == 0:
        # No features extracted, return zeros
        return np.zeros((len(df), 1))
    
    # Concatenate all features
    feature_matrix = np.hstack(features)
    
    return feature_matrix
