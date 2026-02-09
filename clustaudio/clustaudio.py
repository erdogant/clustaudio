"""Python package clustaudio is for unsupervised clustering of audio files.

 Name        : clustaudio.py
 Author      : E.Taskesen (adapted for audio by AI assistant)
 Contact     : erdogant@gmail.com
 github      : https://github.com/erdogant/clustaudio
 Licence     : See licences

 Description : Audio clustering library for organizing, analyzing, and finding similar audio files
               using various feature extraction methods and clustering algorithms.
"""

# Core dependencies (kept from original)
from pca import pca
from clusteval import clusteval
from distfit import distfit
from scatterd import scatterd
import pypickle as pypickle
from ismember import ismember
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# Standard library imports
import os
import logging
import fnmatch
import matplotlib.pyplot as plt
from scipy.spatial import distance
import tempfile
import uuid
import shutil
import random

# Audio-specific imports (NEW)
try:
    import librosa
    import librosa.display
except ImportError:
    raise ImportError(
        "The 'librosa' library is not installed. Please install it using:\n"
        ">pip install librosa")

try:
    import soundfile as sf
except ImportError:
    raise ImportError(
        "The 'soundfile' library is not installed. Please install it using:\n"
        ">pip install soundfile")

# Optional: Advanced audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available. Some audio format conversions may be limited.")

# Optional: Audio fingerprinting
try:
    import acoustid
    import chromaprint
    FINGERPRINT_AVAILABLE = True
except ImportError:
    FINGERPRINT_AVAILABLE = False
    logging.warning("acoustid/chromaprint not available. Audio fingerprinting will be disabled.")

try:
    import clustaudio.vad as vad
    import clustaudio.separation as separation
    import clustaudio.visualization as visualization
    import clustaudio.fingerprint as fingerprint
    import clustaudio.metadata as metadata
except ImportError:
    import vad as vad
    import separation as separation
    import visualization as visualization
    import fingerprint as fingerprint
    import metadata as metadata

logger = logging.getLogger(__name__)

#%%
class Clustaudio():
    """Clustering of audio files.

    Clustering audio files through pre-processing, feature extraction,
    embedding and cluster evaluation. This library enables:

    - Organizing large audio collections by similarity
    - Finding duplicate or similar audio files
    - Genre/mood-based clustering
    - Voice segment extraction
    - Playlist generation from clusters

    The following steps are taken:

    * Step 1. Pre-processing:
        Audio files are imported with specific extensions (['wav', 'mp3', 'flac', 'ogg', 'm4a']).
        Each audio file is loaded with a target sample rate and can be converted to mono.
        Audio can be normalized and trimmed for silence.

    * Step 2. Feature extraction:
        Features are extracted using various methods:
        - 'mfcc': Mel-Frequency Cepstral Coefficients (default)
        - 'spectral': Spectral features (centroid, rolloff, contrast, bandwidth)
        - 'chroma': Chromagram features for pitch/harmony analysis
        - 'mfcc-spectral': Combined MFCC and spectral features
        - 'tempogram': Tempo and rhythm features
        - 'embeddings': Deep learning audio embeddings (VGGish, YAMNet, OpenL3)
        - 'fingerprint': Audio fingerprinting for duplicate detection
        - 'metadata': Use audio metadata (ID3 tags) for clustering

    * Step 3. Embedding:
        The feature space is transformed non-linearly using t-SNE or UMAP.
        The embedding coordinates are used for visualization.

    * Step 4. Cluster evaluation:
        The feature space is used for cluster evaluation to determine
        the optimal number of clusters using silhouette, dbindex, or derivative methods.

    * Step 5. Results and visualization:
        Results are stored and can be visualized using waveforms, spectrograms,
        scatter plots, and cluster-based audio playback.

    Parameters
    ----------
    method : str, (default: 'mfcc')
        Method to extract features from audio files:
            * 'mfcc' : Mel-Frequency Cepstral Coefficients
            * 'spectral' : Spectral features (centroid, rolloff, contrast, bandwidth)
            * 'chroma' : Chromagram features
            * 'mfcc-spectral' : Combined MFCC and spectral features
            * 'tempogram' : Tempo and rhythm features
            * 'mel-spectrogram' : Mel-scaled spectrogram
            * 'embeddings' : Deep learning embeddings (requires additional setup)
            * 'fingerprint': Audio fingerprinting for duplicates
            * 'metadata': Cluster by audio metadata (artist, album, genre, etc.)
            * None : Use raw audio features

    embedding : str, (default: 'tsne')
        Perform embedding on extracted features for visualization:
            * 'tsne' : t-SNE embedding
            * 'umap' : UMAP embedding
            * None : Use first two features for plotting

    sr : int, (default: 22050)
        Target sample rate for audio loading (Hz).
        Common values: 22050, 44100, 48000

    duration : float or None, (default: None)
        Maximum duration to load from each audio file (seconds).
        None = load entire file
        Setting a value (e.g., 30) loads only first N seconds.

    mono : bool, (default: True)
        Convert audio to mono (single channel).
        False: Keep stereo/multi-channel audio

    normalize : bool, (default: True)
        Normalize audio amplitude to [-1, 1] range.

    trim_silence : bool, (default: False)
        trim_silence audio silence

    dirpath : str, (default: 'clustaudio')
        Directory to write temporary files and extracted segments.

    ext : list, (default: ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'])
        Audio file extensions to process.

    params_pca : dict, default: {'n_components':0.95, 'detect_outliers':None}
        Parameters for PCA dimensionality reduction (used with some methods).

    params_mfcc : dict, default: {'n_mfcc':13, 'n_fft':2048, 'hop_length':512}
        Parameters for MFCC feature extraction:
            * 'n_mfcc': Number of MFCC coefficients (typically 13-40)
            * 'n_fft': FFT window size
            * 'hop_length': Number of samples between successive frames
            * 'n_mels': Number of mel bands (default: 128)
            * 'fmin': Minimum frequency (Hz, default: 0)
            * 'fmax': Maximum frequency (Hz, default: None = sr/2)

    params_spectral : dict, default: {}
        Parameters for spectral feature extraction (uses librosa defaults).

    params_chroma : dict, default: {'n_chroma':12}
        Parameters for chroma feature extraction:
            * 'n_chroma': Number of chroma bins (default: 12)

    params_fingerprint : dict, default: {'algorithm':'chromaprint'}
        Parameters for audio fingerprinting:
            * 'algorithm': 'chromaprint' or other fingerprinting method

    params_metadata : dict, default: {'cluster_by':['artist', 'album', 'genre']}
        Parameters for metadata-based clustering:
            * 'cluster_by': List of metadata fields to use
            * 'similarity_threshold': Threshold for string matching (0-1)

    use_audio_cache : bool (Default: True)
        Cache extracted features to disk for faster reprocessing.

    use_spectrogram_cache : bool (Default: True)
        Cache computed spectrograms to speed up visualization.

    ncores : int, (default: 4)
        Number of CPU cores for parallel processing.

    verbose : int or str, (default: 'info')
        Logging level. Options: 'debug', 'info', 'warning', 'error'
        Or numeric: 10 (debug), 20 (info), 30 (warning), 40 (error)

    Returns
    -------
    Object containing:
        results : dict
            Dictionary with clustering results:
                * feat : Extracted features array
                * xycoord : 2D embedding coordinates
                * pathnames : List of file paths
                * filenames : List of file names
                * labels : Cluster labels
                * sample_rates : Sample rates of loaded files
                * durations : Duration of each file

    Examples
    --------
    >>> from clustaudio import Clustaudio
    >>>
    >>> # Initialize with MFCC features
    >>> cl = Clustaudio(method='mfcc', sr=22050)
    >>>
    >>> # Load example audio dataset
    >>> X = cl.import_example(data='various')
    >>>
    >>> # Cluster audio files
    >>> results = cl.fit_transform(X)
    >>>
    >>> # Visualize results
    >>> cl.clusteval.scatter(s=350)
    >>> cl.plot(labels=[0, 1, 2], show_waveform=True, show_spectrogram=True)
    >>>
    >>> # Find similar audio
    >>> results_find = cl.find('./clustaudio/test/audio/various/dog_barking_1.wav', k=2)
    >>> cl.plot_find()
    >>>
    >>> # Organize files by cluster
    >>> cl.move_to_dir(create_playlists=True)
    >>>
    """

    def __init__(self,
                 method='mfcc',
                 embedding='tsne',
                 sr=22050,
                 duration=None,
                 mono=True,
                 normalize=True,
                 trim_silence=False,
                 dirpath=None,
                 use_audio_cache=False,
                 use_spectrogram_cache=True,
                 tempdir=None,
                 ext=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
                 params_pca={'n_components': 0.95},
                 params_mfcc={'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512, 'n_mels': 128},
                 params_spectral={},
                 params_chroma={'n_chroma': 12},
                 params_fingerprint={'algorithm': 'chromaprint'},
                 params_metadata={'cluster_by': ['artist', 'album', 'genre'], 'similarity_threshold': 0.8},
                 ncores=4,
                 verbose='info',
                 ):
        """Initialize clustaudio with user-defined parameters."""
        # Set logger
        set_logger(verbose=verbose)

        # Clean any previous fitted models
        self.clean_init()
        os.environ["LOKY_MAX_CPU_COUNT"] = str(ncores)

        # Validate method
        valid_methods = [None, 'mfcc', 'spectral', 'chroma', 'mfcc-spectral',
                        'tempogram', 'mel-spectrogram', 'embeddings',
                        'fingerprint', 'metadata']
        if method not in valid_methods:
            raise Exception(logger.error(f'method: "{method}" is unknown. Valid methods: {valid_methods}'))

        # Check fingerprinting availability
        if method == 'fingerprint' and not FINGERPRINT_AVAILABLE:
            logger.warning('Fingerprinting requires acoustid/chromaprint. Install with: pip install pyacoustid')
            logger.warning('Falling back to mfcc method.')
            method = 'mfcc'

        # Validate sample rate
        if sr not in [8000, 11025, 16000, 22050, 44100, 48000]:
            logger.warning(f'Unusual sample rate: {sr} Hz. Common values: 22050, 44100, 48000')

        # Validate duration
        if duration is not None and duration <= 0:
            logger.error('duration must be positive or None')
            duration = None

        # Store core parameters
        self.params = {}
        self.params['method'] = method
        self.params['embedding'] = embedding
        self.params['sr'] = sr
        self.params['duration'] = duration
        self.params['mono'] = mono
        self.params['normalize'] = normalize
        self.params['trim_silence'] = trim_silence

        # Setup directories
        self.params['dirpath'] = _set_tempdir(dirpath, show_logger=True)
        self.params['tempdir'] = _set_tempdir(tempdir, show_logger=False)
        self.params['filepath'] = os.path.join(_set_tempdir(None, show_logger=False), 'clustaudio.pkl')
        self.params['ext'] = ext
        self.params['use_audio_cache'] = use_audio_cache
        self.params['use_spectrogram_cache'] = use_spectrogram_cache

        # Feature extraction parameters
        # MFCC parameters
        mfcc_defaults = {'n_mfcc': 13, 'n_fft': 2048, 'hop_length': 512, 'n_mels': 128, 'fmin': 0, 'fmax': None}
        params_mfcc = {**mfcc_defaults, **params_mfcc}
        self.params_mfcc = params_mfcc

        # Spectral parameters
        spectral_defaults = {'n_fft': 2048, 'hop_length': 512}
        params_spectral = {**spectral_defaults, **params_spectral}
        self.params_spectral = params_spectral

        # Chroma parameters
        chroma_defaults = {'n_chroma': 12, 'n_fft': 2048, 'hop_length': 512}
        params_chroma = {**chroma_defaults, **params_chroma}
        self.params_chroma = params_chroma

        # PCA parameters (for dimensionality reduction)
        pca_defaults = {'n_components': 0.95, 'detect_outliers': None, 'random_state': None}
        params_pca = {**pca_defaults, **params_pca}
        self.params_pca = params_pca

        # Fingerprint parameters
        fingerprint_defaults = {'algorithm': 'chromaprint'}
        params_fingerprint = {**fingerprint_defaults, **params_fingerprint}
        self.params_fingerprint = params_fingerprint

        # Metadata parameters
        metadata_defaults = {'cluster_by': ['artist', 'album', 'genre'],
                           'similarity_threshold': 0.8,
                           'use_duration': False,
                           'use_bitrate': False}
        params_metadata = {**metadata_defaults, **params_metadata}
        self.params_metadata = params_metadata

        # Track if find functionality is being used
        self.find_func = False
        logger.info(f'Initialized clustaudio with method={method}, sr={sr}Hz, mono={mono}')

        # Add functionalities
        self.separation = separation
        self.vad = vad
        self.viz = visualization
        self.fingerprint = fingerprint
        self.metadata = metadata



    def import_example(self, data='wave', n=10, freq=440, duration=1.0, sr=22050, amplitude=0.5):
        return import_example(data=data, n=n, freq=freq, duration=duration, sr=sr, amplitude=amplitude)
    

    def check_verbosity(self):
        """Check the verbosity level."""
        logger.debug('DEBUG')
        logger.info('INFO')
        logger.warning('WARNING')
        logger.critical('CRITICAL')

    def audioread(self, filepath, sr=None, duration=None, offset=0.0,
                  mono=None, normalize=None, return_success=False):
        """Read and preprocess audio file.

        This function loads audio files and applies preprocessing steps:
        1. Load audio with target sample rate
        2. Convert to mono (optional)
        3. Normalize amplitude (optional)

        Parameters
        ----------
        filepath : str
            Path to audio file
        sr : int or None
            Target sample rate (Hz). If None, uses self.params['sr']
        duration : float or None
            Duration to load (seconds). If None, uses self.params['duration']
        offset : float
            Start reading after offset (seconds)
        mono : bool or None
            Convert to mono. If None, uses self.params['mono']
        normalize : bool or None
            Normalize amplitude. If None, uses self.params['normalize']
        return_success : bool
            If True, return (audio, success) tuple

        Returns
        -------
        audio : np.ndarray
            Audio waveform, shape (n_samples,) for mono or (n_channels, n_samples) for stereo
        success : bool (only if return_success=True)
            Whether the audio was successfully loaded

        Examples
        --------
        >>> from clustaudio import Clustaudio
        >>>
        >>> cl = Clustaudio()
        >>> audio = cl.audioread('audio.wav', sr=22050, mono=True)
        >>> print(audio.shape)
        """
        # Use instance defaults if not specified
        if sr is None:
            sr = self.params['sr']
        if duration is None:
            duration = self.params['duration']
        if mono is None:
            mono = self.params['mono']
        if normalize is None:
            normalize = self.params['normalize']

        audio = np.array([])
        success = False

        try:
            # Load audio using librosa
            audio, sr_loaded = librosa.load(
                filepath,
                sr=sr,
                duration=duration,
                offset=offset,
                mono=mono
            )

            # Normalize if requested
            if normalize and len(audio) > 0:
                audio = librosa.util.normalize(audio)

            success = True
            logger.debug(f'Successfully loaded: {filepath} (sr={sr_loaded}Hz, duration={len(audio)/sr_loaded:.2f}s)')

        except Exception as e:
            logger.warning(f'Could not read: [{filepath}]: {str(e)}')
            audio = np.array([])
            success = False

        if return_success:
            return audio, success
        return audio

    def preprocessing(self, pathnames, sr=None, duration=None, mono=None,
                     normalize=None, trim_silence=False, flatten=False):
        """Preprocess audio files and return consistent output.

        This function handles multiple input types and applies preprocessing:
        1. Load audio files
        2. Convert to target sample rate
        3. Convert to mono (optional)
        4. Trim silence (optional)
        5. Normalize amplitude (optional)

        Parameters
        ----------
        pathnames : str, list, or np.ndarray
            Input audio files:
                * str: Path to directory or single file
                * list: List of file paths
                * np.ndarray: Pre-loaded audio array
        sr : int or None
            Target sample rate
        duration : float or None
            Maximum duration per file (seconds)
        mono : bool or None
            Convert to mono
        normalize : bool or None
            Normalize amplitude
        trim_silence : bool
            Remove leading/trailing silence
        flatten : bool
            Flatten audio to 1D (for compatibility, typically False for audio)

        Returns
        -------
        dict
            Dictionary containing:
                * 'audio': List of audio arrays
                * 'pathnames': List of file paths
                * 'filenames': List of filenames
                * 'sample_rates': List of sample rates
                * 'durations': List of durations (seconds)
        """
        # Use instance defaults
        if sr is None:
            sr = self.params['sr']
        if duration is None:
            duration = self.params['duration']
        if mono is None:
            mono = self.params['mono']
        if normalize is None:
            normalize = self.params['normalize']
        if trim_silence is None:
            trim_silence = self.params['trim_silence']

        # Ensure pathnames is a list
        if isinstance(pathnames, str):
            pathnames = [pathnames]

        # Filter valid file paths
        pathnames = np.array(pathnames)
        pathnames = pathnames[pathnames != None]
        pathnames = list(pathnames[list(map(os.path.isfile, pathnames))])
        filenames = list(map(os.path.basename, pathnames))

        # Initialize output
        out = {
            'audio': [],
            'pathnames': pathnames,
            'filenames': filenames,
            'sample_rates': [],
            'durations': []
        }

        if len(pathnames) == 0:
            logger.warning('No valid audio files found')
            return out

        logger.info(f"Preprocessing {len(pathnames)} audio files...")

        # Load and preprocess each audio file
        audio_list = []
        valid_indices = []

        for i, pathname in enumerate(tqdm(pathnames, disable=disable_tqdm(), desc='[clustaudio]')):
            audio, success = self.audioread(
                pathname,
                sr=sr,
                duration=duration,
                mono=mono,
                normalize=normalize,
                return_success=True
            )

            if success and len(audio) > 0:
                # Trim silence if requested
                if trim_silence:
                    audio, _ = librosa.effects.trim(audio, top_db=20)

                # Store results
                audio_list.append(audio)
                valid_indices.append(i)

                # Calculate duration
                audio_duration = len(audio) / sr
                out['sample_rates'].append(sr)
                out['durations'].append(audio_duration)

        # Filter to only valid files
        if len(valid_indices) < len(pathnames):
            logger.info(f"[{len(pathnames) - len(valid_indices)}] audio file(s) could not be read and are excluded.")
            out['pathnames'] = [pathnames[i] for i in valid_indices]
            out['filenames'] = [filenames[i] for i in valid_indices]

        out['audio'] = audio_list

        logger.info(f"Successfully preprocessed {len(audio_list)} audio files")

        return out

    def extract_mfcc(self, audio, sr=None, **kwargs):
        """Extract MFCC (Mel-Frequency Cepstral Coefficients) features.

        MFCCs are widely used audio features that represent the short-term
        power spectrum of sound, commonly used in speech and music analysis.

        Parameters
        ----------
        audio : np.ndarray or list
            Audio waveform(s). Can be single array or list of arrays.
        sr : int or None
            Sample rate. If None, uses self.params['sr']
        **kwargs : dict
            Additional parameters for librosa.feature.mfcc

        Returns
        -------
        np.ndarray
            MFCC features. For single audio: shape (n_features,)
            For multiple: shape (n_samples, n_features)
            Features include mean and std of each MFCC coefficient.

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>> audio, sr = librosa.load('audio.wav')
        >>> mfcc_features = cl.extract_mfcc(audio, sr)
        """
        if sr is None:
            sr = self.params['sr']

        # Get MFCC parameters
        mfcc_params = {**self.params_mfcc, **kwargs}

        # Handle single audio vs list of audios
        if isinstance(audio, list):
            features = []
            for aud in tqdm(audio, disable=disable_tqdm(), desc='[clustaudio] Extracting MFCC'):
                feat = self._extract_mfcc_single(aud, sr, **mfcc_params)
                features.append(feat)
            return np.vstack(features)
        else:
            return self._extract_mfcc_single(audio, sr, **mfcc_params)

    def _extract_mfcc_single(self, audio, sr, **mfcc_params):
        """Extract MFCC features from a single audio signal."""
        try:
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, **mfcc_params)

            # Compute statistics: mean and std across time
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            # Concatenate mean and std
            features = np.concatenate([mfcc_mean, mfcc_std])

            return features
        except Exception as e:
            logger.error(f'Error extracting MFCC: {str(e)}')
            # Return zeros if extraction fails
            n_features = mfcc_params.get('n_mfcc', 13) * 2
            return np.zeros(n_features)

    def extract_spectral(self, audio, sr=None, **kwargs):
        """Extract spectral features from audio.

        Extracts multiple spectral features that characterize the frequency
        content and distribution of audio:
        - Spectral centroid (brightness)
        - Spectral rolloff (frequency below which X% of energy is contained)
        - Spectral contrast (difference between peaks and valleys)
        - Spectral bandwidth (spread of frequencies)
        - Zero crossing rate (percussiveness)
        - RMS energy (loudness)

        Parameters
        ----------
        audio : np.ndarray or list
            Audio waveform(s)
        sr : int or None
            Sample rate
        **kwargs : dict
            Additional parameters for spectral feature extraction

        Returns
        -------
        np.ndarray
            Spectral features with mean and std for each feature type

        Examples
        --------
        >>> cl = Clustaudio(method='spectral')
        >>> audio = cl.audioread('audio.wav')
        >>> spectral_features = cl.extract_spectral(audio)
        """
        if sr is None:
            sr = self.params['sr']

        # Handle single audio vs list
        if isinstance(audio, list):
            features = []
            for aud in tqdm(audio, disable=disable_tqdm(), desc='[clustaudio] Extracting Spectral'):
                feat = self._extract_spectral_single(aud, sr, **kwargs)
                features.append(feat)
            return np.vstack(features)
        else:
            return self._extract_spectral_single(audio, sr, **kwargs)

    def _extract_spectral_single(self, audio, sr, **spectral_params):
        """Extract spectral features from a single audio signal."""
        try:
            features = []

            # Spectral centroid (center of mass of spectrum)
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, **spectral_params)[0]
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])

            # Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, **spectral_params)[0]
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

            # Spectral contrast (difference between peaks and valleys in spectrum)
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio, sr=sr, **spectral_params)
            # Spectral contrast returns multiple bands, take mean and std of all
            features.extend([np.mean(spectral_contrast), np.std(spectral_contrast)])

            # Spectral bandwidth (spread around spectral centroid)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sr, **spectral_params)[0]
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

            # Zero crossing rate (number of times signal crosses zero)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.extend([np.mean(zcr), np.std(zcr)])

            # RMS energy (root mean square energy)
            rms = librosa.feature.rms(y=audio)[0]
            features.extend([np.mean(rms), np.std(rms)])

            return np.array(features)

        except Exception as e:
            logger.error(f'Error extracting spectral features: {str(e)}')
            # Return zeros if extraction fails (12 features total: 6 types * 2 stats)
            return np.zeros(12)

    def extract_chroma(self, audio, sr=None, **kwargs):
        """Extract chromagram features.

        Chroma features represent the 12 different pitch classes
        (C, C#, D, ..., B), useful for music analysis and chord recognition.

        Parameters
        ----------
        audio : np.ndarray or list
            Audio waveform(s)
        sr : int or None
            Sample rate
        **kwargs : dict
            Additional parameters for chroma extraction

        Returns
        -------
        np.ndarray
            Chroma features (mean and std for each of 12 pitch classes)

        Examples
        --------
        >>> cl = Clustaudio(method='chroma')
        >>> audio = cl.audioread('music.mp3')
        >>> chroma_features = cl.extract_chroma(audio)
        """
        if sr is None:
            sr = self.params['sr']

        chroma_params = {**self.params_chroma, **kwargs}

        # Handle single audio vs list
        if isinstance(audio, list):
            features = []
            for aud in tqdm(audio, disable=disable_tqdm(), desc='[clustaudio] Extracting Chroma'):
                feat = self._extract_chroma_single(aud, sr, **chroma_params)
                features.append(feat)
            return np.vstack(features)
        else:
            return self._extract_chroma_single(audio, sr, **chroma_params)

    def _extract_chroma_single(self, audio, sr, **chroma_params):
        """Extract chroma features from a single audio signal."""
        try:
            # Extract chromagram
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, **chroma_params)

            # Compute mean and std for each chroma bin
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)

            # Concatenate
            features = np.concatenate([chroma_mean, chroma_std])

            return features

        except Exception as e:
            logger.error(f'Error extracting chroma features: {str(e)}')
            n_chroma = chroma_params.get('n_chroma', 12)
            return np.zeros(n_chroma * 2)

    def extract_feat(self, Xraw):
        """Extract features based on the selected method.

        This is the main feature extraction dispatcher that calls the
        appropriate extraction method based on self.params['method'].

        Parameters
        ----------
        Xraw : dict
            Dictionary containing preprocessed audio data with keys:
                * 'audio': List of audio arrays
                * 'pathnames': List of file paths
                * 'filenames': List of filenames

        Returns
        -------
        np.ndarray
            Extracted features array of shape (n_samples, n_features)

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>> pathnames = ['audio1.wav', 'audio2.wav']
        >>> X = cl.preprocessing(pathnames)
        >>> features = cl.extract_feat(X)
        """
        logger.info(f"Extracting features using [{self.params['method']}] method...")

        method = self.params['method']
        audio_list = Xraw['audio']

        if audio_list is None or len(audio_list) == 0:
            logger.warning('No audio data to extract features from')
            return np.array([]).reshape(0, 0)

        # Extract features based on method
        if method == 'mfcc':
            Xfeat = self.extract_mfcc(audio_list)

        elif method == 'spectral':
            Xfeat = self.extract_spectral(audio_list)

        elif method == 'chroma':
            Xfeat = self.extract_chroma(audio_list)

        elif method == 'mfcc-spectral':
            # Combined features
            logger.info('Extracting combined MFCC and spectral features')
            mfcc_feat = self.extract_mfcc(audio_list)
            spectral_feat = self.extract_spectral(audio_list)
            Xfeat = np.hstack([mfcc_feat, spectral_feat])

        elif method == 'tempogram':
            # Tempo and rhythm features
            logger.info('Extracting tempogram features')
            Xfeat = self._extract_tempogram(audio_list)

        elif method == 'mel-spectrogram':
            # Mel-spectrogram features
            logger.info('Extracting mel-spectrogram features')
            Xfeat = self._extract_mel_spectrogram(audio_list)

        elif method == 'fingerprint':
            # Audio fingerprinting
            logger.info('Extracting audio fingerprints')
            Xfeat = self._extract_fingerprint(Xraw['pathnames'])

        elif method == 'metadata':
            # Metadata-based features
            logger.info('Extracting metadata features')
            Xfeat = self._extract_metadata(Xraw['pathnames'])

        elif method is None:
            # Raw audio features (averaged over time)
            logger.info('Using raw audio features (time-averaged)')
            Xfeat = self._extract_raw_features(audio_list)
        else:
            logger.error(f'Unknown method: {method}')
            return np.array([]).reshape(0, 0)

        # Store results if not in find mode
        if not self.find_func:
            self.results['feat'] = Xfeat

        if Xfeat is not None and Xfeat.size > 0:
            logger.info(f"Extracted features: samples={Xfeat.shape[0]}, features={Xfeat.shape[1]}")

        return Xfeat

    def _extract_tempogram(self, audio_list):
        """Extract tempo and rhythm features."""
        features = []
        for audio in tqdm(audio_list, disable=disable_tqdm(), desc='[clustaudio] Extracting Tempo'):
            try:
                # Extract tempo
                tempo, beats = librosa.beat.beat_track(y=audio, sr=self.params['sr'])

                # Compute tempogram
                tempogram = librosa.feature.tempogram(y=audio, sr=self.params['sr'])

                # Aggregate features
                feat = np.array([
                    tempo,
                    np.mean(tempogram),
                    np.std(tempogram),
                    len(beats) / (len(audio) / self.params['sr'])  # beats per second
                ])
                features.append(feat)
            except:
                features.append(np.zeros(4))

        return np.vstack(features)

    def _extract_mel_spectrogram(self, audio_list):
        """Extract mel-spectrogram features."""
        features = []
        for audio in tqdm(audio_list, disable=disable_tqdm(), desc='[clustaudio] Extracting Mel-Spec'):
            try:
                # Compute mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.params['sr'],
                    n_mels=self.params_mfcc['n_mels']
                )

                # Convert to dB scale
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Aggregate: mean and std across time for each mel band
                mel_mean = np.mean(mel_spec_db, axis=1)
                mel_std = np.std(mel_spec_db, axis=1)

                feat = np.concatenate([mel_mean, mel_std])
                features.append(feat)
            except:
                n_mels = self.params_mfcc['n_mels']
                features.append(np.zeros(n_mels * 2))

        return np.vstack(features)

    def _extract_fingerprint(self, pathnames):
        """Extract audio fingerprints for duplicate detection."""
        if not FINGERPRINT_AVAILABLE:
            logger.error('Fingerprinting requires acoustid. Install: pip install pyacoustid')
            return np.array([]).reshape(len(pathnames), 0)

        # Placeholder - actual implementation would use chromaprint
        logger.warning('Fingerprint extraction not fully implemented yet')
        return np.random.rand(len(pathnames), 32)  # Placeholder

    def _extract_metadata(self, pathnames):
        """Extract features from audio metadata."""
        logger.warning('Metadata extraction not fully implemented yet')
        # Batch extraction
        df = metadata.extract_metadata_batch(pathnames)
        # Convert to numerical features for ML
        features = metadata.create_metadata_features(df)
        return features

    def _extract_raw_features(self, audio_list):
        """Extract raw audio features by averaging over time."""
        features = []
        target_length = 1000  # Fixed feature length

        for audio in tqdm(audio_list, disable=disable_tqdm(), desc='[clustaudio] Raw features'):
            # Resample/pad to fixed length
            if len(audio) > target_length:
                audio_resampled = librosa.resample(audio, orig_sr=len(audio), target_sr=target_length)
            else:
                audio_resampled = np.pad(audio, (0, target_length - len(audio)))

            features.append(audio_resampled[:target_length])

        return np.vstack(features)

    def import_data(self, Xraw, flatten=False, black_list=None, recursive=True):
        """Import audio files and return in a consistent format.

        This function handles multiple input types and returns a standardized
        dictionary structure. Supports:
        - Directory paths (scans for audio files)
        - List of file paths
        - Pre-loaded audio arrays

        Parameters
        ----------
        Xraw : str, list, or np.ndarray
            Input can be:
                * "c://music//" : Path to directory with audio files
                * ['audio1.wav', 'audio2.mp3', ...] : List of file paths
                * [[.., ..], ...] : np.array of audio data
        flatten : bool
            Kept for API compatibility, typically False for audio
        black_list : list
            Exclude directories from processing (e.g., ['temp', 'cache'])
        recursive : bool
            Scan subdirectories recursively

        Returns
        -------
        dict
            Dictionary with keys:
                * 'audio': List of audio arrays
                * 'pathnames': List of file paths
                * 'filenames': List of filenames
                * 'sample_rates': List of sample rates
                * 'durations': List of durations
        """
        # Check input validity
        if isinstance(Xraw, str) and not os.path.exists(Xraw):
            logger.error(f'Path does not exist: {Xraw}')
            return {'audio': [], 'pathnames': [], 'filenames': [], 'sample_rates': [], 'durations': []}

        # Handle directory input
        if isinstance(Xraw, str) and os.path.isdir(Xraw):
            Xraw = listdir(Xraw, ext=self.params['ext'], black_list=black_list, recursive=recursive)

        # For metadata method, just store pathnames
        if self.params['method'] == 'metadata':
            # Batch extraction
            # df = metadata.extract_metadata_batch(Xraw)
            self.results['audio'] = 'metadata'
            self.results['pathnames'] = Xraw
            self.results['filenames'] = list(map(os.path.basename, Xraw))
            return self.results

        logger.info(f"Importing {len(Xraw) if isinstance(Xraw, list) else 'data'}...")

        # Handle single file
        if isinstance(Xraw, str) and os.path.isfile(Xraw):
            Xraw = [Xraw]

        # Handle list of paths
        if isinstance(Xraw, list) and all(isinstance(x, str) for x in Xraw):
            # Flatten any nested lists
            if isinstance(Xraw, list) and any(isinstance(item, (list)) for item in Xraw):
                Xraw = np.vstack(Xraw)

            # Preprocess audio files
            X = self.preprocessing(
                Xraw['pathnames'] if isinstance(Xraw, dict) else Xraw,
                sr=self.params['sr'],
                duration=self.params['duration'],
                mono=self.params['mono'],
                normalize=self.params['normalize'],
                trim_silence=self.params['trim_silence'],
            )

            # Return or store
            if self.find_func:
                return X
            else:
                defaults = self.results
                self.results = X
                self.results = {**defaults, **self.results}

        elif isinstance(Xraw, np.ndarray) or all(isinstance(x, np.ndarray) for x in Xraw):
            # Handle pre-loaded audio arrays
            logger.info('Processing pre-loaded audio arrays')

            # Ensure 2D
            # if all(isinstance(x, np.ndarray) for x in Xraw):
                # Xraw = np.vstack(Xraw)

            # Normalize audio
            audio_list = []
            for audio in Xraw:
                if self.params['normalize']: audio = librosa.util.normalize(audio)
                # Trim silence if requested
                if self.params['trim_silence']: audio, _ = librosa.effects.trim(audio, top_db=20)
                # Store
                audio_list.append(audio)

            # Store to disk if caching enabled
            pathnames = None
            filenames = None
            if self.params['use_audio_cache']:
                pathnames, filenames = self._store_audio_to_disk(audio_list, self.params['tempdir'])

            # Build result dict
            if self.find_func:
                return {
                    'audio': audio_list,
                    'pathnames': pathnames,
                    'filenames': filenames,
                    'sample_rates': [self.params['sr']] * len(audio_list),
                    'durations': [len(a) / self.params['sr'] for a in audio_list]
                }
            else:
                self.results['audio'] = audio_list
                self.results['pathnames'] = pathnames
                self.results['filenames'] = filenames
                self.results['sample_rates'] = [self.params['sr']] * len(audio_list)
                self.results['durations'] = [len(a) / self.params['sr'] for a in audio_list]

        return self.results

    def clean_init(self):
        """Clean previous results to ensure fresh start."""
        if hasattr(self, 'results'):
            logger.info('Cleaning previous fitted model results')
            if hasattr(self, 'results'):
                del self.results
            if hasattr(self, 'results_unique'):
                del self.results_unique
            if hasattr(self, 'distfit'):
                del self.distfit
            if hasattr(self, 'clusteval'):
                del self.clusteval
            if hasattr(self, 'pca'):
                del self.pca

        # Initialize results dictionary
        self.results = {
            'audio': None,
            'feat': None,
            'xycoord': None,
            'pathnames': None,
            'filenames': None,
            'labels': None,
            'sample_rates': None,
            'durations': None,
            'url': None
        }

    def embedding(self, X, metric="euclidean", embedding=None):
        """Compute embedding for visualization.

        Reduces high-dimensional features to 2D for visualization using
        t-SNE or UMAP. The 2D coordinates are used for scatter plots.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape (n_samples, n_features)
        metric : str
            Distance metric for embedding ('euclidean', 'cosine', etc.)
        embedding : str or None
            Embedding method: 'tsne', 'umap', or None (use first 2 features)

        Returns
        -------
        np.ndarray
            2D coordinates of shape (n_samples, 2)
        """
        if X.shape[0] <= 2:
            logger.warning('Too few samples for embedding')
            return np.zeros((X.shape[0], 2))

        if embedding is None:
            embedding = self.params.get('embedding', 'tsne')

        # For metadata method, might have lat/lon
        if self.params['method'] == 'metadata' and isinstance(self.results['feat'], pd.DataFrame):
            if 'lat' in self.results['feat'].columns and 'lon' in self.results['feat'].columns:
                xycoord = self.results['feat'][['lat', 'lon']].values
                self.results['xycoord'] = xycoord
                return xycoord

        # Perform embedding
        if embedding == 'tsne':
            logger.info(f"Computing t-SNE embedding")
            perplexity = min(X.shape[0] - 1, 30)
            xycoord = TSNE(
                n_components=2,
                init='random',
                metric=metric,
                perplexity=perplexity
            ).fit_transform(X)

        elif embedding == 'umap':
            logger.info("Computing UMAP embedding")
            logger.warning('UMAP temporarily disabled. Using first 2 features.')
            xycoord = X[:, 0:2]
        elif embedding == 'pca':
            logger.info("Computing PCA embedding")
            model = pca(**self.params_pca)
            out = model.fit_transform(X)
            xycoord = out['PC'].values
            self.pca = model
        else:
            # Use first two features
            logger.info("Using first two features as coordinates")
            xycoord = X[:, 0:2]

        # Ensure correct dtype
        if isinstance(xycoord, np.ndarray) and xycoord.dtype == object:
            xycoord = np.asarray(xycoord.tolist(), dtype=np.float32)

        # Store
        self.results['xycoord'] = xycoord

        return xycoord

    def fit_transform(self, X, cluster='agglomerative', evaluate='silhouette',
                     metric='euclidean', linkage='ward', min_clust=3, max_clust=25,
                     cluster_space='high', black_list=None, recursive=True):
        """Cluster audio files based on their acoustic similarity.

        This is the main workflow method that:
        1. Imports audio files
        2. Extracts features
        3. Computes embedding for visualization
        4. Evaluates and determines optimal clusters
        5. Returns clustering results

        Parameters
        ----------
        X : str, list, or np.ndarray
            Input audio files:
                * "c://music//" : Directory path
                * ['audio1.wav', ...] : List of file paths
                * np.ndarray : Pre-loaded audio or features
        cluster : str
            Clustering algorithm: 'agglomerative', 'kmeans', 'dbscan', 'hdbscan'
        evaluate : str
            Cluster evaluation method: 'silhouette', 'dbindex', 'derivative'
        metric : str
            Distance metric: 'euclidean', 'cosine', 'manhattan', etc.
        linkage : str
            Linkage method for hierarchical: 'ward', 'single', 'complete', 'average'
        min_clust : int
            Minimum number of clusters to evaluate
        max_clust : int
            Maximum number of clusters to evaluate
        cluster_space : str
            Feature space for clustering: 'high' (original) or 'low' (embedded)
        black_list : list
            Directories to exclude from scanning
        recursive : bool
            Scan subdirectories recursively

        Returns
        -------
        dict
            Results dictionary containing:
                * feat: Extracted features
                * xycoord: 2D embedding coordinates
                * pathnames: File paths
                * filenames: File names
                * labels: Cluster labels
                * sample_rates: Sample rates
                * durations: Audio durations

        Examples
        --------
        >>> from clustaudio import Clustaudio
        >>>
        >>> # Initialize
        >>> cl = Clustaudio(method='mfcc', sr=22050)
        >>>
        >>> # Cluster audio files
        >>> results = cl.fit_transform('path/to/music/')
        >>>
        >>> # Visualize
        >>> cl.scatter(dotsize=50)
        >>> cl.plot(labels=[0, 1, 2])
        """
        # Validate inputs
        if isinstance(X, str) and not os.path.exists(X):
            logger.error(f'Path not found: {X}')
            return None

        # Clean previous results
        self.clean_init()

        # Step 1: Import and preprocess data
        _ = self.import_data(X, black_list=black_list, recursive=recursive)

        # Step 2: Extract features
        _ = self.extract_feat(self.results)

        # Step 3: Compute embedding for visualization
        _ = self.embedding(self.results['feat'], metric=metric)

        # Step 4: Cluster
        self.cluster(
            cluster=cluster,
            evaluate=evaluate,
            cluster_space=cluster_space,
            metric=metric,
            linkage=linkage,
            min_clust=min_clust,
            max_clust=max_clust
        )

        return self.results


    def cluster(self, cluster='agglomerative', evaluate='silhouette',
               metric='euclidean', linkage='ward', min_clust=3, max_clust=25,
               cluster_space='high'):
        """Determine optimal number of clusters and assign cluster labels.

        Uses clustaudio library to evaluate different cluster counts and
        select the optimal number based on silhouette, dbindex, or derivative methods.

        Parameters
        ----------
        cluster : str
            Clustering algorithm
        evaluate : str
            Evaluation method
        metric : str
            Distance metric
        linkage : str
            Linkage method for hierarchical clustering
        min_clust : int
            Minimum clusters to evaluate
        max_clust : int
            Maximum clusters to evaluate
        cluster_space : str
            'high' for original features, 'low' for embedded coordinates

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        if self.results.get('feat', None) is None:
            raise Exception(logger.error('No features found. Run fit_transform() first.'))

        self.params['cluster_space'] = cluster_space

        if min_clust == max_clust:
            max_clust = min_clust + 1

        if len(self.results['feat']) == 0:
            return None

        # Select feature space
        if cluster_space == 'low':
            feat = self.results['xycoord']
            logger.info(f"Clustering using [{cluster_space}] dimensional space from embedding")
        else:
            feat = self.results['feat']
            logger.info(f"Clustering using [{cluster_space}] dimensional feature space")

        # Initialize clusteval
        ce = clusteval(
            cluster=cluster,
            evaluate=evaluate,
            metric=metric,
            linkage=linkage,
            min_clust=min_clust,
            max_clust=max_clust,
            normalize=False,
            verbose=get_logger()
        )

        # Fit clustering model
        _ = ce.fit(feat)

        # Get cluster labels
        labels = ce.results['labx']

        logger.info(f'Assigned {len(np.unique(labels))} cluster labels based on {feat.shape} feature space')

        # Store results
        self.results['labels'] = labels
        self.params['cluster_space'] = cluster_space
        self.params['metric_find'] = metric
        self.clusteval = ce

        # Store clustering parameters
        self.params_clusteval = {
            'cluster': cluster,
            'evaluate': evaluate,
            'metric': metric,
            'linkage': linkage,
            'min_clust': min_clust,
            'max_clust': max_clust
        }

        # Find unique representative samples
        if self.params['method'] != 'metadata':
            self.unique(metric=metric)

        return self.results['labels']

    def _check_status(self):
        """Check if results exist."""
        if not hasattr(self, 'results'):
            raise Exception(logger.error('No results found. Run fit_transform() first.'))

    def unique(self, metric=None):
        """Find representative (centroid) audio file for each cluster.

        For each cluster, finds the audio file closest to the cluster center.

        Parameters
        ----------
        metric : str or None
            Distance metric to use

        Returns
        -------
        dict
            Dictionary with representative samples:
                * labels: Cluster labels
                * idx: Indices of representative samples
                * xycoord_center: Coordinates of cluster centers
                * pathnames: Paths to representative files
        """
        self._check_status()

        if self.results['xycoord'] is None:
            logger.warning('Missing coordinates. Run embedding() first.')
            return None

        logger.info('Finding representative audio samples for each cluster...')

        if metric is None:
            metric = self.params_clusteval.get('metric', 'euclidean')

        center_idx, center_coord, pathnames = [], [], []
        uilabels = np.unique(self.results['labels'])

        if len(uilabels) >= 1:
            for label in uilabels:
                # Get samples in this cluster
                idx = np.where(self.results['labels'] == label)[0]

                # Compute cluster center
                xycoord_center = np.mean(self.results['xycoord'][idx, :], axis=0)

                # Find closest sample to center
                dist = distance.cdist(
                    self.results['xycoord'],
                    xycoord_center.reshape(-1, 1).T,
                    metric=metric
                )
                idx_closest = np.argmin(dist)

                # Store
                center_idx.append(idx_closest)
                center_coord.append(xycoord_center)

                if self.results.get('pathnames', None) is not None:
                    pathnames.append(self.results['pathnames'][idx_closest])
                else:
                    pathnames.append('')

            # Store results
            self.results_unique = {
                'labels': uilabels,
                'idx': center_idx,
                'xycoord_center': np.vstack(center_coord),
                'pathnames': pathnames
            }
        else:
            self.results_unique = {
                'labels': uilabels,
                'idx': None,
                'xycoord_center': None,
                'pathnames': None
            }

        return self.results_unique

    def _store_audio_to_disk(self, audio_list, tempdir):
        """Store audio arrays to disk as WAV files.

        Parameters
        ----------
        audio_list : list
            List of audio arrays
        tempdir : str
            Directory to save files

        Returns
        -------
        tuple
            (pathnames, filenames) lists
        """
        logger.info(f'Saving audio to tempdir [{tempdir}]')

        pathnames, filenames = [], []

        for i, audio in enumerate(tqdm(audio_list, disable=disable_tqdm(), desc='[clustaudio]')):
            filename = str(uuid.uuid4()) + '.wav'
            pathname = os.path.join(tempdir, filename)

            # Write audio to disk
            try:
                sf.write(pathname, audio, self.params['sr'])
                filenames.append(filename)
                pathnames.append(pathname)
            except Exception as e:
                logger.warning(f'Could not write audio {i}: {str(e)}')

        return pathnames, filenames


    # ========================================================================
    # PHASE 2: ADVANCED FEATURES
    # ========================================================================

    def extract_voice_segments(self, pathnames, output_dir=None, method='energy',
                               energy_threshold=0.02, min_segment_duration=0.5):
        """Detect and extract voice/speech segments from audio files.

        Similar to face detection in images, this finds speech segments in audio.

        Parameters
        ----------
        pathnames : str or list
            Path(s) to audio file(s)
        output_dir : str, optional
            Directory to save segments
        method : str
            VAD method: 'energy', 'webrtcvad', or 'librosa'
        energy_threshold : float
            Energy threshold for voice detection
        min_segment_duration : float
            Minimum segment duration in seconds

        Returns
        -------
        dict
            Dictionary containing:
                pathnames : Original file paths
                pathnames_segments : Paths to extracted segments
                timestamps : Start/end times
                num_segments : Number of segments per file

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>> segments = cl.extract_voice_segments('podcast.wav')
        >>> # Cluster the voice segments
        >>> results = cl.fit_transform(segments['pathnames_segments'])
        """
        # Ensure list
        if isinstance(pathnames, str):
            pathnames = [pathnames]

        logger.info(f"Extracting voice segments from {len(pathnames)} files...")

        results = vad.batch_extract_voice_segments(
            pathnames,
            output_dir=output_dir,
            sr=self.params['sr'],
            method=method,
            energy_threshold=energy_threshold,
            min_segment_duration=min_segment_duration
        )

        # Store results
        self.results_voice_segments = {
            'results': results,
            'pathnames': [r['pathname'] for r in results],
            'pathnames_segments': [r['pathnames_segments'] for r in results],
            'timestamps': [r['timestamps'] for r in results],
            'num_segments': [r['num_segments'] for r in results]
        }

        logger.info(f"Extracted {sum([r['num_segments'] for r in results])} total voice segments")

        return self.results_voice_segments

    def find(self, Xnew, metric=None, k=5, alpha=0.05):
        """Find similar audio files to query audio.

        Finding similar audio can be based on:
        - k-nearest neighbors
        - Statistical significance (probability-based)
        - Or both (union of results)

        Parameters
        ----------
        Xnew : str, list, or np.ndarray
            Query audio:
                * str: Path to audio file
                * list: List of paths
                * np.ndarray: Audio waveform or features
        metric : str, optional
            Distance metric. If None, uses metric from fit_transform
        k : int, optional
            Number of nearest neighbors to find
        alpha : float, optional
            Significance level for probability-based matching

        Returns
        -------
        dict
            Dictionary with results for each query containing:
                y_idx : Indices of similar files
                distance : Distances to query
                y_proba : Probability scores
                y_filenames : Filenames of similar files
                y_pathnames : Paths to similar files
                x_pathnames : Query file path

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>> results = cl.fit_transform('/music/')
        >>>
        >>> # Find similar to a specific song
        >>> similar = cl.find('query_song.mp3', k=10)
        >>> print(similar['query_song.mp3']['y_filenames'])
        """
        out = None
        self.find_func = True

        if (k is None) and (alpha is None):
            raise Exception(logger.error('Either "k" or "alpha" must be specified'))

        if metric is None:
            metric = self.params_clusteval.get('metric', 'euclidean')

        logger.info(f'Finding similar audio with metric={metric}, k={k}, alpha={alpha}')

        # Import query audio
        Xnew = self.import_data(Xnew)

        # Compute distances
        Y, feat = self._compute_distances(Xnew, metric=metric, alpha=alpha)

        # Collect results
        out = self._collect_similar(Xnew, Y, k, alpha, feat, todf=True)

        # Set the find_func to false because it is the end of this find function.
        self.find_func = False

        # Store
        self.results['predict'] = out
        self.params['metric_find'] = metric
        # return
        return self.results['predict']

    def _compute_distances(self, X, metric, alpha):
        """Compute distances between query and database samples."""
        # Extract features from query
        Xmapped = self.extract_feat(X)

        # Compute distances
        Y = distance.cdist(self.results['feat'], Xmapped, metric=metric)

        # Check for NaN
        if np.any(np.isnan(Y)):
            logger.warning(f'Metric {metric} resulted in NaN values!')

        # Fit distribution if using probability-based matching
        if (alpha is not None) and ((not hasattr(self, 'distfit')) or (self.distfit is None)):
            # Sample subset for efficiency
            Ytot = distance.cdist(self.results['feat'], self.results['feat'], metric=metric)

            x_max, y_max = min(500, Ytot.shape[0]), min(500, Ytot.shape[1])
            if x_max > 1 and y_max > 1:
                xrow = random.sample(range(x_max), x_max - 1)
                yrow = random.sample(range(y_max), y_max - 1)
                Ytot = Ytot[xrow, :][:, yrow]

                # Fit distribution
                self.distfit = distfit(bound='down', multtest=None,
                                      distr=['norm', 'expon', 'uniform'],
                                      verbose='warning')
                _ = self.distfit.fit_transform(Ytot.flatten())

        return Y, Xmapped

    def _collect_similar(self, X, Y, k, alpha, feat, todf=True):
        """Collect similar samples based on distance and/or probability."""
        filenames = X['filenames']
        out = {}
        out['feat'] = feat

        for i, filename in enumerate(filenames):
            store_key = {}
            idx_dist, idx_k = None, None

            # k-nearest neighbors
            if k is not None:
                idx_k = np.argsort(Y[:, i])[:k]

            # Probability-based
            if alpha is not None and self.distfit is not None:
                dist_results = self.distfit.predict(Y[:, i], verbose=0)
                idx_dist = np.where(dist_results['y_proba'] <= alpha)[0]
                idx_dist = idx_dist[np.argsort(dist_results['y_proba'][idx_dist])]
            else:
                dist_results = {'y_proba': np.array([np.nan] * Y.shape[0])}

            # Combine results
            idx = np.unique(np.append(idx_dist, idx_k) if idx_dist is not None else idx_k)
            idx = idx[~np.isnan(idx)].astype(int)

            logger.info(f'Found {len(idx)} similar files for {filename}')

            store_key = {
                'y_idx': idx,
                'distance': Y[idx, i],
                'y_proba': dist_results['y_proba'][idx],
                'labels': np.array(self.results['labels'])[idx].tolist(),
                'y_filenames': np.array(self.results['filenames'])[idx].tolist(),
                'y_pathnames': np.array(self.results['pathnames'])[idx].tolist(),
                'x_pathnames': X['pathnames'][i]
            }

            if todf:
                store_key = pd.DataFrame(store_key)

            out[filename] = store_key

        return out

    def create_playlist(self, labels=None, format='m3u8', savedir=None,
                       playlist_name=None, relative_paths=False):
        """Create audio playlists from clusters.

        Generates playlist files in various formats based on cluster assignments.

        Parameters
        ----------
        labels : int, list, or None
            Cluster label(s) to create playlists for.
            None = create playlist for each cluster
        format : str
            Playlist format:
                * 'm3u': M3U format
                * 'm3u8': Extended M3U (default, includes metadata)
                * 'pls': PLS format
                * 'json': JSON format with metadata
        savedir : str, optional
            Directory to save playlists. Default: self.params['dirpath']
        playlist_name : str, optional
            Custom playlist name. If None, uses 'cluster_X'
        relative_paths : bool
            Use relative paths in playlist (default: False = absolute)

        Returns
        -------
        dict
            Dictionary mapping labels to playlist paths

        Examples
        --------
        >>> cl = Clustaudio()
        >>> results = cl.fit_transform('/music/')
        >>>
        >>> # Create M3U8 playlists for all clusters
        >>> playlists = cl.create_playlist(format='m3u8')
        >>>
        >>> # Create single playlist for specific cluster
        >>> playlist = cl.create_playlist(labels=3, playlist_name='favorites')
        """
        self._check_status()

        if self.results.get('pathnames') is None:
            logger.error('No audio files found in results')
            return {}

        if savedir is None:
            savedir = self.params['dirpath']

        os.makedirs(savedir, exist_ok=True)

        # Determine labels to process
        if labels is None:
            labels = np.unique(self.results['labels'])
        elif not isinstance(labels, (list, np.ndarray)):
            labels = [labels]

        playlists = {}

        for label in labels:
            # Get files in this cluster
            idx = np.where(self.results['labels'] == label)[0]

            if len(idx) == 0:
                continue

            pathnames = np.array(self.results['pathnames'])[idx]

            # Generate playlist name
            if playlist_name is None:
                plist_name = f'cluster_{label}'
            else:
                plist_name = f'{playlist_name}_{label}' if len(labels) > 1 else playlist_name

            # Create playlist file
            if format in ['m3u', 'm3u8']:
                playlist_path = self._create_m3u_playlist(
                    pathnames, plist_name, savedir,
                    extended=(format == 'm3u8'),
                    relative_paths=relative_paths
                )
            elif format == 'pls':
                playlist_path = self._create_pls_playlist(
                    pathnames, plist_name, savedir,
                    relative_paths=relative_paths
                )
            elif format == 'json':
                playlist_path = self._create_json_playlist(
                    pathnames, idx, plist_name, savedir,
                    relative_paths=relative_paths
                )
            else:
                logger.error(f'Unknown playlist format: {format}')
                continue

            playlists[label] = playlist_path
            logger.info(f'Created playlist: {playlist_path} ({len(pathnames)} tracks)')

        return playlists

    def _create_m3u_playlist(self, pathnames, name, savedir, extended=True, relative_paths=False):
        """Create M3U/M3U8 playlist."""
        ext = '.m3u8' if extended else '.m3u'
        filepath = os.path.join(savedir, f'{name}{ext}')

        with open(filepath, 'w', encoding='utf-8') as f:
            if extended:
                f.write('#EXTM3U\n')

            for pathname in pathnames:
                if extended:
                    # Try to get duration
                    try:
                        duration = librosa.get_duration(path=pathname)
                        duration_int = int(duration)
                    except:
                        duration_int = -1

                    # Get title
                    title = os.path.splitext(os.path.basename(pathname))[0]
                    f.write(f'#EXTINF:{duration_int},{title}\n')

                # Write path
                if relative_paths:
                    path = os.path.relpath(pathname, savedir)
                else:
                    path = os.path.abspath(pathname)

                f.write(f'{path}\n')

        return filepath

    def _create_pls_playlist(self, pathnames, name, savedir, relative_paths=False):
        """Create PLS playlist."""
        filepath = os.path.join(savedir, f'{name}.pls')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('[playlist]\n')

            for i, pathname in enumerate(pathnames, 1):
                # Write path
                if relative_paths:
                    path = os.path.relpath(pathname, savedir)
                else:
                    path = os.path.abspath(pathname)

                f.write(f'File{i}={path}\n')

                # Title
                title = os.path.splitext(os.path.basename(pathname))[0]
                f.write(f'Title{i}={title}\n')

                # Duration
                try:
                    duration = librosa.get_duration(path=pathname)
                    f.write(f'Length{i}={int(duration)}\n')
                except:
                    f.write(f'Length{i}=-1\n')

            f.write(f'NumberOfEntries={len(pathnames)}\n')
            f.write('Version=2\n')

        return filepath

    def _create_json_playlist(self, pathnames, indices, name, savedir, relative_paths=False):
        """Create JSON playlist with metadata."""
        import json

        filepath = os.path.join(savedir, f'{name}.json')

        tracks = []
        for i, pathname in zip(indices, pathnames):
            track = {
                'index': int(i),
                'path': os.path.relpath(pathname, savedir) if relative_paths else os.path.abspath(pathname),
                'filename': os.path.basename(pathname),
                'cluster_label': int(self.results['labels'][i])
            }

            # Add duration if available
            if self.results.get('durations') is not None and i < len(self.results['durations']):
                track['duration'] = float(self.results['durations'][i])

            tracks.append(track)

        playlist_data = {
            'name': name,
            'format': 'clustaudio_json_v1',
            'track_count': len(tracks),
            'tracks': tracks
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(playlist_data, f, indent=2)

        return filepath

    def move_to_dir(self, target_labels=None, savedir=None, action='copy',
                   overwrite=False, user_input=True, create_playlists=False):
        """Move or copy audio files into directories by cluster labels.

        Organizes audio files based on clustering results. Can also generate
        playlists for each cluster.

        Parameters
        ----------
        target_labels : dict, optional
            Mapping of cluster labels to directory names.
            Example: {0: 'rock', 1: 'jazz', 2: 'classical'}
            None: Auto-generate names like 'cluster_0', 'cluster_1'
        savedir : str, optional
            Base directory for organized files
        action : str
            'copy' or 'move'
        overwrite : bool
            Overwrite existing files
        user_input : bool
            Ask for confirmation before each cluster
        create_playlists : bool
            Generate M3U8 playlists for each cluster

        Returns
        -------
        dict
            Status dictionary for each moved/copied file

        Examples
        --------
        >>> cl = Clustaudio()
        >>> results = cl.fit_transform('/music/')
        >>>
        >>> # Organize with auto-generated names
        >>> cl.move_to_dir(action='copy', create_playlists=True)
        >>>
        >>> # Organize with custom names
        >>> labels = {0: 'rock', 1: 'jazz', 2: 'classical'}
        >>> cl.move_to_dir(target_labels=labels, savedir='/organized_music/')
        """
        if target_labels is None:
            logger.info('Auto-generating folder names: cluster_0, cluster_1, ...')
            target_labels = {
                label: f'cluster_{label}'
                for label in np.unique(self.results['labels'])
            }

        file_status = {}
        playlists = {}

        for label, dirname in target_labels.items():
            # Get files for this cluster
            idx = np.where(self.results['labels'] == label)[0]

            if len(idx) == 0:
                logger.warning(f'No files found for label {label}')
                continue

            pathnames = self.results['pathnames'][idx]

            # Determine output directory
            if savedir is None:
                # Use base directory of first file
                base_dir = os.path.dirname(pathnames[0])
                output_dir = os.path.join(base_dir, dirname)
            else:
                output_dir = os.path.join(savedir, dirname)

            # Ask user
            if user_input:
                logger.info('-' * 70)
                logger.info(f'Cluster {label}: {action} {len(pathnames)} files to {output_dir}')
                response = input(f'[clustaudio] Press Enter to continue, q to quit: ')
                if response.lower() == 'q':
                    break

            # Move/copy files
            from clustaudio_phase2 import move_files
            status = move_files(pathnames, output_dir, action=action, overwrite=overwrite)
            file_status.update(status)

            # Create playlist if requested
            if create_playlists:
                playlist = self.create_playlist(
                    labels=label,
                    savedir=output_dir,
                    playlist_name=dirname
                )
                playlists.update(playlist)

        # Store results
        self.results['file_organization'] = {
            'status': file_status,
            'playlists': playlists
        }

        return file_status

    def save(self, filepath='clustaudio.pkl', overwrite=False):
        """Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to save file
        overwrite : bool
            Overwrite if exists

        Returns
        -------
        bool
            Success status
        """
        if filepath == 'clustaudio.pkl' or filepath is None:
            filepath = self.params['filepath']

        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'

        self.params['filepath'] = filepath

        # Prepare data for saving
        storedata = {
            'results': self.results,
            'params': self.params,
            'params_pca': self.params_pca,
            'params_mfcc': self.params_mfcc,
            'params_spectral': self.params_spectral,
            'params_chroma': self.params_chroma,
            'params_fingerprint': self.params_fingerprint,
            'params_metadata': self.params_metadata
        }

        # Add optional results
        if hasattr(self, 'results_voice_segments'): storedata['results_voice_segments'] = self.results_voice_segments
        if hasattr(self, 'results_unique'): storedata['results_unique'] = self.results_unique
        if hasattr(self, 'distfit'): storedata['distfit'] = self.distfit
        if hasattr(self, 'clusteval'): storedata['clusteval'] = self.clusteval
        if hasattr(self, 'pca'): storedata['pca'] = self.pca

        # Save
        status = pypickle.save(filepath, storedata, overwrite=overwrite, verbose=3)
        logger.info(f'Model saved to {filepath}')

        return status

    def load(self, filepath='clustaudio.pkl', verbose=3):
        """Load previously saved model.

        Parameters
        ----------
        filepath : str
            Path to saved model
        verbose : int
            Verbosity level

        Returns
        -------
        dict
            Loaded results
        """
        if filepath == 'clustaudio.pkl' or filepath is None:
            filepath = self.params['filepath']

        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'

        # Load
        storedata = pypickle.load(filepath, verbose=verbose)

        if storedata is not None:
            self.results = storedata['results']
            self.params = storedata['params']
            self.params_pca = storedata.get('params_pca', {})
            self.params_mfcc = storedata.get('params_mfcc', {})
            self.params_spectral = storedata.get('params_spectral', {})
            self.params_chroma = storedata.get('params_chroma', {})
            self.params_fingerprint = storedata.get('params_fingerprint', {})
            self.params_metadata = storedata.get('params_metadata', {})

            # Restore optional results
            self.results_voice_segments = storedata.get('results_voice_segments', None)
            self.results_unique = storedata.get('results_unique', None)
            self.distfit = storedata.get('distfit', None)
            self.clusteval = storedata.get('clusteval', None)
            self.pca = storedata.get('pca', None)

            self.find_func = False

            logger.info('Model loaded successfully!')
        else:
            logger.warning('Could not load model!')

        return self.results


    # ========================================================================
    # PHASE 3: VISUALIZATION, DENOISING, AND AUDIO SEPARATION
    # ========================================================================

    def separate_foreground_background(self, pathnames, output_dir=None, method='spectral_masking',
                                      denoise=True, denoise_method='spectral', save_both=True):
        """Separate foreground (signal) from background (noise) in audio files.

        This is particularly useful for noisy communications like ATC, radio transmissions,
        or any audio with significant background noise. Similar to VAD but extracts both
        signal and noise components.

        Parameters
        ----------
        pathnames : str or list
            Path(s) to audio file(s)
        output_dir : str, optional
            Output directory for separated files
        method : str
            Separation method:
                * 'spectral_masking': Spectral masking based on SNR (default)
                * 'harmonic_percussive': HPSS-based separation
                * 'nmf': Non-negative matrix factorization
                * 'wiener': Wiener filter approach
        denoise : bool
            Apply denoising before separation
        denoise_method : str
            Denoising method: 'spectral', 'wiener', 'bandpass', 'noisereduce'
        save_both : bool
            Save both foreground and background files

        Returns
        -------
        dict
            Dictionary containing:
                results : List of separation results per file
                pathnames_foreground : Paths to foreground audio files
                pathnames_background : Paths to background audio files
                snr_db : Signal-to-noise ratios

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>>
        >>> # Separate foreground/background
        >>> results = cl.separate_foreground_background(
        ...     'ATC_recording.wav',
        ...     method='spectral_masking',
        ...     denoise=True
        ... )
        >>>
        >>> # Cluster foreground (voice)
        >>> voice_clusters = cl.fit_transform(results['pathnames_foreground'])
        >>>
        >>> # Cluster background (noise types)
        >>> noise_clusters = cl.fit_transform(results['pathnames_background'])
        """
        # Ensure list
        if isinstance(pathnames, str):
            pathnames = [pathnames]

        logger.info(f"Separating foreground/background from {len(pathnames)} files...")
        logger.info(f"Method: {method}, Denoise: {denoise}, Denoise method: {denoise_method}")

        results = self.separation.batch_separate_foreground_background(
            pathnames,
            output_dir=output_dir,
            sr=self.params['sr'],
            method=method,
            denoise=denoise,
            denoise_method=denoise_method,
            save_both=save_both
        )

        # Store results
        self.results_separation = {
            'results': results,
            'pathnames_foreground': [r['pathname_foreground'] for r in results if r.get('pathname_foreground')],
            'pathnames_background': [r['pathname_background'] for r in results if r.get('pathname_background')],
            'snr_db': [r.get('snr_db', None) for r in results]
        }

        n_success = len(self.results_separation['pathnames_foreground'])
        avg_snr = np.mean([s for s in self.results_separation['snr_db'] if s is not None])

        logger.info(f"Separated {n_success} files successfully")
        logger.info(f"Average SNR: {avg_snr:.2f} dB")

        return self.results_separation

    def denoise(self, pathnames=None, method='spectral', output_dir=None,
                save_denoised=False, **kwargs):
        """Apply denoising to audio files.

        Parameters
        ----------
        pathnames : str, list, or None
            Path(s) to audio files. If None, uses self.results['pathnames']
        method : str
            Denoising method:
                * 'spectral': Spectral subtraction (default)
                * 'wiener': Wiener filtering
                * 'noisereduce': Advanced noise reduction (requires noisereduce)
                * 'median': Median filtering
                * 'bandpass': Band-pass filtering (good for speech)
        output_dir : str, optional
            Output directory for denoised files
        save_denoised : bool
            Save denoised files to disk
        **kwargs
            Additional parameters for specific methods

        Returns
        -------
        dict
            Dictionary containing denoised audio arrays and paths

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>>
        >>> # Denoise files before clustering
        >>> denoised = cl.denoise(
        ...     pathnames,
        ...     method='spectral',
        ...     save_denoised=True
        ... )
        >>>
        >>> # Cluster denoised audio
        >>> results = cl.fit_transform(denoised['pathnames_denoised'])
        """

        # Use existing pathnames if none provided
        if pathnames is None:
            if self.results.get('pathnames') is None:
                raise Exception(logger.error('No pathnames provided and none found in results'))
            pathnames = self.results['pathnames']

        # Ensure list
        if isinstance(pathnames, str):
            pathnames = [pathnames]

        logger.info(f"Denoising {len(pathnames)} files using {method} method...")

        # Setup output directory
        if output_dir is None and save_denoised:
            output_dir = os.path.join(self.params['dirpath'], 'denoised')

        if save_denoised:
            os.makedirs(output_dir, exist_ok=True)

        # Process each file
        denoised_audio = []
        pathnames_denoised = []

        for pathname in tqdm(pathnames, disable=disable_tqdm(), desc='[clustaudio] Denoising'):
            try:
                # Load audio
                audio, sr = librosa.load(pathname, sr=self.params['sr'], mono=self.params['mono'])
                # Denoise
                audio_denoised = self.separation.denoise_audio(audio, sr=sr, method=method, **kwargs)
                denoised_audio.append(audio_denoised)

                # Save if requested
                if save_denoised:
                    filename = os.path.basename(pathname)
                    name, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_dir, f'{name}_denoised.wav')
                    sf.write(output_path, audio_denoised, sr)
                    pathnames_denoised.append(output_path)
                else:
                    pathnames_denoised.append(None)

            except Exception as e:
                logger.error(f"Failed to denoise {pathname}: {str(e)}")
                denoised_audio.append(None)
                pathnames_denoised.append(None)

        results = {
            'audio_denoised': denoised_audio,
            'pathnames_denoised': pathnames_denoised,
            'pathnames_original': pathnames,
            'method': method
        }

        self.results_denoised = results

        logger.info(f"Denoised {len([p for p in pathnames_denoised if p is not None])} files")

        return results

    def enhance_atc_audio(self, pathnames, method='aggressive', output_dir=None):
        """Enhance noisy ATC (Air Traffic Control) communication audio.

        Specialized enhancement for radio communications with:
        - Radio static and interference
        - Wind noise
        - Engine noise
        - Low SNR conditions

        Parameters
        ----------
        pathnames : str or list
            Path(s) to ATC audio files
        method : str
            Enhancement level:
                * 'light': Minimal processing
                * 'moderate': Balanced enhancement
                * 'aggressive': Maximum enhancement (default)
        output_dir : str, optional
            Output directory for enhanced files

        Returns
        -------
        dict
            Dictionary with enhanced audio and paths

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc', sr=16000)
        >>>
        >>> # Enhance ATC audio
        >>> enhanced = cl.enhance_atc_audio('ATC_recordings/', method='aggressive')
        >>>
        >>> # Cluster enhanced communications
        >>> results = cl.fit_transform(enhanced['pathnames_enhanced'])
        """

        # Ensure list
        if isinstance(pathnames, str):
            if os.path.isdir(pathnames):
                pathnames = listdir(pathnames, ext=self.params['ext'])
            else:
                pathnames = [pathnames]

        logger.info(f"Enhancing {len(pathnames)} ATC audio files (method: {method})...")

        # Setup output
        if output_dir is None:
            output_dir = os.path.join(self.params['dirpath'], 'atc_enhanced')

        os.makedirs(output_dir, exist_ok=True)

        # Process files
        enhanced_audio = []
        pathnames_enhanced = []

        for pathname in tqdm(pathnames, disable=disable_tqdm(), desc='[clustaudio] ATC Enhancement'):
            try:
                # Load
                audio, sr = librosa.load(pathname, sr=self.params['sr'], mono=True)

                # Enhance
                audio_enhanced = self.separation.enhance_atc_audio(audio, sr=sr, method=method)

                enhanced_audio.append(audio_enhanced)

                # Save
                filename = os.path.basename(pathname)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f'{name}_enhanced.wav')
                sf.write(output_path, audio_enhanced, sr)
                pathnames_enhanced.append(output_path)

            except Exception as e:
                logger.error(f"Failed to enhance {pathname}: {str(e)}")
                enhanced_audio.append(None)
                pathnames_enhanced.append(None)

        results = {
            'audio_enhanced': enhanced_audio,
            'pathnames_enhanced': pathnames_enhanced,
            'pathnames_original': pathnames,
            'method': method
        }

        self.results_atc_enhanced = results

        logger.info(f"Enhanced {len([p for p in pathnames_enhanced if p is not None])} ATC files")

        return results

    def plot(self, labels=None, show_waveform=True, show_spectrogram=False,
            max_per_cluster=4, figsize=(16, 12)):
        """Visualize clustered audio files.

        Creates visualization grid showing waveforms and/or spectrograms
        for each cluster.

        Parameters
        ----------
        labels : int, list, or None
            Cluster label(s) to plot. None = plot all
        show_waveform : bool
            Show waveforms
        show_spectrogram : bool
            Show spectrograms
        max_per_cluster : int
            Maximum samples per cluster
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> cl = Clustaudio(method='mfcc')
        >>> results = cl.fit_transform('/audio/')
        >>>
        >>> # Plot waveforms for all clusters
        >>> fig = cl.plot(show_waveform=True)
        >>>
        >>> # Plot spectrograms for specific clusters
        >>> fig = cl.plot(labels=[0, 1, 2], show_spectrogram=True)
        """
        import visualization as viz

        self._check_status()

        if self.results.get('pathnames') is None:
            logger.error('No audio files found in results')
            return None

        # Determine labels to plot
        if labels is None:
            labels_to_plot = np.unique(self.results['labels'])
        elif not isinstance(labels, (list, np.ndarray)):
            labels_to_plot = [labels]
        else:
            labels_to_plot = labels

        # Filter pathnames and labels
        mask = np.isin(self.results['labels'], labels_to_plot)
        pathnames = np.array(self.results['pathnames'])[mask]
        cluster_labels = np.array(self.results['labels'])[mask]

        # Create visualization
        if show_waveform and show_spectrogram:
            # Create combined plot
            logger.info(f"Plotting {len(labels_to_plot)} clusters with waveforms and spectrograms...")
            # TODO: Implement combined view
            fig = viz.plot_cluster_audio_grid(
                pathnames, cluster_labels,
                max_per_cluster=max_per_cluster,
                sr=self.params['sr'],
                figsize=figsize,
                show_waveform=True
            )
        elif show_waveform:
            logger.info(f"Plotting {len(labels_to_plot)} clusters with waveforms...")
            fig = viz.plot_cluster_audio_grid(
                pathnames, cluster_labels,
                max_per_cluster=max_per_cluster,
                sr=self.params['sr'],
                figsize=figsize,
                show_waveform=True
            )
        elif show_spectrogram:
            logger.info(f"Plotting {len(labels_to_plot)} clusters with spectrograms...")
            fig = viz.plot_cluster_audio_grid(
                pathnames, cluster_labels,
                max_per_cluster=max_per_cluster,
                sr=self.params['sr'],
                figsize=figsize,
                show_waveform=False
            )
        else:
            logger.warning('Both show_waveform and show_spectrogram are False, nothing to plot')
            return None

        plt.show()

        return fig

    def plot_find(self, cmap=None, figsize=(15, 10), invert_colors=False):
        """Plot the input image together with the predicted images.

        Parameters
        ----------
        cmap : str, (default: None)
            Colorscheme for the images.
            'gray', 'binary',  None (uses rgb colorscheme)
        figsize : tuple, (default: (15, 10).
            Size of the figure (height,width).
        invert_colors: Invert colors for the plot.
            True: RGB-> BGR
            False: Keep as is

        Returns
        -------
        None.

        """
        from pathlib import Path
        import visualization as viz

        # Plot the images that are similar to each other.
        if self.results.get('predict', None) is not None:
            for key in self.results['predict'].keys():
                if key=='feat':
                    continue

                try:
                    if (self.results['predict'].get(key).get('y_idx', None) is not None):
                        # Collect images
                        input_audio = self.results['predict'][key]['x_pathnames'][0]
                        find_audio = self.results['predict'][key]['y_pathnames'].values

                        # ---- Normalize types to Python str + normalize paths ----
                        input_list = [str(Path(input_audio))]
                        find_list = [str(Path(p)) for p in find_audio.tolist()]

                        n_cols = max(len(input_list), len(find_list))
                        fig, axes = plt.subplots(2, n_cols, figsize=figsize, squeeze=False)

                        # ---- Top row: input audio ----
                        for i, audio in enumerate(input_list):
                            viz.plot_waveform(audio, ax=axes[0, i])
                            axes[0, i].set_title(f"Input: {os.path.basename(audio)}")

                        # ---- Bottom row: found audios ----
                        for i, audio in enumerate(find_list):
                            viz.plot_waveform(audio, ax=axes[1, i])
                            axes[1, i].set_title(f"Match {i+1}:  {os.path.basename(audio)}")

                        # ---- Turn off unused axes ----
                        for i in range(len(input_list), n_cols):
                            axes[0, i].axis("off")

                        for i in range(len(find_list), n_cols):
                            axes[1, i].axis("off")

                        plt.tight_layout()
                        plt.show()



                        # verbose
                        logger.info(f'[{len(find_audio)}] similar audio detected for input audio: [{key}]')
                except:
                    pass
        else:
            logger.warning('No prediction results are found. Hint: Try to run the .find() functionality first.')


    def plot_separation(self, index=0, figsize=(20, 15)):
        """Plot foreground/background separation results.

        Parameters
        ----------
        index : int
            Index of file to plot from separation results
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'results_separation'):
            logger.error('No separation results found. Run separate_foreground_background() first.')
            return None

        result = self.results_separation['results'][index]

        # Load original
        original, sr = librosa.load(result['pathname'], sr=self.params['sr'], mono=True)

        # Load separated
        foreground, _ = librosa.load(result['pathname_foreground'], sr=sr, mono=True)
        background, _ = librosa.load(result['pathname_background'], sr=sr, mono=True)

        # Plot
        fig = self.viz.plot_separation_result(
            original, foreground, background,
            sr=sr, figsize=figsize,
            snr_db=result.get('snr_db')
        )

        plt.show()

        return fig

    def plot_denoise(self, index=0, figsize=(14, 10)):
        """Plot denoising comparison.

        Parameters
        ----------
        index : int
            Index of file to plot
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not hasattr(self, 'results_denoised'):
            logger.error('No denoising results found. Run denoise() first.')
            return None

        original_path = self.results_denoised['pathnames_original'][index]
        denoised_audio = self.results_denoised['audio_denoised'][index]
        method = self.results_denoised['method']

        # Load original
        original, sr = librosa.load(original_path, sr=self.params['sr'], mono=True)

        # Plot comparison
        fig = self.viz.plot_denoise_comparison(
            original, denoised_audio,
            sr=sr, method_name=f'{method.capitalize()} Denoised',
            figsize=figsize
        )

        plt.show()

        return fig

    def plot_statistics(self, figsize=(14, 10)):
        """Plot comprehensive cluster statistics.

        Parameters
        ----------
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._check_status()

        fig = self.viz.plot_cluster_statistics(self.results, figsize=figsize)

        plt.show()

        return fig


# ========================================================================
# HELPER FUNCTIONS FOR FILE ORGANIZATION
# ========================================================================
def get_logger():
    """Return current logger level."""
    return logger.getEffectiveLevel()


def set_logger(verbose: [str, int] = 'info'):
    """Set the logger verbosity level.

    Parameters
    ----------
    verbose : str or int
        Verbosity level:
            * [0, 60, None, 'silent', 'off']: No messages
            * [10, 'debug']: Debug and above
            * [20, 'info']: Info and above
            * [30, 'warning']: Warning and above
            * [40, 'error']: Error only
            * [50, 'critical']: Critical only
    """
    # Handle None and 0
    if (verbose == 0) or (verbose is None):
        verbose = 60

    # Convert string to level
    if isinstance(verbose, str):
        levels = {
            'silent': 60,
            'off': 60,
            'no': 60,
            'debug': 10,
            'info': 20,
            'warning': 30,
            'error': 40,
            'critical': 50
        }
        verbose = levels.get(verbose, 20)

    logger.setLevel(verbose)


def disable_tqdm():
    """Determine if tqdm progress bars should be disabled."""
    return logger.getEffectiveLevel() >= 30


def listdir(dirpath, ext=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac'],
           black_list=None, recursive=True):
    """Recursively collect audio files from directory.

    Parameters
    ----------
    dirpath : str
        Directory path to scan
    ext : list
        List of file extensions to include
    black_list : list
        Directory names to exclude
    recursive : bool
        Scan subdirectories

    Returns
        list
        List of file paths
    """
    if isinstance(ext, str):
        ext = [ext]

    if not os.path.isdir(dirpath):
        raise Exception(f'Directory not found: {dirpath}')

    audio_files = []

    if recursive:
        # Recursive scan
        for root, _, filenames in os.walk(dirpath):
            # Check blacklist
            if black_list and any(bl in os.path.split(root)[1] for bl in black_list):
                logger.info(f'Excluded directory: {root}')
                continue

            # Collect files with matching extensions
            for ext_type in ext:
                for filename in fnmatch.filter(filenames, f'*.{ext_type}'):
                    audio_files.append(os.path.join(root, filename))
    else:
        # Non-recursive: top level only
        _, _, filenames = next(os.walk(dirpath))
        for ext_type in ext:
            for filename in fnmatch.filter(filenames, f'*.{ext_type}'):
                audio_files.append(os.path.join(dirpath, filename))

    logger.info(f'Found in total [{len(audio_files)}] audio files in {dirpath}')

    return audio_files


def _set_tempdir(dirpath, show_logger=True):
    """Set up temporary directory.

    Parameters
    ----------
    dirpath : str or None
        Directory path or None for system temp
    show_logger : bool
        Whether to log the directory path

    Returns
    -------
    str
        Absolute path to temp directory
    """
    try:
        if dirpath is None:
            dirpath = os.path.join(tempfile.gettempdir(), 'clustaudio')
        elif os.path.isdir(dirpath):
            pass
        elif isinstance(dirpath, str):
            dirpath = os.path.join(tempfile.gettempdir(), dirpath)

        # Create if doesn't exist
        if not os.path.isdir(dirpath):
            if show_logger:
                logger.info(f'Creating directory: [{dirpath}]')
            os.makedirs(dirpath, exist_ok=True)
    except Exception as e:
        raise Exception(logger.error(f'Cannot create directory [{dirpath}]: {str(e)}'))

    dirpath = os.path.abspath(dirpath)

    if show_logger:
        logger.info(f"Directory set to [{dirpath}]")

    return dirpath


def import_example(data='atc', n=10, freq=440, duration=1.0, sr=22050, amplitude=0.5):
    """
    Import example audio data or generate a sine wave.

    Parameters
    ----------
    data : str, optional
        Type of example data to import or generate:
            * 'wave': Generate a sine wave with specified parameters.
            * 'atc': Return list of ATC (Air Traffic Control) audio file paths.
            * 'various': Return list of various example audio file paths.
    freq : float, optional
        Frequency of sine wave in Hz (used if data='wave'). Default is 440.
    duration : float, optional
        Duration of sine wave in seconds (used if data='wave'). Default is 1.0.
    sr : int, optional
        Sample rate for generated sine wave (used if data='wave'). Default is 22050.
    amplitude : float, optional
        Amplitude of sine wave (0-1, used if data='wave'). Default is 0.5.

    Returns
    -------
    tuple or list
        If data='wave': tuple (audio, sr) where audio is a numpy array.
        If data='atc' or 'various': list of file paths to example audio files.

    Examples
    --------
    >>> from clustaudio import import_example
    >>> audio = import_example(data='wave', freq=440, duration=2.0)
    >>> atc_files = import_example(data='atc')
    >>> various_files = import_example(data='various')
    """
    if os.path.isdir(r'./tests/audio/'):
        audiodir = r'./tests/audio/'
    elif os.path.isdir(r'./clustaudio/tests/audio/'):
        audiodir = r'./clustaudio/tests/audio/'

    if (data=='wave'):
        # Generate n random sine waves with random freq, amplitude, phase
        rng = np.random.default_rng()
        audios = []
        for _ in range(n):
            freq_i = freq
            amp_i = amplitude
            phase_i = 0.0
            # Add randomness if n > 1
            if n > 1:
                freq_i = rng.uniform(0.7 * freq, 1.3 * freq)
                amp_i = rng.uniform(0.5 * amplitude, amplitude)
                phase_i = rng.uniform(0, 2 * np.pi)
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            audio = amp_i * np.sin(2 * np.pi * freq_i * t + phase_i)
            audios.append(audio.astype(np.float32))
        if n == 1:
            return audios[0]
        else:
            return audios
            #return np.vstack(audios)
    elif data=='atc':
        return listdir(os.path.join(audiodir, 'atc'))[0:n]
    elif data=='various':
        return listdir(os.path.join(audiodir, 'various'))[0:n]



def move_files(pathnames, savedir, action='copy', overwrite=False):
    """Move or copy audio files to a directory.

    Parameters
    ----------
    pathnames : list
        List of file paths
    savedir : str
        Destination directory
    action : str
        'move' or 'copy'
    overwrite : bool
        Overwrite existing files

    Returns
    -------
    dict
        Status for each file
    """
    # Create directory
    os.makedirs(savedir, exist_ok=True)

    # Select action function
    if action.lower() == 'move':
        action_func = shutil.move
    else:
        action_func = shutil.copy

    file_status = {}

    for filepath in pathnames:
        if not os.path.isfile(filepath):
            logger.warning(f'File not found: {filepath}')
            continue

        filename = os.path.basename(filepath)
        dest_path = os.path.join(savedir, filename)

        file_status[filepath] = {
            'success': False,
            'action': action,
            'destination': dest_path
        }

        try:
            if os.path.exists(dest_path) and not overwrite:
                logger.warning(f'File exists, skipping: {filename}')
                continue

            action_func(filepath, dest_path)
            file_status[filepath]['success'] = True
            logger.debug(f'{action}: {filepath} -> {dest_path}')

        except Exception as e:
            logger.error(f'Failed to {action} {filepath}: {str(e)}')

    successful = sum(1 for v in file_status.values() if v['success'])
    logger.info(f'{action.capitalize()}d {successful}/{len(pathnames)} files to {savedir}')

    return file_status

def create_dir(pathname, savedir=None):
    """Create directory for file organization."""
    dirname = os.path.dirname(pathname)
    filename = os.path.basename(pathname)

    if savedir is None:
        movedir = os.path.join(dirname, 'clustaudio')
    else:
        movedir = savedir

    os.makedirs(movedir, exist_ok=True)

    return movedir, dirname, filename
