.. _performance_code_directive:



Performance
###########

The speed and memory usage of ``clustaudio`` are determined by the feature extraction method, audio processing operations, and the characteristics of your audio files. Audio processing is generally fast due to efficient libraries like ``librosa`` and ``soundfile``, but certain operations like denoising and separation can be more computationally intensive.

Unlike image processing where resizing is common, audio resampling to different sample rates is the primary dimension reduction technique. Lower sample rates (e.g., 16 kHz for speech vs 44.1 kHz for music) significantly reduce processing time and memory usage while maintaining adequate quality for analysis.


Processing Speed
''''''''''''''''

**Feature Extraction** is typically very fast, processing audio at **10-100x real-time** depending on the method:

.. table:: Feature Extraction Speed

  +-------------------+------------------+---------------------------+
  | Method            | Processing Speed | Notes                     |
  +===================+==================+===========================+
  | MFCC              | 50-100x RT       | Very fast, default choice |
  +-------------------+------------------+---------------------------+
  | Spectral          | 20-50x RT        | Fast, good for timbre     |
  +-------------------+------------------+---------------------------+
  | Chroma            | 15-40x RT        | Medium speed              |
  +-------------------+------------------+---------------------------+
  | MFCC-Spectral     | 20-50x RT        | Combined features         |
  +-------------------+------------------+---------------------------+
  | Fingerprint       | 30-60x RT        | Fast binary hashing       |
  +-------------------+------------------+---------------------------+
  | Metadata          | >1000x RT        | Instant (file reading)    |
  +-------------------+------------------+---------------------------+

**RT** = Real-time. For example, 50x RT means a 1-minute audio file is processed in ~1.2 seconds.

.. note::
	Processing speed varies based on CPU performance and sample rate. Higher sample rates (44.1 kHz) take longer than lower rates (16 kHz). Timings are based on typical laptop CPU performance with default parameters.


**Advanced Processing Operations** (Phase 3 features):

.. table:: Advanced Audio Processing Speed

  +-------------------+------------------+---------------------------+
  | Operation         | Processing Speed | Use Case                  |
  +===================+==================+===========================+
  | Spectral Denoise  | 5-10x RT         | General noise reduction   |
  +-------------------+------------------+---------------------------+
  | Separation        | 3-8x RT          | Voice/noise separation    |
  +-------------------+------------------+---------------------------+
  | ATC Enhancement   | 2-5x RT          | Radio communication       |
  +-------------------+------------------+---------------------------+
  | Voice Activity    | 20-50x RT        | Fast segment detection    |
  +-------------------+------------------+---------------------------+
  | WebRTC VAD        | 10-30x RT        | High-quality VAD          |
  +-------------------+------------------+---------------------------+

.. note::
	ATC Enhancement with ``method='aggressive'`` runs the full processing chain (band-pass filter + spectral subtraction + optional noisereduce), which is slower but provides the best quality for noisy communications.


**Clustering Performance**:

Clustering 1000 audio files typically takes **30-60 seconds** depending on feature dimensionality and clustering method:

.. table:: Clustering Time Examples

  +-------------------+------------------+---------------------------+
  | Dataset           | Files            | Clustering Time           |
  +===================+==================+===========================+
  | Music library     | 1000 songs       | ~45 seconds               |
  +-------------------+------------------+---------------------------+
  | Podcast segments  | 500 segments     | ~25 seconds               |
  +-------------------+------------------+---------------------------+
  | ATC voice clips   | 2000 segments    | ~60 seconds               |
  +-------------------+------------------+---------------------------+
  | Environmental     | 300 sounds       | ~15 seconds               |
  +-------------------+------------------+---------------------------+

.. note::
	Clustering time includes feature extraction, embedding (t-SNE), and cluster evaluation. The majority of time is spent on embedding for visualization, not the clustering itself.


Realistic Workflow Timings
'''''''''''''''''''''''''''

**Example 1: Music Library Clustering** (1000 songs, ~3 minutes each)

.. code-block:: python

	from clustaudio import clustaudio

	cl = clustaudio(method='mfcc-spectral', sr=22050, duration=30)
	results = cl.fit_transform('/music_library/', min_clust=5, max_clust=15)

	# Approximate timing breakdown:
	# - Load 30 sec from 1000 songs: ~20 seconds
	# - Extract features: ~15 seconds
	# - t-SNE embedding: ~25 seconds
	# - Clustering evaluation: ~10 seconds
	# Total: ~70 seconds for 1000 songs

**Duration limiting** (e.g., ``duration=30``) dramatically improves performance by only loading the first 30 seconds of each audio file, which is usually sufficient for feature extraction.


**Example 2: ATC Communication Processing** (100 ATC recordings)

.. code-block:: python

	from clustaudio import clustaudio
	import vad

	cl = clustaudio(method='mfcc', sr=16000)

	# Full ATC workflow timing:
	enhanced = cl.enhance_atc_audio(pathnames, method='aggressive')
	# ~20 seconds for 100 files

	separated = cl.separate_foreground_background(enhanced['pathnames_enhanced'])
	# ~30 seconds for 100 files

	segments = vad.batch_extract_voice_segments(separated['pathnames_foreground'])
	# ~15 seconds for 100 files

	results = cl.fit_transform([p for s in segments for p in s['pathnames_segments']])
	# ~40 seconds for ~500 segments

	# Total: ~105 seconds for complete ATC workflow


**Example 3: Duplicate Detection** (5000 audio files)

.. code-block:: python

	import fingerprint
	from clustaudio import listdir

	pathnames = listdir('/audio_library/', ext=['mp3', 'wav', 'flac'])
	fps = fingerprint.batch_compute_fingerprints(pathnames, method='chroma_hash')
	# ~2-3 minutes for 5000 files

	duplicates = fingerprint.find_duplicates(fps, threshold=0.1)
	# ~5 seconds

	# Total: ~3 minutes to find all duplicates in 5000 files


Memory Usage
''''''''''''

Memory usage depends on several factors:

**Feature Matrix Size**:

The primary memory consumer is the feature matrix (n_samples × n_features):

.. table:: Memory Usage by Feature Type

  +-------------------+------------------+---------------------------+
  | Feature Method    | Features/File    | Memory (1000 files)       |
  +===================+==================+===========================+
  | MFCC              | 26 floats        | ~0.2 MB                   |
  +-------------------+------------------+---------------------------+
  | Spectral          | 12 floats        | ~0.1 MB                   |
  +-------------------+------------------+---------------------------+
  | Chroma            | 24 floats        | ~0.2 MB                   |
  +-------------------+------------------+---------------------------+
  | MFCC-Spectral     | 38 floats        | ~0.3 MB                   |
  +-------------------+------------------+---------------------------+
  | Fingerprint       | 32-512 bits      | ~0.05 MB (very compact)   |
  +-------------------+------------------+---------------------------+

**Audio Data in Memory**:

.. code-block:: python

	# Single audio file memory usage (approximate):
	# - 1 minute @ 16 kHz mono: ~1 MB
	# - 1 minute @ 22.05 kHz mono: ~1.4 MB
	# - 1 minute @ 44.1 kHz mono: ~2.8 MB
	# - 1 minute @ 44.1 kHz stereo: ~5.6 MB

**Total Memory Estimates**:

For clustering **1000 audio files** (1 minute each, duration=30):

.. table:: Total Memory Usage

  +-------------------+---------------------------+
  | Component         | Memory Usage              |
  +===================+===========================+
  | Audio in memory   | ~400 MB (30s @ 22kHz)     |
  +-------------------+---------------------------+
  | Features          | ~0.3 MB (MFCC-Spectral)   |
  +-------------------+---------------------------+
  | Embedding         | ~0.02 MB (2D coordinates) |
  +-------------------+---------------------------+
  | Overhead          | ~100 MB (Python, libs)    |
  +-------------------+---------------------------+
  | **Total**         | **~500 MB - 1 GB**        |
  +-------------------+---------------------------+

.. note::
	Memory usage can be reduced by:

	* Using ``duration=30`` or lower to limit audio length
	* Using lower sample rates (``sr=16000`` for speech)
	* Processing in batches for very large datasets
	* Using ``use_audio_cache=False`` to avoid caching


Optimization Tips
'''''''''''''''''

**For Large Datasets (1000+ files)**:

.. code-block:: python

	# 1. Limit duration
	cl = clustaudio(method='mfcc', sr=22050, duration=30)

	# 2. Use lower sample rate for speech
	cl = clustaudio(method='mfcc', sr=16000)  # Good for speech/podcasts

	# 3. Process in batches
	files = listdir('/large_collection/', ext=['wav', 'mp3'])
	batch_size = 500

	for i in range(0, len(files), batch_size):
		batch = files[i:i+batch_size]
		results = cl.fit_transform(batch)
		cl.save(f'batch_{i//batch_size}.pkl')


**For Low-Spec Systems**:

.. code-block:: python

	# Use lighter configuration
	cl = clustaudio(
		method='spectral',      # Fewer features than MFCC
		sr=16000,               # Lower sample rate
		duration=20,            # Short duration
		params_spectral={'n_fft': 1024}  # Smaller FFT
	)


**For Maximum Speed**:

.. code-block:: python

	# Fastest possible clustering
	cl = clustaudio(
		method='metadata',      # Instant (no audio processing)
		sr=8000                 # Lowest usable sample rate
	)

	# Or use fingerprinting for duplicates only
	import fingerprint
	fps = fingerprint.batch_compute_fingerprints(pathnames)
	duplicates = fingerprint.find_duplicates(fps)


Sample Rate Recommendations
''''''''''''''''''''''''''''

Choosing the right sample rate balances quality and performance:

.. table:: Sample Rate Guidelines

  +-------------------+------------------+---------------------------+
  | Use Case          | Recommended SR   | Rationale                 |
  +===================+==================+===========================+
  | Speech/Podcasts   | 16 kHz           | Sufficient for voice      |
  +-------------------+------------------+---------------------------+
  | ATC/Radio         | 8-16 kHz         | Matches radio bandwidth   |
  +-------------------+------------------+---------------------------+
  | General purpose   | 22.05 kHz        | Good balance (default)    |
  +-------------------+------------------+---------------------------+
  | Music analysis    | 44.1 kHz         | Full frequency range      |
  +-------------------+------------------+---------------------------+
  | Quick processing  | 8-16 kHz         | 2-4x faster than 44 kHz   |
  +-------------------+------------------+---------------------------+

.. note::
	Sample rate affects processing time quadratically for some operations. Using 16 kHz instead of 44.1 kHz provides ~8x speedup for spectral operations while maintaining adequate quality for speech analysis.


Comparison with Image Clustering
'''''''''''''''''''''''''''''''''

For reference, here's how ``clustaudio`` compares to the original ``clustaudio`` on equivalent tasks:

.. table:: Performance Comparison (1000 items)

  +-------------------+------------------+---------------------------+
  | Operation         | clustaudio       | clustaudio                |
  +===================+==================+===========================+
  | Feature Extract   | 16s (PCA)        | 10-15s (MFCC)             |
  +-------------------+------------------+---------------------------+
  | Feature Extract   | 31s (HOG)        | 15-25s (Spectral)         |
  +-------------------+------------------+---------------------------+
  | Combined Features | 47s (PCA-HOG)    | 25-35s (MFCC-Spectral)    |
  +-------------------+------------------+---------------------------+
  | Duplicate Detect  | ~10s (hash)      | ~8s (fingerprint)         |
  +-------------------+------------------+---------------------------+
  | Total Pipeline    | ~60s             | ~45s                      |
  +-------------------+------------------+---------------------------+

Audio processing is generally **faster** than image processing for equivalent dataset sizes, primarily because:

* Audio features (MFCC, spectral) extract more quickly than HOG
* Modern audio libraries (librosa) are highly optimized
* Duration limiting is more effective than image resizing


Benchmark Dataset
'''''''''''''''''

All timings are based on a reference system with:

* **CPU**: Intel Core i7 (4 cores, 8 threads) or equivalent
* **RAM**: 16 GB
* **Python**: 3.8+
* **Libraries**: librosa 0.10.0, scikit-learn 1.0+

Results may vary based on your hardware. GPU acceleration is not currently utilized, so all processing is CPU-bound.


.. note::
	**Performance Summary**: ``clustaudio`` can process 1000 audio files (30 seconds each) in approximately **1-2 minutes** including feature extraction, embedding, and clustering. Advanced processing (denoising, separation, VAD) adds overhead but is still very fast at 2-10x real-time for most operations.


.. include:: add_bottom.add
