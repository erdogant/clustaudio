
MFCC
''''''''''

Mel-Frequency Cepstral Coefficients (MFCC) is the primary feature extraction method for audio analysis and speech processing.
In ``clustaudio`` the MFCC features are extracted using the `librosa`_ library, providing a compact representation of the audio's spectral characteristics.
MFCCs capture the timbral characteristics of audio and are widely used in speech recognition, music genre classification, and audio similarity detection.

The MFCC feature descriptor represents the short-term power spectrum of sound, approximating the human auditory system's response.
It is computed by:

 * Converting the audio signal to a mel-frequency spectrogram
 * Applying the Discrete Cosine Transform (DCT) to the mel-spectrogram
 * Extracting typically 13-40 coefficients (default: 13)
 * Computing mean and standard deviation statistics over time

MFCCs are particularly effective for:
 * Speech and voice analysis (speaker identification, emotion recognition)
 * Music analysis (genre classification, similarity detection)
 * General audio clustering and organization


.. code:: python

	# Initialize with MFCC and 13 coefficients (default)
	cl = clustaudio(method='mfcc', params_mfcc={'n_mfcc': 13})
	
	# Use 20 coefficients for more detailed representation
	cl = clustaudio(method='mfcc', params_mfcc={'n_mfcc': 20, 'n_mels': 128})

	# Load example audio data
	pathnames = cl.import_example(data='music')

	# Import audio files
	X = cl.import_data(pathnames)
	
	# Extract features using method
	Xfeat = cl.extract_feat(X)
	
	# Alternatively, the features are also stored in the results dict
	cl.results['feat']

	# Alternatively, the features are also stored in the results dict using the run-at-once function.
	results = cl.fit_transform(pathnames)
	
	# Extracted MFCC features
	results['feat']


**MFCC Parameters:**

The input parameters for MFCC extraction can be configured via ``params_mfcc``:

    * ``n_mfcc``: Number of MFCC coefficients to extract (default: 13, typical range: 13-40)
    * ``n_fft``: FFT window size (default: 2048)
    * ``hop_length``: Number of samples between successive frames (default: 512)
    * ``n_mels``: Number of mel frequency bands (default: 128)
    * ``fmin``: Minimum frequency in Hz (default: 0)
    * ``fmax``: Maximum frequency in Hz (default: None, which uses sr/2)


.. code:: python

	# Advanced MFCC configuration
	cl = clustaudio(
	    method='mfcc',
	    sr=22050,  # Sample rate
	    params_mfcc={
	        'n_mfcc': 20,      # More coefficients for detail
	        'n_fft': 2048,     # FFT window size
	        'hop_length': 512, # Frame hop
	        'n_mels': 128,     # Mel bands
	        'fmin': 0,         # Min frequency
	        'fmax': 8000       # Max frequency (useful for speech)
	    }
	)
	
	# Load and cluster
	results = cl.fit_transform('/path/to/audio/files/')


Spectral Features
''''''''''''''''''

Spectral features capture the frequency domain characteristics of audio signals. These features describe the distribution and dynamics of energy across different frequencies.
In ``clustaudio``, spectral features include:

 * **Spectral Centroid**: The "center of mass" of the spectrum, indicating brightness
 * **Spectral Rolloff**: The frequency below which 85% of spectral energy is contained
 * **Spectral Contrast**: The difference between peaks and valleys in the spectrum
 * **Spectral Bandwidth**: The spread of frequencies around the centroid
 * **Zero Crossing Rate**: Rate at which the signal changes sign, indicating percussiveness
 * **RMS Energy**: Root Mean Square energy, representing overall loudness

These features are effective for:
 * Timbral similarity and texture analysis
 * Music genre classification
 * Audio quality assessment
 * Distinguishing between different sound types


.. code:: python

	# Initialize with spectral features
	cl = clustaudio(method='spectral', sr=22050)

	# Load example audio data
	pathnames = cl.import_example(data='environmental_sounds')

	# Extract spectral features
	results = cl.fit_transform(pathnames)
	
	# The features contain: centroid, rolloff, contrast, bandwidth, ZCR, RMS
	# Each with mean and std statistics = 12 features total
	print(f"Feature shape: {results['feat'].shape}")


**Spectral Parameters:**

The input parameters for spectral features can be configured via ``params_spectral``:

    * ``n_fft``: FFT window size (default: 2048)
    * ``hop_length``: Hop length between frames (default: 512)


Chroma Features
'''''''''''''''

Chroma features (chromagram or pitch class profiles) represent the 12 different pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).
These features are particularly useful for music analysis where harmonic and melodic characteristics are important.

Chroma features capture:
 * Harmonic content of the audio
 * Pitch class distribution over time
 * Musical similarity independent of timbre

Applications include:
 * Music similarity and cover song detection
 * Chord recognition
 * Key detection
 * Harmonic analysis


.. code:: python

	# Initialize with chroma features
	cl = clustaudio(method='chroma', params_chroma={'n_chroma': 12})

	# Load music files
	pathnames = listdir('/path/to/music/', ext=['mp3', 'wav'])

	# Extract chroma features
	results = cl.fit_transform(pathnames)
	
	# Features include mean and std for each of 12 pitch classes = 24 features
	print(f"Chroma features: {results['feat'].shape}")


**Chroma Parameters:**

    * ``n_chroma``: Number of chroma bins (default: 12 for 12 pitch classes)
    * ``n_fft``: FFT window size (default: 2048)
    * ``hop_length``: Hop length (default: 512)


Combined Features
'''''''''''''''''

For comprehensive audio analysis, ``clustaudio`` supports combining multiple feature types. The most effective combination is MFCC with spectral features, providing both cepstral and frequency-domain characteristics.


.. code:: python

	# Initialize with combined MFCC and spectral features
	cl = clustaudio(method='mfcc-spectral', sr=22050)

	# This extracts both MFCC (26 features: 13 coeffs × 2 stats)
	# and spectral (12 features: 6 types × 2 stats)
	# Total: 38 features

	# Load and cluster
	results = cl.fit_transform('/path/to/audio/')
	
	print(f"Combined features: {results['feat'].shape}")
	# Output: (n_files, 38)


Another example combining MFCC with chroma:

.. code:: python

	# Initialize with MFCC
	cl = clustaudio(method='mfcc', sr=22050)
	
	# Load audio
	pathnames = listdir('/music/', ext=['mp3', 'flac'])
	X = cl.import_data(pathnames)
	
	# Extract MFCC
	mfcc_feat = cl.extract_mfcc(X['audio'])
	
	# Extract chroma
	chroma_feat = cl.extract_chroma(X['audio'])
	
	# Combine manually
	import numpy as np
	combined_feat = np.hstack([mfcc_feat, chroma_feat])
	
	print(f"MFCC shape: {mfcc_feat.shape}")
	print(f"Chroma shape: {chroma_feat.shape}")
	print(f"Combined shape: {combined_feat.shape}")


Audio Fingerprinting
''''''''''''''''''''

Audio fingerprinting creates compact perceptual hashes for duplicate detection and similarity matching. Unlike MFCC or spectral features which capture acoustic properties, fingerprints are designed to be robust to encoding variations, compression, and minor distortions.

``clustaudio`` supports multiple fingerprinting methods:

 * **chroma_hash**: Chroma-based perceptual hashing (default)
 * **spectral_hash**: Mel-spectrogram based hashing
 * **chromaprint**: Industry-standard Chromaprint/AcoustID (requires installation)

Applications:
 * Finding exact duplicates (different encodings, bitrates)
 * Detecting near-duplicates (covers, remixes)
 * Fast similarity comparison
 * Copyright detection


.. code:: python

	import fingerprint
	from clustaudio import listdir
	
	# Get audio files
	pathnames = listdir('/music_library/', ext=['mp3', 'flac', 'wav'])
	
	# Compute fingerprints for all files
	fps = fingerprint.batch_compute_fingerprints(
	    pathnames,
	    method='chroma_hash'  # or 'chromaprint' if installed
	)
	
	# Find duplicates (threshold: 0 = exact, 1 = very different)
	duplicates = fingerprint.find_duplicates(fps, threshold=0.1)
	
	# Review duplicate groups
	for group in duplicates:
	    print("Duplicates found:")
	    for idx in group:
	        print(f"  - {pathnames[idx]}")


**Comparing specific files:**

.. code:: python

	import fingerprint
	
	# Compute fingerprints
	fp1 = fingerprint.compute_fingerprint(filepath='song1.mp3', method='chroma_hash')
	fp2 = fingerprint.compute_fingerprint(filepath='song2.mp3', method='chroma_hash')
	
	# Compare (returns similarity score: 0 = identical, 1 = very different)
	similarity = fingerprint.compare_fingerprints(fp1, fp2, method='hamming')
	
	if similarity < 0.1:
	    print("Files are duplicates or very similar")
	elif similarity < 0.3:
	    print("Files are similar (possibly cover or remix)")
	else:
	    print("Files are different")


**Integration with clustering:**

.. code:: python

	from clustaudio import clustaudio
	import fingerprint
	import numpy as np
	
	# Compute fingerprints
	pathnames = listdir('/music/', ext=['mp3'])
	fps = fingerprint.batch_compute_fingerprints(pathnames)
	
	# Compute distance matrix for clustering
	dist_matrix = fingerprint.compute_fingerprint_matrix(fps, method='hamming')
	
	# Use with clusteval for clustering
	from clusteval import clusteval
	ce = clusteval()
	ce.fit(dist_matrix)
	ce.plot()
	ce.scatter()


Metadata Features
'''''''''''''''''

Audio metadata (ID3 tags for MP3, Vorbis comments for OGG/FLAC, MP4 metadata) contains valuable information that can be used for clustering audio files, especially for organizing music libraries.

Metadata fields include:
 * **Artist, Album, Title**: Track identification
 * **Genre, Year**: Musical categorization
 * **Duration, Bitrate, Sample Rate**: Technical properties
 * **Track Number**: Album organization

This approach is particularly useful for:
 * Organizing music libraries by artist/album
 * Grouping by genre or year
 * Finding files with missing or inconsistent metadata


.. code:: python

	import metadata
	
	# Extract metadata from a single file
	meta = metadata.extract_metadata('song.mp3')
	print(f"Artist: {meta['artist']}")
	print(f"Album: {meta['album']}")
	print(f"Genre: {meta['genre']}")
	print(f"Duration: {meta['duration']:.2f} seconds")


**Batch metadata extraction:**

.. code:: python

	import metadata
	from clustaudio import listdir
	
	# Get all audio files
	pathnames = listdir('/music/', ext=['mp3', 'flac', 'ogg', 'm4a'])
	
	# Extract metadata from all files
	df = metadata.extract_metadata_batch(pathnames)
	
	# View results
	print(df[['filename', 'artist', 'album', 'genre', 'year']].head())
	
	# Check metadata coverage
	print(f"\nMetadata coverage:")
	print(f"Artist: {df['artist'].notna().sum()}/{len(df)}")
	print(f"Album: {df['album'].notna().sum()}/{len(df)}")
	print(f"Genre: {df['genre'].notna().sum()}/{len(df)}")


**Clustering by metadata:**

.. code:: python

	import metadata
	
	# Extract metadata
	df = metadata.extract_metadata_batch(pathnames)
	
	# Cluster by artist and album
	labels = metadata.cluster_by_metadata(
	    df,
	    cluster_by=['artist', 'album'],
	    similarity_threshold=0.8
	)
	
	# View clusters
	for label in np.unique(labels):
	    cluster_files = df[labels == label]
	    print(f"\nCluster {label}:")
	    print(cluster_files[['artist', 'album', 'filename']].head())


**Integration with main class:**

.. code:: python

	from clustaudio import clustaudio
	
	# Initialize with metadata method
	cl = clustaudio(method='metadata')
	
	# Automatically clusters by artist/album/genre
	results = cl.fit_transform('/music_library/')
	
	# Results contain cluster labels based on metadata
	print(f"Found {len(np.unique(results['labels']))} groups")
	
	# Organize files
	cl.move_to_dir(create_playlists=True)


**Converting metadata to ML features:**

For use with traditional clustering algorithms, metadata can be converted to numerical features:

.. code:: python

	import metadata
	
	# Extract metadata
	df = metadata.extract_metadata_batch(pathnames)
	
	# Convert to numerical features
	# Includes: duration, bitrate, year (normalized)
	# Plus one-hot encoding for top 20 artists/albums/genres
	features = metadata.create_metadata_features(df, encode_categorical=True)
	
	# Use with clustaudio
	cl = clustaudio(method='mfcc')  # Or any method
	
	# Use metadata features directly
	results = cl.cluster(features, min_clust=5, max_clust=20)


Feature Comparison
''''''''''''''''''

Choosing the right feature extraction method depends on your specific use case:

.. list-table:: Feature Method Comparison
   :widths: 20 25 15 15 25
   :header-rows: 1

   * - Method
     - Best For
     - Speed
     - Features
     - Use Case
   * - MFCC
     - General audio, speech
     - Fast
     - 26 (default)
     - Voice clustering, general purpose
   * - Spectral
     - Timbre, texture
     - Fast
     - 12
     - Sound classification, quality
   * - Chroma
     - Music, harmony
     - Medium
     - 24
     - Music similarity, covers
   * - MFCC-Spectral
     - Comprehensive
     - Medium
     - 38
     - Best overall performance
   * - Fingerprint
     - Duplicates
     - Fast
     - 32-512 bits
     - Duplicate detection
   * - Metadata
     - Organization
     - Very Fast
     - Varies
     - Library organization


**Recommendations:**

* **For speech/podcasts**: Use MFCC with sr=16000
* **For music similarity**: Use MFCC-Spectral or Chroma
* **For duplicate detection**: Use fingerprinting
* **For library organization**: Use metadata clustering
* **For noisy audio (ATC, radio)**: Use MFCC with enhancement (see below)
* **For best overall results**: Use MFCC-Spectral with sr=22050


Advanced: ATC and Noisy Audio
''''''''''''''''''''''''''''''

For challenging audio scenarios like Air Traffic Control communications, radio transmissions, or other noisy environments, ``clustaudio`` provides specialized enhancement before feature extraction:


.. code:: python

	from clustaudio import clustaudio
	import vad
	
	# Initialize for speech (16kHz typical for radio)
	cl = clustaudio(method='mfcc', sr=16000)
	
	# Step 1: Enhance noisy ATC audio
	enhanced = cl.enhance_atc_audio(
	    pathnames,
	    method='aggressive'  # 'light', 'moderate', or 'aggressive'
	)
	
	# Step 2: Separate voice from noise
	separated = cl.separate_foreground_background(
	    enhanced['pathnames_enhanced'],
	    method='spectral_masking'
	)
	
	# Step 3: Extract voice segments
	segments = vad.batch_extract_voice_segments(
	    separated['pathnames_foreground'],
	    method='energy'
	)
	
	# Step 4: Cluster voice segments
	pathnames_voice = [
	    p for seg in segments 
	    if seg.get("pathnames_segments")
	    for p in seg["pathnames_segments"]
	]
	
	results = cl.fit_transform(pathnames_voice)
	
	print(f"Clustered {len(pathnames_voice)} voice segments")
	print(f"Found {len(np.unique(results['labels']))} speaker/controller groups")





.. _librosa: https://librosa.org/


.. include:: add_bottom.add
