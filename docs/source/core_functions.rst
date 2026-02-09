This section describes the core functionalities of ``clustaudio``.
Many of the functionalities are written in a generic manner which allows to be used in various applications.

Core functionalities
''''''''''''''''''''''
The are 8 core functionalities of ``clustaudio`` that allows to preprocess the input audio files, robustly determines the optimal number of clusters, and then optimize the clusters if desired.

    * fit_transform()
    * extract_voice_segments()
    * separate_foreground_background()
    * denoise()
    * enhance_atc_audio()
    * cluster()
    * find()
    * unique()
    * move_to_dir()

Fit and transform
^^^^^^^^^^^^^^^^^^^^
The *fit_transform* function allows to detect natural groups or clusters of audio files. It works using a multi-step process of pre-processing, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters are determined using well known methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
Based on the clustering results, the unique audio samples are also gathered.

Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.fit_transform`

The fit_transform contains 4 core functionalities that can also be used separately which provides more control:

    * import_data()
    * extract_feat()
    * embedding()
    * cluster()


import_data
^^^^^^^^^^^^^^
The input for the :func:`clustaudio.clustaudio.clustaudio.import_data` can have multiple forms; path to directory, list of strings and array-like input.
The following steps are used for which the parameters need to be set during initialization:

    1. Audio files are imported with specific extensions (['wav','mp3','flac','ogg','m4a','aac']),
    2. Each input audio file can be converted to mono.
    3. Audio is resampled to target sample rate (e.g., sr=22050 Hz for music, sr=16000 Hz for speech).
    4. Duration can be limited (e.g., first 30 seconds only for faster processing).
    5. Independent of the input, a dict is returned in a consistent manner.

.. code:: python

	# Initialize
	cl = clustaudio(method='mfcc', sr=22050)
	# Import data
	pathnames = cl.import_example(data='music')
	# Check whether it is dir, list of files or array-like
	X = cl.import_data(pathnames)
	print(cl.results.keys())
	# dict_keys(['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations'])
	# Note that only the keys audio, pathnames, filenames, sample_rates, and durations are filled.


extract_feat
^^^^^^^^^^^^^^
Extracting of features is performed in the :func:`clustaudio.clustaudio.clustaudio.extract_feat` function.
There are different options to extract features from audio files as listed below. Note that these settings need to be set during initialization.

    * 'mfcc' : Mel-Frequency Cepstral Coefficients (default for speech/audio)
    * 'spectral' : Spectral features (centroid, rolloff, contrast, bandwidth, ZCR, RMS)
    * 'chroma' : Chromagram features for pitch/harmony analysis
    * 'mfcc-spectral' : Combined MFCC and spectral features
    * 'tempogram' : Tempo and rhythm features
    * 'mel-spectrogram' : Mel-scaled spectrogram features
    * 'fingerprint' : Audio fingerprinting (chroma_hash, spectral_hash, chromaprint)
    * 'metadata' : Cluster by audio metadata (ID3 tags, artist, album, genre)
    * None : Use raw audio features

.. code:: python

	# Initialize
	cl = clustaudio(method='mfcc', sr=22050)
	# Import data
	pathnames = cl.import_example(data='music')
	# Check whether it is dir, list of files or array-like
	X = cl.import_data(pathnames)
	# Extract features using method
	Xfeat = cl.extract_feat(X)
	print(cl.results.keys())
	# dict_keys(['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations'])
	# At this point, the key: 'feat' is filled.


embedding
^^^^^^^^^^^^^^
The embedding is performed using t-SNE in the :func:`clustaudio.clustaudio.clustaudio.embedding` function.
The coordinates are used for visualization purposes only. However, when setting the ``cluster_space`` parameter to 'low' in the ``cluster`` function, the clustering will be performed in the low-dimensional t-SNE space.

.. code:: python

	# Initialize
	cl = clustaudio(method='mfcc', sr=22050)
	# Import data
	pathnames = cl.import_example(data='music')
	# Check whether it is dir, list of files or array-like
	X = cl.import_data(pathnames)
	# Extract features using method
	Xfeat = cl.extract_feat(X)
	# Embedding using t-SNE
	xycoord = cl.embedding(Xfeat)
	print(cl.results.keys())
	# dict_keys(['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations'])
	# At this point, the key: 'xycoord' is filled.


cluster
^^^^^^^^^
The *cluster* function is built on ``clusteval``, which is a python package that provides various evaluation methods for unsupervised cluster validation.
The optimal number of clusters are determined using well known methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
This function can be run after the ``fit_transform`` function to solely optimize the clustering results or try-out different evaluation approaches without repeatedly performing all the steps of preprocessing.
Besides changing evaluation methods and metrics, it is also possible to cluster on the low-embedded feature space. This can be done by setting the parameter ``cluster_space='low'``.

.. code:: python

	# Initialize
	cl = clustaudio(method='mfcc', sr=22050)
	# Import data
	pathnames = cl.import_example(data='music')
	# Check whether it is dir, list of files or array-like
	X = cl.import_data(pathnames)
	# Extract features using method
	Xfeat = cl.extract_feat(X)
	# Embedding using t-SNE
	xycoord = cl.embedding(Xfeat)
	# Cluster
	labels = cl.cluster(cluster='agglomerative', evaluate='silhouette', metric='cosine', linkage='ward', min_clust=3, max_clust=25, cluster_space='high')
	print(cl.results.keys())
	# dict_keys(['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations'])
	# At this point, the key: 'labels' is filled.


More examples can also be found here: :func:`clustaudio.clustaudio.clustaudio.cluster`


extract_voice_segments
^^^^^^^^^^^^^^^^^^^^^^^
To cluster voices or speakers in audio recordings, we first need to detect and extract the voice segments from the audio files.
The *extract_voice_segments* function does this task using Voice Activity Detection (VAD).
Voice segments are detected using energy-based methods, librosa onset detection, or WebRTC VAD.

This is particularly useful for:
    * Podcast speaker identification
    * ATC (Air Traffic Control) communication clustering
    * Interview transcription and analysis
    * Multi-speaker audio segmentation

.. code:: python

	from clustaudio import clustaudio

	# Initialize
	cl = clustaudio(method='mfcc', sr=16000)

	# Extract voice segments from a podcast
	segments = cl.extract_voice_segments(
	    'podcast.wav',
	    method='webrtcvad',  # or 'energy' or 'librosa'
	    min_segment_duration=0.5,
	    energy_threshold=0.02
	)

	# segments contains:
	# - pathnames_segments: List of extracted segment file paths
	# - timestamps: Start/end times for each segment
	# - durations: Duration of each segment

	# Now cluster the voice segments
	results = cl.fit_transform(segments['pathnames_segments'])

	print(f"Extracted {len(segments['pathnames_segments'])} voice segments")
	print(f"Found {len(np.unique(results['labels']))} speaker clusters")


Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.extract_voice_segments`


separate_foreground_background
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The *separate_foreground_background* function separates audio into foreground (signal/voice) and background (noise) components.
This is particularly useful for noisy communications like ATC (Air Traffic Control), radio transmissions, or any audio with significant background noise.

Unlike VAD which only detects voice, this function extracts both the signal and noise for separate analysis.

.. code:: python

	from clustaudio import clustaudio

	# Initialize
	cl = clustaudio(method='mfcc', sr=16000)

	# Separate foreground/background
	separated = cl.separate_foreground_background(
	    'noisy_communication.wav',
	    method='spectral_masking',  # or 'harmonic_percussive', 'nmf', 'wiener'
	    denoise=True,
	    save_both=True
	)

	# separated contains:
	# - pathnames_foreground: Clean voice/signal audio
	# - pathnames_background: Extracted noise audio
	# - snr_db: Signal-to-noise ratio improvements

	# Cluster foreground (voice)
	voice_results = cl.fit_transform(separated['pathnames_foreground'])

	# Cluster background (noise types)
	noise_results = cl.fit_transform(separated['pathnames_background'])

	print(f"Voice clusters: {len(np.unique(voice_results['labels']))}")
	print(f"Noise types: {len(np.unique(noise_results['labels']))}")
	print(f"Average SNR improvement: {np.mean(separated['snr_db']):.2f} dB")


Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.separate_foreground_background`


denoise
^^^^^^^^
The *denoise* function applies noise reduction to audio files using various methods.
This is useful for cleaning up noisy recordings before clustering or analysis.

Available denoising methods:
    * 'spectral' : Spectral subtraction (default, works well for stationary noise)
    * 'wiener' : Wiener filtering (adaptive noise reduction)
    * 'noisereduce' : Advanced noise reduction (requires noisereduce library)
    * 'median' : Median filtering for impulse noise
    * 'bandpass' : Band-pass filtering (good for speech: 300-3400 Hz)

.. code:: python

	from clustaudio import clustaudio

	# Initialize
	cl = clustaudio(method='spectral', sr=22050)

	# Denoise audio files
	denoised = cl.denoise(
	    pathnames,
	    method='spectral',
	    stationary=False,  # For non-stationary noise
	    save_denoised=True,
	    output_dir='denoised_audio/'
	)

	# Cluster denoised audio
	results = cl.fit_transform(denoised['pathnames_denoised'])

	# Visualize denoising effect
	cl.plot_denoise(index=0)


Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.denoise`


enhance_atc_audio
^^^^^^^^^^^^^^^^^^
The *enhance_atc_audio* function is specialized for enhancing noisy Air Traffic Control (ATC) communication audio.
This handles radio static, wind noise, engine noise, and low SNR conditions typical in aviation communications.

Enhancement levels:
    * 'light' : Minimal processing (band-pass filter only)
    * 'moderate' : Balanced enhancement (band-pass + spectral subtraction)
    * 'aggressive' : Maximum enhancement (recommended for very noisy ATC)

.. code:: python

	from clustaudio import clustaudio
	import vad

	# Initialize for speech (16kHz typical for radio)
	cl = clustaudio(method='mfcc', sr=16000)

	# Step 1: Enhance noisy ATC audio
	enhanced = cl.enhance_atc_audio(
	    './ATC_recordings/',
	    method='aggressive'
	)

	# Step 2: Separate foreground/background
	separated = cl.separate_foreground_background(
	    enhanced['pathnames_enhanced'],
	    method='spectral_masking'
	)

	# Step 3: Extract voice segments
	segments = vad.batch_extract_voice_segments(
	    separated['pathnames_foreground'],
	    method='energy'
	)

	# Step 4: Cluster by speaker/airport/controller
	pathnames = [p for seg in segments
	            for p in seg['pathnames_segments']]
	results = cl.fit_transform(pathnames)

	print(f"Found {len(np.unique(results['labels']))} ATC clusters")


Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.enhance_atc_audio`


find
^^^^^^^
The ``find`` function :func:`clustaudio.clustaudio.clustaudio.find` allows to find audio files that are similar to the input audio.
Finding similar audio can be performed in two manners:

    * Based on the k-nearest neighbour
    * Based on significance after probability density fitting

In both cases, the adjacency matrix is first computed using the distance metric (default Euclidean, but cosine recommended for audio features).
In case of the k-nearest neighbour approach, the k nearest neighbours are determined.
In case of significance, the adjacency matrix is used to estimate the best fit for the loc/scale/arg parameters across various theoretical distributions.
The tested distributions are *['norm', 'expon', 'uniform', 'gamma', 't']*. The fitted distribution is basically the similarity-distribution of samples.
For each new (unseen) input audio file, the probability of similarity is computed across all audio files, and the files are returned that are P <= *alpha* in the lower bound of the distribution.
If both *k* and *alpha* are specified, the union of detected samples is taken.
Note that the metric can be changed in this function but this may lead to confusion as the results will not intuitively match with the scatter plots as these are determined using the metric in the fit_transform() function.

Example to find similar audio files using audio waveform or features as input.

.. code:: python

	from clustaudio import clustaudio
	import matplotlib.pyplot as plt
	import pandas as pd

	# Init with default settings
	cl = clustaudio(method='mfcc', sr=22050)

	# Load example audio dataset
	pathnames = cl.import_example(data='music')

	# Cluster audio files
	results = cl.fit_transform(pathnames)

	# Find similar audio files (k-nearest neighbor)
	results_find = cl.find(pathnames[0], k=10)

	# Show whatever is found
	cl.plot_find()
	cl.scatter(zoom=3)

	# Extract the first input audio filename
	filename = [*results_find.keys()][1]

	# Plot the probabilities
	plt.figure(figsize=(8,6))
	plt.plot(results_find[filename]['y_proba'],'.')
	plt.grid(True)
	plt.xlabel('samples')
	plt.ylabel('P-value')

	# Extract the cluster labels for the input audio
	results_find[filename]['labels']

	# The majority of labels is for a specific cluster
	print(pd.value_counts(results_find[filename]['labels']))


**Example to find similar audio files based on the pathname as input.**

.. code:: python

	from clustaudio import clustaudio

	# Init with default settings
	cl = clustaudio(method='mfcc-spectral', sr=22050)

	# Load example music collection
	pathnames = cl.import_example(data='music')

	# Cluster music files
	results = cl.fit_transform(pathnames)

	# Find songs similar to a specific track
	results_find = cl.find(pathnames[10], k=20, alpha=0.05)

	# Show whatever is found
	cl.plot_find()
	cl.scatter()

	# Get similar audio files
	similar_songs = results_find[pathnames[10]]['y_pathnames']
	print(f"Found {len(similar_songs)} similar songs")


**Example: Music Recommendation System**

.. code:: python

	from clustaudio import clustaudio

	# Initialize
	cl = clustaudio(method='mfcc-spectral', sr=22050)

	# Build music library index
	results = cl.fit_transform('/music_library/')

	# Find songs similar to user's favorite
	favorite_song = 'my_favorite_track.mp3'
	similar = cl.find(favorite_song, k=20, alpha=0.1)

	# Create recommendation playlist
	recommendations = similar[favorite_song]['y_pathnames']

	# Generate playlist
	cl.create_playlist(
	    labels=similar[favorite_song]['labels'],
	    playlist_name='recommendations',
	    format='m3u8'
	)


Examples can be found here: :func:`clustaudio.clustaudio.clustaudio.find`

unique
^^^^^^^^^^
The unique audio samples can be computed using the unique :func:`clustaudio.clustaudio.clustaudio.unique` and are detected by first computing the center of the cluster, and then taking the audio file closest to the center.
Let's demonstrate this by example with a music dataset.

.. code:: python

	from clustaudio import clustaudio

	# Init with default settings
	cl = clustaudio(method='mfcc', sr=22050)

	# Load example music collection
	pathnames = cl.import_example(data='music')

	# Find natural groups of audio files
	results = cl.fit_transform(pathnames)

	# Show the unique detected audio samples
	cl.results_unique.keys()

	# Plot the audio file that is located in the center of each cluster
	cl.plot_unique(show_waveform=True)

	# Compute again with other metric if desired
	cl.unique(metric='cosine')


Preprocessing
''''''''''''''''

Audio preprocessing is performed in the :func:`clustaudio.clustaudio.clustaudio.audioread` function and contains several functions to handle the import and preprocessing of audio files.
This function requires the full path to the audio file for which the first step is reading the audio and performing sample rate conversion.

The pre-processing has 5 steps and are executed in this order:

    * 1. Import audio file
    * 2. Resample to target sample rate (user defined, e.g., sr=22050)
    * 3. Convert to mono (user defined, default=True)
    * 4. Normalize amplitude to [-1, 1] range (user defined, default=True)
    * 5. Trim silence (optional)

.. code:: python

    # Import libraries
    from clustaudio import clustaudio
    import matplotlib.pyplot as plt
    import librosa.display

    # Init
    cl = clustaudio(sr=22050, mono=True, normalize=True)
    # Load example dataset
    pathnames = cl.import_example(data='music')
    # Preprocessing of the first audio file
    audio = cl.audioread(pathnames[0])

    # Plot waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=cl.params['sr'])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Preprocessed Audio Waveform')


Output
''''''''''''''''

The results obtained from the :func:`clustaudio.clustaudio.clustaudio.fit_transform` or :func:`clustaudio.clustaudio.clustaudio.cluster` is a dictionary containing the following keys:

.. code-block:: bash

     *  audio         : Audio waveforms of the preprocessed audio files
     *  feat          : Extracted features
     *  xycoord       : X and Y coordinates from the embedding
     *  pathnames     : Absolute path location to the audio file
     *  filenames     : File names of the audio file
     *  labels        : Cluster labels
     *  sample_rates  : Sample rates of audio files
     *  durations     : Duration of each audio file (seconds)

For demonstration purposes I will load a music dataset and cluster the audio files.

.. code:: python

    # Import library
    from clustaudio import clustaudio
    # Initialize
    cl = clustaudio(method='mfcc', sr=22050)
    # Import data
    pathnames = cl.import_example(data='music')
    # Cluster audio files
    results = cl.fit_transform(pathnames)

.. code:: python

    # All results are stored in a dict:
    print(cl.results.keys())
    # Which is the same as:
    print(results.keys())
    # dict_keys(['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations'])


Extract audio files that belong to a certain cluster and make some plots.

.. code:: python

    # Extracting audio files that belong to cluster label=0:
    label = 0
    Iloc = cl.results['labels']==label
    pathnames = cl.results['pathnames'][Iloc]

    # Extracting xy-coordinates for the scatterplot for cluster 0:
    import matplotlib.pyplot as plt
    xycoord = cl.results['xycoord'][Iloc]
    plt.figure()
    plt.scatter(xycoord[:,0], xycoord[:,1])
    plt.title('Cluster %.0d' %label)

    # Get audio for cluster 0:
    audio_files = cl.results['audio'][Iloc]

    # Plot waveforms for cluster 0:
    import librosa.display
    for i, (audio, pathname) in enumerate(zip(audio_files, pathnames)):
        if i >= 3:  # Limit to first 3 for demonstration
            break
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=cl.params['sr'])
        plt.title(pathname)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')


Generic functions
''''''''''''''''''''''''
``clustaudio`` contains various generic functionalities that are internally used but may be useful too in other applications.

wget
^^^^^^^^^
Download files from the internet and store on disk.
Examples can be found here: :func:`clustaudio.clustaudio.wget`

.. code:: python

    # Import library
    import clustaudio as cl
    # Download
    audio_files = cl.wget('https://example.com/audio_dataset.zip', 'c://temp//audio_dataset.zip')


unzip
^^^^^^^^^
Unzip files into a destination directory.
Examples can be found here: :func:`clustaudio.clustaudio.unzip`

.. code:: python

    # Import library
    import clustaudio as cl
    # Unzip to path
    dirpath = cl.unzip('c://temp//audio_dataset.zip')


listdir
^^^^^^^^^
Recursively list the audio files in the directory.
Examples can be found here: :func:`clustaudio.clustaudio.listdir`

.. code:: python

    # Import library
    from clustaudio import listdir
    # List audio files in directory
    dirpath = 'c://temp//audio_dataset'
    pathnames = listdir(dirpath, ext=['wav', 'mp3', 'flac'])


set_logger
^^^^^^^^^^^^
Change status of the logger.
Examples can be found here: :func:`clustaudio.clustaudio.set_logger`

.. code:: python

    # Change to verbosity message of warnings and higher
    from clustaudio import set_logger
    set_logger(verbose=30)


extract_mfcc
^^^^^^^^^^^^^
Mel-Frequency Cepstral Coefficients (MFCC) are the primary audio features used for clustering.
Examples: :func:`clustaudio.clustaudio.clustaudio.extract_mfcc` More detailed explanation can be found in the **Feature Extraction** - **MFCC** section.


extract_spectral
^^^^^^^^^^^^^^^^^
Spectral features capture timbral characteristics of audio.
Examples: :func:`clustaudio.clustaudio.clustaudio.extract_spectral` More detailed explanation can be found in the **Feature Extraction** - **Spectral Features** section.



Merge/ Expand Clusters
''''''''''''''''''''''''''''''
The number of clusters are optimized using the clusteval library. However, when desired it is also possible to manually merge or expand the number of clusters.


Optimized Clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

	import numpy as np
	import matplotlib.pyplot as plt
	from clustaudio import clustaudio

	# Initialize
	cl = clustaudio(method='mfcc', sr=22050)

	# Import data
	pathnames = cl.import_example(data='music')
	# Fit transform
	cl.fit_transform(pathnames)
	# Check number of clusters
	len(np.unique(cl.results['labels']))
	# 9

	# Scatter
	cl.scatter(dotsize=75)
	# Create dendrogram
	cl.dendrogram();


Force to K clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's merge some of the clusters and set it to 5 clusters.

.. code:: python

	# Set to 5 clusters
	labels = cl.cluster(min_clust=5, max_clust=5)

	# Check number of clusters
	len(np.unique(cl.results['labels']))
	# 5

	# Scatter
	cl.scatter(dotsize=75)
	# Create dendrogram
	cl.dendrogram();


Set clusters by dendrogram threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another manner to change the number of clusters is by specifying the height of the dendrogram (setting a threshold point or cut-off). The number of clusters is automatically derived from that point.

.. code:: python

	# Look at the dendrogram y-axis and specify the height to merge clusters
	dendro_results = cl.dendrogram(max_d=60000)

	# Check number of clusters
	len(np.unique(cl.results['labels']))
	# 3

	# Scatter
	cl.scatter(dotsize=75)


Selection on the cluster labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

	# All results are stored in cl.results
	cl.results.keys()
	['audio', 'feat', 'xycoord', 'pathnames', 'labels', 'filenames', 'sample_rates', 'durations', 'predict']

	# The cluster labels are stored in labels
	cl.results['labels']

	# Select cluster 0
	Iloc = cl.results['labels']==0

	# Select files for cluster 0
	cl.results['pathnames'][Iloc]

	# Select filenames for cluster 0
	cl.results['filenames'][Iloc]

	# Select xy-coordinates for cluster 0
	cl.results['xycoord'][Iloc]


Move files based on cluster labels
''''''''''''''''''''''''''''''''''
The ``move_to_dir`` function physically moves or copies audio files into subdirectories based on cluster labels.
First perform the desired clustering approach and with the following steps it allows you to organize the audio files by physically moving them into subdirectories.

Additionally, playlists can be automatically generated for each cluster.

.. code:: python

    # Show the cluster labels
    print(cl.results['labels'])

    # Step 1. Use the plot function to determine what each cluster represents.
    # Do not plot cluster 0 (rest group), and only plot when a cluster contains 3 or more audio files.
    cl.plot(blacklist=[0], min_samples=3, show_waveform=True)

    # Step 2. Create a dict that specifies the cluster number with its folder names.
    # The first column is the cluster label and the second string is the destination subfolder name.
    # Look at the clusters and specify the subdirectories. For example:
    # cluster 0 might be rock music, cluster 1 might be jazz, cluster 2 might be classical
    # All audio files that are not in the specified clusters will remain untouched.
    target_labels = {
        0: 'rock',
        1: 'jazz',
        2: 'classical',
        3: 'electronic',
    }

    # Step 3. Move/copy the audio files to the specified directories using the cluster labels.
    # Also create playlists for each cluster
    cl.move_to_dir(
        target_labels=target_labels,
        action='copy',  # or 'move'
        create_playlists=True
    )


**Example: Organize ATC Communications**

.. code:: python

	from clustaudio import clustaudio
	import vad

	# Initialize
	cl = clustaudio(method='mfcc', sr=16000)

	# Process ATC recordings
	enhanced = cl.enhance_atc_audio('./ATC/', method='aggressive')
	separated = cl.separate_foreground_background(enhanced['pathnames_enhanced'])
	segments = vad.batch_extract_voice_segments(separated['pathnames_foreground'])

	# Cluster voice segments
	pathnames = [p for seg in segments for p in seg['pathnames_segments']]
	results = cl.fit_transform(pathnames)

	# Organize by identified clusters (controllers, airports, etc.)
	target_labels = {
	    0: 'tower_controllers',
	    1: 'pilot_transmissions',
	    2: 'ground_control',
	    3: 'approach_control',
	}

	cl.move_to_dir(target_labels=target_labels, create_playlists=True)


Plotting
''''''''''''''''''''''''''''''''''
clustaudio has various plotting functionalities designed for audio visualization and analysis:

    * **plot**: Plots waveforms and/or spectrograms of audio files in clusters
    * **plot_find**: Plot the input audio together with the predicted similar audio files
    * **plot_unique**: Plot the unique/representative audio samples from each cluster
    * **dendrogram**: Plot hierarchical clustering dendrogram
    * **scatter**: Scatter plot of embedded features with cluster colors
    * **plot_separation**: Visualize foreground/background separation results
    * **plot_denoise**: Compare original and denoised audio
    * **plot_statistics**: Comprehensive cluster statistics and distributions

**Example: Comprehensive Audio Visualization**

.. code:: python

	from clustaudio import clustaudio

	# Initialize and cluster
	cl = clustaudio(method='mfcc-spectral', sr=22050)
	results = cl.fit_transform('/music/')

	# Plot waveforms for clusters
	cl.plot(labels=[0, 1, 2], show_waveform=True, show_spectrogram=True)

	# Scatter plot
	cl.scatter(dotsize=50, zoom=1.0)

	# Dendrogram
	cl.dendrogram()

	# Cluster statistics
	cl.plot_statistics()

	# Find similar audio and visualize
	similar = cl.find('query_song.mp3', k=10)
	cl.plot_find()


.. include:: add_bottom.add
