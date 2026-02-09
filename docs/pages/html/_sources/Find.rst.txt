This section describes how to find similar audio files using a readily fitted model.

The ``find`` function :func:`clustaudio.clustaudio.clustaudio.find` allows finding audio files that are similar to what has already been seen by the model. Finding similar audio can be performed in two manners as described below. In both cases, the adjacency matrix is first computed using the distance metric (default: cosine for audio features).

k-nearest neighbour
'''''''''''''''''''
The k-nearest neighbour approach searches the k nearest neighbours to that of the input audio file using the specified distance metric (default: cosine for MFCC/spectral features). This approach returns distances to the closest neighbors. If both *k* and *alpha* are specified, the union of detected samples is taken.

Example to find similar audio samples for unseen files using k-nearest neighbour approach.

.. code:: python

	from clustaudio import clustaudio
	import numpy as np

	# Init with MFCC features
	cl = clustaudio(method='mfcc', sr=22050)
	
	# Load music library
	pathnames = cl.import_example(data='music')

	# Build the audio database
	results = cl.fit_transform(pathnames)
	
	# Results are also stored in object results
	cl.results.keys()
	
	# Scatter results
	cl.scatter(zoom=3, dotsize=50, figsize=(25, 15), legend=False, text=False)

	# Find similar audio files for a query song
	query_audio = 'path/to/my_favorite_song.mp3'
	results_find = cl.find(query_audio, k=10, alpha=None)
	
	# Make scatter showing the query and similar files
	cl.scatter(zoom=5, dotsize=100, text=False, figsize=(35, 20))

	# Print results for first query
	keys = list(results_find.keys())
	print(results_find.get(keys[0]).columns)
	# ['y_idx', 'distance', 'y_proba', 'labels', 'y_filenames', 'y_pathnames', 'x_pathnames']

	print(results_find.get(keys[0])[['labels', 'distance','y_proba']])
	#    labels    distance  y_proba
	# 0       3     0.124536      NaN
	# 1       3     0.185432      NaN
	# 2       3     0.203891      NaN
	# 3       3     0.231056      NaN
	# 4       5     0.267834      NaN
	# 5       3     0.289421      NaN
	# 6       5     0.302187      NaN
	# 7       3     0.318942      NaN
	# 8       5     0.334502      NaN
	# 9       3     0.351234      NaN

	# Get most common cluster label
	key = keys[0]
	uiy, ycounts = np.unique(results_find.get(key)['labels'], return_counts=True)
	y_predict = uiy[np.argmax(ycounts)]
	print(f'Most similar cluster: {y_predict}')
	
	# Get the actual filenames of similar audio
	similar_files = results_find.get(key)['y_filenames']
	print(f"\nTop 10 similar audio files:")
	for i, filename in enumerate(similar_files[:10], 1):
	    distance = results_find.get(key)['distance'].iloc[i-1]
	    print(f"  {i}. {filename} (distance: {distance:.4f})")


**Music recommendation example:**

.. code:: python

	from clustaudio import clustaudio
	
	# Build music library index
	cl = clustaudio(method='mfcc-spectral', sr=22050, duration=30)
	results = cl.fit_transform('/my_music_library/')
	
	# Find songs similar to favorites
	favorite_song = '/my_music_library/favorite_track.mp3'
	similar = cl.find(favorite_song, k=20)
	
	# Create discovery playlist
	similar_paths = similar[favorite_song.split('/')[-1]]['y_pathnames']
	
	print(f"Songs similar to '{favorite_song}':")
	for i, path in enumerate(similar_paths[:10], 1):
	    print(f"  {i}. {os.path.basename(path)}")


Probability density fitting
'''''''''''''''''''''''''''
The probability density fitting method fits a model on the input features to determine the loc/scale/arg parameters across various theoretical distributions. For audio features (MFCC, spectral, etc.), these are the feature vectors. The tested distributions are *['norm', 'expon', 'uniform', 'gamma', 't']*. The fitted distribution represents the similarity-distribution of audio samples.

For each new (unseen) input audio file, the probability of similarity is computed across all audio files in the database, and files with P <= *alpha* (lower bound) are returned. This approach identifies statistically significant matches rather than just the closest neighbors.

Note that the metric can be changed in this function but this may lead to confusion as the results will not intuitively match with the scatter plots, which are determined using the metric in the fit_transform() function.

Example to find similar audio samples using probability density fitting.

.. code:: python

	from clustaudio import clustaudio
	import numpy as np

	# Init with MFCC features
	cl = clustaudio(method='mfcc', sr=22050)
	
	# Load audio dataset
	pathnames = cl.import_example(data='music')

	# Build audio database
	results = cl.fit_transform(pathnames)
	
	# Results are stored in object results
	cl.results.keys()
	
	# Scatter results
	cl.scatter(zoom=3, dotsize=50, figsize=(25, 15), legend=False, text=False)

	# Find statistically similar audio files (alpha=0.05 means 5% significance)
	query_audio = 'query_track.mp3'
	results_find = cl.find(query_audio, alpha=0.05)
	
	# Make scatter
	cl.scatter(zoom=5, dotsize=100, text=False, figsize=(35, 20))

	# Print first key
	keys = list(results_find.keys())
	print(results_find.get(keys[0]).columns)
	# ['y_idx', 'distance', 'y_proba', 'labels', 'y_filenames', 'y_pathnames', 'x_pathnames']
	
	print(results_find.get(keys[0])[['labels', 'distance','y_proba']])

	#     labels    distance   y_proba
	# 0        2     0.142156  0.000035
	# 1        2     0.215290  0.000822
	# 2        2     0.268588  0.001812
	# 3        2     0.291933  0.001956
	# 4        2     0.321231  0.002412
	# ..     ...         ...       ...
	# 42       2     0.806617  0.044141
	# 43       2     0.817522  0.044750
	# 44       2     0.828852  0.045657
	# 45       2     0.839517  0.046116
	# 46       2     0.851862  0.047765

	# Get most common cluster label
	key = keys[0]
	uiy, ycounts = np.unique(results_find.get(key)['labels'], return_counts=True)
	y_predict = uiy[np.argmax(ycounts)]
	print(f'Most similar cluster: {y_predict}')
	print(f'Number of statistically similar files: {len(results_find.get(key))}')


**Combining k-NN and probability:**

.. code:: python

	from clustaudio import clustaudio
	
	cl = clustaudio(method='mfcc-spectral')
	results = cl.fit_transform('/audio_library/')
	
	# Find both top-10 nearest AND statistically significant (union)
	query = 'query_audio.wav'
	similar = cl.find(query, k=10, alpha=0.05)
	
	# This returns the union of:
	# - Top 10 nearest neighbors
	# - All files with p-value <= 0.05
	
	result_key = list(similar.keys())[0]
	n_similar = len(similar[result_key])
	print(f"Found {n_similar} similar audio files")


More examples
'''''''''''''

**Find similar music tracks:**

.. code:: python

	from clustaudio import clustaudio
	import matplotlib.pyplot as plt
	import pandas as pd

	# Init with combined features for best music analysis
	cl = clustaudio(method='mfcc-spectral', sr=22050, duration=30)

	# Load music library
	music_files = listdir('/music/', ext=['mp3', 'flac', 'wav'])

	# Cluster music tracks
	results = cl.fit_transform(music_files)

	# Query: Find songs similar to a specific track
	query_track = music_files[0]
	
	# Find similar tracks
	results_find = cl.find(query_track, k=15, alpha=0.05)

	# Visualize results
	cl.plot_find()
	cl.scatter(zoom=3)

	# Extract results for the query
	filename = list(results_find.keys())[0]

	# Plot the probability distribution
	plt.figure(figsize=(10, 6))
	plt.plot(results_find[filename]['y_proba'], '.')
	plt.grid(True)
	plt.xlabel('Sample index')
	plt.ylabel('P-value')
	plt.title(f'Similarity probability for {filename}')
	plt.axhline(y=0.05, color='r', linestyle='--', label='alpha=0.05')
	plt.legend()

	# Extract cluster labels for similar files
	similar_labels = results_find[filename]['labels']
	
	# The majority label indicates the genre/mood cluster
	print(pd.value_counts(similar_labels))
	# Example output:
	# 3    42
	# 5     8
	# 2     3
	# Name: labels, dtype: int64
	# This means cluster 3 is dominant (likely same genre/mood)


**Speaker identification in podcast:**

.. code:: python

	from clustaudio import clustaudio
	import vad
	
	# Initialize for speech
	cl = clustaudio(method='mfcc', sr=16000)
	
	# Extract voice segments from podcast
	segments = cl.extract_voice_segments(
	    'podcast_episode.wav',
	    method='webrtcvad'
	)
	
	# Cluster speakers
	results = cl.fit_transform(segments['pathnames_segments'])
	
	# Find all segments similar to a known speaker
	known_speaker_segment = segments['pathnames_segments'][5]
	similar_segments = cl.find(known_speaker_segment, k=None, alpha=0.01)
	
	# Get all segments from the same speaker
	speaker_key = list(similar_segments.keys())[0]
	speaker_files = similar_segments[speaker_key]['y_pathnames']
	
	print(f"Found {len(speaker_files)} segments from the same speaker")
	
	# Create playlist of this speaker's segments
	from clustaudio import create_playlist
	playlist_path = '/speaker_segments.m3u8'
	with open(playlist_path, 'w') as f:
	    f.write('#EXTM3U\n')
	    for path in speaker_files:
	        f.write(f'{path}\n')


**Environmental sound classification:**

.. code:: python

	from clustaudio import clustaudio
	
	# Initialize with spectral features (good for environmental sounds)
	cl = clustaudio(method='spectral', sr=22050)
	
	# Load environmental recordings
	env_sounds = listdir('/environmental_audio/', ext=['wav'])
	
	# Cluster sounds
	results = cl.fit_transform(env_sounds)
	
	# Find sounds similar to a bird chirp
	bird_sound = '/environmental_audio/bird_chirp_sample.wav'
	similar_sounds = cl.find(bird_sound, k=20, alpha=0.05)
	
	# Analyze results
	bird_key = list(similar_sounds.keys())[0]
	
	print(f"Found {len(similar_sounds[bird_key])} similar sounds")
	print(f"\nCluster distribution:")
	print(pd.value_counts(similar_sounds[bird_key]['labels']))
	
	# Get file paths
	similar_files = similar_sounds[bird_key]['y_pathnames']
	for i, file in enumerate(similar_files[:10], 1):
	    distance = similar_sounds[bird_key]['distance'].iloc[i-1]
	    print(f"{i}. {os.path.basename(file)} (distance: {distance:.4f})")


**Duplicate and near-duplicate detection:**

.. code:: python

	import fingerprint
	from clustaudio import clustaudio
	
	# For exact duplicates, use fingerprinting
	pathnames = listdir('/music/', ext=['mp3', 'flac'])
	fps = fingerprint.batch_compute_fingerprints(pathnames)
	duplicates = fingerprint.find_duplicates(fps, threshold=0.05)
	
	print(f"Found {len(duplicates)} groups of duplicates")
	
	# For near-duplicates (covers, remixes), use find()
	cl = clustaudio(method='mfcc-spectral')
	results = cl.fit_transform(pathnames)
	
	# Find covers/remixes of a song
	original = '/music/original_song.mp3'
	similar = cl.find(original, k=50, alpha=0.1)
	
	# Filter by distance threshold for near-duplicates
	result_key = list(similar.keys())[0]
	near_duplicates = similar[result_key][similar[result_key]['distance'] < 0.3]
	
	print(f"\nFound {len(near_duplicates)} near-duplicates (possible covers/remixes)")
	for idx, row in near_duplicates.iterrows():
	    print(f"  {row['y_filenames']} (distance: {row['distance']:.4f})")


ATC Speaker Clustering Example
'''''''''''''''''''''''''''''''

Here's a complete example for identifying controllers/pilots in Air Traffic Control communications:

.. code:: python

	from clustaudio import clustaudio, listdir
	import vad
	import numpy as np
	
	# Initialize for radio speech
	cl = clustaudio(method='mfcc', sr=16000, mono=True)
	
	# Get ATC recordings
	atc_files = listdir('/ATC_recordings/', ext=['wav'])
	
	# Enhance noisy audio
	enhanced = cl.enhance_atc_audio(atc_files, method='aggressive')
	
	# Separate voice from noise
	separated = cl.separate_foreground_background(
	    enhanced['pathnames_enhanced'],
	    method='spectral_masking'
	)
	
	# Extract voice segments
	segments = vad.batch_extract_voice_segments(
	    separated['pathnames_foreground'],
	    method='energy'
	)
	
	# Collect segments
	voice_segments = [
	    p for seg in segments 
	    if seg.get("pathnames_segments")
	    for p in seg["pathnames_segments"]
	]
	
	# Cluster speakers
	results = cl.fit_transform(voice_segments)
	
	# Identify segments from a known controller
	known_controller = voice_segments[15]  # Example known segment
	
	# Find all segments from the same speaker
	same_speaker = cl.find(
	    known_controller,
	    k=None,           # No k limit
	    alpha=0.01        # Strict similarity
	)
	
	# Analyze results
	speaker_key = list(same_speaker.keys())[0]
	n_segments = len(same_speaker[speaker_key])
	
	print(f"Found {n_segments} segments from the same speaker")
	print(f"Cluster: {same_speaker[speaker_key]['labels'].iloc[0]}")
	
	# Create timeline of this speaker's transmissions
	speaker_files = same_speaker[speaker_key]['y_pathnames']
	timestamps = []
	
	for filepath in speaker_files:
	    # Extract timestamp from filename or file metadata
	    # timestamps.append(extract_timestamp(filepath))
	    pass
	
	# Generate speaker transcript file list
	with open('controller_timeline.txt', 'w') as f:
	    for i, filepath in enumerate(speaker_files, 1):
	        f.write(f"{i}. {os.path.basename(filepath)}\n")


Distance Metrics
''''''''''''''''

Different distance metrics can be used for finding similar audio:

.. code:: python

	# Cosine similarity (recommended for MFCC/spectral)
	similar = cl.find('query.wav', k=10, metric='cosine')
	
	# Euclidean distance
	similar = cl.find('query.wav', k=10, metric='euclidean')
	
	# Manhattan distance
	similar = cl.find('query.wav', k=10, metric='manhattan')


**Metric recommendations:**
 * **Cosine**: Best for MFCC and spectral features (default)
 * **Euclidean**: General purpose
 * **Manhattan**: Robust to outliers


Plot Find Results
'''''''''''''''''

Visualize the found similar audio files:

.. code:: python

	from clustaudio import clustaudio
	
	cl = clustaudio(method='mfcc-spectral')
	results = cl.fit_transform('/audio/')
	
	# Find similar
	similar = cl.find('query.mp3', k=15, alpha=0.05)
	
	# Plot with annotations
	cl.plot_find()
	
	# Or plot specific clusters/waveforms
	cl.plot(labels=[0, 1, 2], show_waveform=True)


.. include:: add_bottom.add
