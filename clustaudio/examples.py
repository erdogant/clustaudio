from clustaudio import Clustaudio

sr = 22050
cl = Clustaudio(sr=sr)
filepaths = cl.import_example(data='atc')

result = cl.vad.extract_voice_segments(
    filepaths,
    sr=sr,
    method='energy',
    energy_threshold=0.01
)


# %%


from clustaudio import Clustaudio

sr = 22050
cl = Clustaudio(sr=sr)
filepaths = cl.import_example(data='atc')
audio = cl.import_data(filepaths)['audio']

segments = cl.vad.detect_voice_segments(
    audio,
    sr=sr,
    method='energy',
    energy_threshold=0.02,
    min_segment_duration=0.1
)


# %%


# Use combined features
from clustaudio import Clustaudio
cl = Clustaudio()
# Check output variables
X = cl.import_example(data='atc', n=10)
# Cluster with custom parameters
cl.fit_transform(X)
cl.clusteval.scatter(cl.results['xycoord'], s=200)


# %%


from clustaudio import Clustaudio
import numpy as np

sr = 44100
cl = Clustaudio(method="mfcc", verbose="warning")
audio = cl.import_example(data='wave', n=3, freq=440, sr=sr, duration=1.0)

# Extract features
features = cl.extract_mfcc(audio, sr=sr)

assert isinstance(features, np.ndarray)

expected = cl.params_mfcc["n_mfcc"] * 2
assert len(features) == expected

assert not np.isnan(features).any()
assert not np.isinf(features).any()


# %%
from clustaudio import Clustaudio
# Initialize with MFCC features
cl = Clustaudio(method='mfcc', sr=22050)
# Load example audio dataset
X = cl.import_example(data='various')
# Cluster audio files
results = cl.fit_transform(X)
# Visualize results
# cl.clusteval.scatter(s=350)
# cl.plot(labels=[0, 1, 2], show_waveform=True, show_spectrogram=True)
# Find similar audio
results_find = cl.find('./tests/audio/various/dog_barking_1.wav', k=3)
cl.plot_find()
# Organize files by cluster
# cl.move_to_dir(create_playlists=True)

# %%





# %%
from clustaudio import Clustaudio
dirpath = r'D://REPOS//clustaudio//data//Air Traffic Control Communication'
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\train'

# Use combined features
cl = Clustaudio(
    method='mfcc',
    sr=44100,
    mono=True,
    normalize=True,
    trim_silence=True,
    params_mfcc={'mfcc': 20, 'n_mels': 128}
)

# Cluster with custom parameters
results = cl.fit_transform(
    dirpath,
    cluster='dbscan',
    evaluate='silhouette',
    metric='euclidean',
    min_clust=3,
    max_clust=15
)

# Get representative samples
unique_samples = cl.unique(metric='cosine')
print(unique_samples['pathnames'])  


# %%
# Audio Files
#     ↓
# Import & Preprocess (normalize, resample, mono)
#     ↓
# Optional: Denoise / Enhance / Separate
#     ↓
# Optional: VAD (extract voice segments)
#     ↓
# Feature Extraction (MFCC, Spectral, Chroma)
#     ↓
# Embedding (t-SNE, UMAP)
#     ↓
# Clustering (Agglomerative, KMeans, DBSCAN)
#     ↓
# Results (labels, centroids, visualization)
#     ↓
# Organization (playlists, file management)


# Phase 1 Implementation Complete! 🎉
# I've successfully completed Phase 1: Core Functionality of the clustaudio image-to-audio conversion. Here's what has been delivered:
# ✅ What Was Accomplished

# Complete Feature Replacement

# ❌ Removed: imread(), OpenCV, PIL, imagehash, HOG features
# ✅ Added: audioread(), librosa, soundfile, MFCC/spectral/chroma features

# Core Methods Implemented (13 essential methods)

# audioread() - Load audio files with preprocessing
# preprocessing() - Audio-specific preprocessing pipeline
# extract_mfcc() - MFCC feature extraction
# extract_spectral() - Spectral features (6 types)
# extract_chroma() - Chromagram features
# extract_feat() - Main feature dispatcher
# import_data() - Multi-format data import
# fit_transform() - Complete clustering workflow
# cluster() - Cluster evaluation and labeling
# embedding() - Dimensionality reduction for visualization
# unique() - Find representative samples
# clean_init() - Reset state


# Code Quality

# ✅ 1,631 lines of well-documented code
# ✅ 24 methods in main class
# ✅ 6 helper functions
# ✅ 162 lines of class documentation
# ✅ Valid Python syntax (AST verified)
# ✅ All audio imports correct
# ✅ All image imports removed

# 🎯 Key Features Working

# ✅ Load audio files (WAV, MP3, FLAC, OGG, M4A, AAC)
# ✅ Extract MFCC features (13-40 coefficients)
# ✅ Extract spectral features (centroid, rolloff, contrast, bandwidth, ZCR, RMS)
# ✅ Extract chroma features (12 pitch classes)
# ✅ Full clustering pipeline with clusteval integration
# ✅ t-SNE embedding for visualization
# ✅ Representative sample finding per cluster
# ✅ Batch processing with progress bars
# ✅ Flexible logging and verbosity control


# %%
from scatterd import scatterd
from clustaudio import Clustaudio
import os

# methods = ['mfcc', 'spectral', 'chroma', 'mfcc-spectral', 'tempogram', 'mel-spectrogram', 'embeddings', 'fingerprint', 'metadata']
method = 'mfcc'

# Initialize with MFCC features
cl = Clustaudio(
    method=method, sr=44100,
    mono=True,
    normalize=True,
    trim_silence=False,
    embedding='tsne',
    )

# Load example audio dataset
# dirpath = r'D://REPOS//clustaudio//data//Air Traffic Control Communication'
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
# pathnames = cl.import_data(dirpath)

# Proces audio files and cluster
results = cl.fit_transform(dirpath, min_clust=10, max_clust=25)
results['y_true']= [os.path.basename(os.path.dirname(p)) for p in results['pathnames']]

try:
    cl.clusteval.plot(title=method)
    cl.clusteval.plot_silhouette(results['xycoord'])
    cl.clusteval.scatter(results['xycoord'])
    # cl.clusteval.dendrogram()
    scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], density=False, labels=results['labels'], title=method)
    cl.pca.scatter3d(labels=results['labels'])
except:
    pass


# %% METHOD TESTING
from clustaudio import Clustaudio

methods = ['mfcc', 'spectral', 'chroma', 'mfcc-spectral', 'tempogram', 'mel-spectrogram', 'embeddings', 'fingerprint', 'metadata']

for method in methods:
    # Initialize with MFCC features
    cl = Clustaudio(
        method=method, sr=44100,
        mono=True,
        normalize=True,
        trim_silence=True,
        )

    # Load example audio dataset
    dirpath = r'D://REPOS//clustaudio//data//Air Traffic Control Communication'
    dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
    # pathnames = cl.import_data(dirpath)
    
    # Proces audio files
    results = cl.fit_transform(dirpath)
    # Cluster
    labx = cl.cluster(min_clust=10, max_clust=25)

    try:
        cl.clusteval.plot(title=method)
        cl.clusteval.plot_silhouette(results['xycoord'])
        cl.clusteval.scatter(results['xycoord'])
        # cl.clusteval.dendrogram()
        
        # from scatterd import scatterd
        # import matplotlib.pyplot as plt
        # scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1])
    except:
        pass

# Get representative samples
# unique_samples = cl.unique(metric='cosine')
# print(unique_samples['pathnames'])  

# %%
import numpy as np
label=12

paths=np.array(results['pathnames'])
print(paths[results['labels']==label])
print(sum(results['labels']==label))

# %%
# Visualize results
cl.scatter(dotsize=50)
cl.plot(labels=[0, 1, 2], show_waveform=True, show_spectrogram=True)

# Find similar audio
results_find = cl.find('path/to/query.mp3', k=10)
cl.plot_find()

# Organize files by cluster
cl.move_to_dir(create_playlists=True)




# %% FIND DUPLICATES
# This phase added advanced features that significantly extend the library's capabilities beyond basic clustering. 
# The implementation includes audio fingerprinting, metadata-based clustering, voice activity detection, playlist generation, 
# and comprehensive similarity search functionality.

# Capabilities:

# 1. Find exact duplicates (different encodings, bitrates)
# 2. Find near-duplicates (covers, remixes)
# 3. Fast binary hash comparison
# 4. Similarity matrix computation for clustering
# 5. Hamming, cosine, and Euclidean distance metrics
  
import os
import fingerprint
from clustaudio import Clustaudio, listdir
 
filepath1 = r'../data\2 Brothers On The 4th Floor - Come Take My Hand.mp3'
filepath2 = r'../data\2 brothers on the 4th floor - Dreams.mp3'


# Compute fingerprint
fp1 = fingerprint.compute_fingerprint(filepath=filepath1, method='chroma_hash')
fp2 = fingerprint.compute_fingerprint(filepath=filepath2, method='chroma_hash')

# Compare
similarity = fingerprint.compare_fingerprints(fp1, fp2, method='hamming')

# Group duplicates in collection
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\train'
pathnames = listdir(dirpath)
fps = fingerprint.batch_compute_fingerprints(pathnames)
duplicates = fingerprint.find_duplicates(fps, threshold=0.1)

# Fingerprint matrix-score for duplicates
dist_matrix = fingerprint.compute_fingerprint_matrix(fps)

# from clusteval import clusteval
# cl = clusteval()
# cl.fit(dist_matrix)
# cl.plot()
# cl.scatter()
# d = cl.dendrogram()

# %% Metadata-Based Clustering
# Comprehensive metadata extraction:
    # ID3 tags (MP3)
    # Vorbis comments (OGG, FLAC)
    # MP4 metadata (M4A, AAC)
    # Basic file properties

# Extracted Fields:
    # Artist, Album, Title, Genre
    # Year, Track Number
    # Duration, Bitrate, Sample Rate
    # File size, Channels

# Libraries Supported (with graceful fallbacks):
    # Primary: mutagen (most comprehensive)
    # Fallback 1: TinyTag (lightweight)
    # Fallback 2: eyed3 (MP3 specific)

from clustaudio import Clustaudio, listdir
import metadata

filepath1 = r'../data\2 Brothers On The 4th Floor - Come Take My Hand.mp3'
dirpath = r'\\NAS_SYNOLOGY\music\90s dance'

# Extract metadata from single file
meta = metadata.extract_metadata(filepath1)
print(meta['artist'], meta['album'], meta['genre'])

# Batch extraction
pathnames = listdir(dirpath)
df = metadata.extract_metadata_batch(pathnames)

# Cluster by metadata
labels = metadata.cluster_by_metadata(df, cluster_by=['artist', 'album'])
df.loc[labels==8]

# Convert to numerical features for ML
features = metadata.create_metadata_features(df)

# %% CLUSTER ON METADATA

from clustaudio import Clustaudio
dirpath = r'\\NAS_SYNOLOGY\music\90s dance'

# Integration with Main Class:
cl = Clustaudio(method='metadata')

# Automatically groups by artist/album/genre
cl.fit_transform(dirpath)

label=1
np.array(cl.results['pathnames'])[cl.results['labels']==label]

# %% VOICE ACTIVITY DETECTION

# Three VAD methods:
# 1. Energy-based (simple, fast, no dependencies)
# 2. Librosa-based (onset detection + silence removal)
# 3. WebRTC VAD (production-quality, requires webrtcvad)

# detect_voice_segments(audio, method='energy')
# extract_voice_segments(filepath, output_dir)
# batch_extract_voice_segments(pathnames)
  
import vad

# filepath = r'../data\2 Brothers On The 4th Floor - Come Take My Hand.mp3'
filepath = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-07R4Pv.wav'
dirpath = r'D:/REPOS/clustaudio/data/Air Traffic Control Communication'
 
# Detect segments
# segments = vad.detect_voice_segments(X, method='librosa')
# Returns: [(start1, end1), (start2, end2), ...]

# Extract and save segments
segments = vad.extract_voice_segments(filepath, output_dir='segments/', method='energy')
segments = vad.extract_voice_segments(filepath, output_dir='segments/', method='webrtcvad')
segments = vad.extract_voice_segments(filepath, output_dir='segments/', method='librosa')


# %% VOICE DETECTION - THEN - CLUSTERING

from clustaudio import Clustaudio, listdir
import vad
import numpy as np

dirpath = r'D:/REPOS/clustaudio/data/Air Traffic Control Communication'

# Initialize
cl = Clustaudio(method='mfcc')

# Extract voice segments in batch process
filepaths = listdir(dirpath)
segments = vad.batch_extract_voice_segments(filepaths, output_dir='segments/', method='energy')
pathnames = [p for seg in segments if seg.get("pathnames_segments") for p in seg["pathnames_segments"]]

# Proces segments
out = cl.fit_transform(pathnames[0:5000])
# Cluster
labx = cl.cluster(min_clust=10, max_clust=25)

cl.clusteval.plot()
cl.clusteval.plot_silhouette(out['xycoord'])
cl.clusteval.scatter(out['xycoord'])

label = 3
np.array(cl.results['pathnames'])[cl.results['labels']==label]


filename = 'ACCU-0agmXf_segment_000.wav'
filepath = f'D:\REPOS\clustaudio\clustaudio\segments/{filename}'
similar = cl.find(filepath, k=10)
# Find statistically significant matches
similar = cl.find(filepath, alpha=0.05)
# Both criteria
similar = cl.find(filepath, k=20, alpha=0.1)

similar[filename]['y_filenames']
similar[filename]['x_pathnames'].loc[0]
similar[filename]['x_pathnames'].loc[1]
similar[filename]['x_pathnames'].loc[2]

# %% Advanced Similarity Search (find method)
from clustaudio import Clustaudio

# 1. k-nearest neighbor search
# 2. Probability-based matching (statistical significance)
# 3. Combined approach (union of k-NN and probability)
# 4. Distance metrics: euclidean, cosine, manhattan, etc.
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'

# initialize
cl = Clustaudio(method='mfcc')
# fit transform
results = cl.fit_transform(dirpath)

# Find 10 most similar songs
filepath = r'D:\REPOS\clustaudio\data\audio_dataset\train\Fart\Fart_100.wav'
similar = cl.find(filepath, k=10)
# Find statistically significant matches
similar = cl.find(filepath, alpha=0.05)
# Both criteria
similar = cl.find(filepath, k=20, alpha=0.1)

# %%
# =============================================================================
# Duplicate Audio Detection
# =============================================================================
from clustaudio import Clustaudio, listdir
import fingerprint

dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
filepaths = listdir(dirpath)

# Compute fingerprints
fps = fingerprint.batch_compute_fingerprints(filepaths)

# Find duplicates
duplicates = fingerprint.find_duplicates(fps, threshold=0.05)

# Review and remove
for group in duplicates:
    print(f"Duplicates: {[filepaths[i] for i in group]}")

# %%
# =============================================================================
# Music Recommendation System
# =============================================================================
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'

cl = Clustaudio(method='mfcc-spectral')
results = cl.fit_transform(dirpath)

# Find similar songs
similar = cl.find('user_favorite_song.mp3', k=20, alpha=0.1)

# Create personalized playlist
recommendation_paths = similar['user_favorite_song.mp3']['y_pathnames']


# %% Full workflow test
from clustaudio import Clustaudio
filepath = r'../data/mp3/2 Brothers On The 4th Floor - Come Take My Hand.mp3'

cl = Clustaudio(method='mfcc')

# 1. Extract voice segments
segments = cl.extract_voice_segments(filepath)

# 2. Cluster segments
results = cl.fit_transform(segments['pathnames_segments'])

# 3. Find similar
similar = cl.find(segments['pathnames_segments'][0], k=5)

# 4. Create playlists
playlists = cl.create_playlist(format='m3u8')

# 5. Save model
cl.save('test_model.pkl')

# 6. Load and verify
cl2 = Clustaudio()
cl2.load('test_model.pkl')
assert len(cl2.results['labels']) == len(results['labels'])

# %% Playlist Generation
# Create playlist per cluster
# Absolute or relative paths
# Include duration metadata
# Track information
# Custom naming

dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
from clustaudio import Clustaudio

cl = Clustaudio()
results = cl.fit_transform(dirpath)

# Create M3U8 playlists for all clusters
playlists = cl.create_playlist(format='m3u8')

# Create for specific cluster
playlist = cl.create_playlist(
    labels=3,
    playlist_name='rock_collection',
    format='m3u8',
    relative_paths=True
)

# Create JSON playlist with metadata
playlist = cl.create_playlist(
    labels=[0, 1, 2],
    format='json',
    savedir='/playlists/'
)

# %%
# =============================================================================
# # Enhance, separate, extract, cluster
# =============================================================================
from clustaudio import Clustaudio
import vad
import os
import numpy as np
from scatterd import scatterd
from sklearn.preprocessing import LabelEncoder
from classeval import classeval


cl = Clustaudio(method='mfcc', sr=16000)

filepath = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-07R4Pv.wav'
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'

enhanced = cl.enhance_atc_audio(dirpath, method='light')
separated = cl.separate_foreground_background(enhanced['pathnames_enhanced'])
segments = vad.batch_extract_voice_segments(separated['pathnames_foreground'])
results = cl.fit_transform([p for s in segments for p in s['pathnames_segments']])
# Cluster
y_pred = cl.cluster(min_clust=10, max_clust=25)


# Get original directory labels
results['pathnames']
labels = [os.path.basename(os.path.dirname(p)) for p in results['pathnames']]
labels_true = [lbl.replace("_enhanced_foreground_segments", "") for lbl in labels]
print(labels_true)
# Create labels
encoder = LabelEncoder()
y_true = encoder.fit_transform(labels_true)


# plot
cl.clusteval.plot()
cl.clusteval.plot_silhouette(results['xycoord'])
cl.clusteval.scatter(results['xycoord'], s=100)
scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], c=y_pred, marker = labels_true, density=False, s=250)
scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], labels=y_true, density=False, s=250)


# Confmatrix evaluation
confmat = classeval.confmatrix.eval(y_true, y_pred, normalize=False)
ax = classeval.confmatrix.plot(confmat)

label = 3
np.array(cl.results['pathnames'])[y_pred==label]

# %%
# =============================================================================
# # Extract and cluster voices of speakers
# =============================================================================
from clustaudio import Clustaudio
filepath = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-07R4Pv.wav'

cl = Clustaudio(method='mfcc')
segments = cl.extract_voice_segments(filepath, method='webrtcvad')
results = cl.fit_transform(segments['pathnames_segments'])

# %%
# =============================================================================
# # Find duplicates
# =============================================================================
import fingerprint
from clustaudio import listdir

dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
filepaths = listdir(dirpath)

fps = fingerprint.batch_compute_fingerprints(filepaths)
duplicates = fingerprint.find_duplicates(fps, threshold=0.5)

# Get the labels
labels = np.array([os.path.basename(os.path.dirname(p)) for p in filepaths])
print(labels)

for d in duplicates:
    print(labels[d])

# %%
# =============================================================================
# # Denoise and cluster
# =============================================================================
from clustaudio import Clustaudio
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'
filepaths = listdir(dirpath)

filepaths = cl.import_example(data='atc')

cl = Clustaudio(method='spectral')
denoised = cl.denoise(filepaths, method='spectral', stationary=False)
results = cl.fit_transform(denoised['pathnames_denoised'])

# Cluster
y_pred = cl.cluster(min_clust=10, max_clust=25)

# plot
cl.clusteval.plot()
cl.clusteval.plot_silhouette(results['xycoord'])
cl.clusteval.scatter(results['xycoord'], s=100)
scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], c=y_pred, marker = labels_true, density=False, s=250)

# %%
# =============================================================================
# ATC Processing
# =============================================================================

from clustaudio import Clustaudio, listdir
import vad

# Initialize
cl = Clustaudio(method='mfcc', sr=16000)

# Get files
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
filepaths = listdir(dirpath)

# Extract voice segments
segments = vad.batch_extract_voice_segments(filepaths, output_dir='segments/')
pathnames = [p for seg in segments if seg.get("pathnames_segments") for p in seg["pathnames_segments"]]

# Cluster
results = cl.fit_transform(pathnames)
labels = cl.cluster(min_clust=10, max_clust=25)

print(f"Found {len(np.unique(labels))} clusters from {len(pathnames)} segments")

# plot
cl.clusteval.plot()
cl.clusteval.plot_silhouette(results['xycoord'])
cl.clusteval.scatter(results['xycoord'], s=200)
scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], c=labels, marker = results['labels'], density=False, s=250)


# %%
# =============================================================================
# ATC Processing - With Enhancement and Separation
# =============================================================================

from clustaudio import Clustaudio, listdir
import vad

cl = Clustaudio(method='mfcc')

# Get files
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
filepaths = listdir(dirpath)

# Full pipeline
enhanced = cl.enhance_atc_audio(filepaths, method='aggressive')
separated = cl.separate_foreground_background(enhanced['pathnames_enhanced'])
segments = vad.batch_extract_voice_segments(separated['pathnames_foreground'])

# Cluster voice
pathnames_voice = [p for s in segments for p in s['pathnames_segments']]
# Fit transform
results_voice = cl.fit_transform(pathnames_voice)
# Cluster
labels = cl.cluster(min_clust=10, max_clust=25)

# Plot
cl.clusteval.plot()
cl.clusteval.plot_silhouette(results_voice['xycoord'])
cl.clusteval.scatter(results_voice['xycoord'], s=200)


# =============================================================================
# RESULTS ON NOISE
# =============================================================================
from clustaudio import Clustaudio, listdir
import vad

cl = Clustaudio(method='mfcc')

# Get files
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
filepaths = listdir(dirpath)

# Full pipeline
enhanced = cl.enhance_atc_audio(filepaths, method='aggressive')
separated = cl.separate_foreground_background(enhanced['pathnames_enhanced'])

results_noise = cl.fit_transform(separated['pathnames_background'])
# Cluster
labels = cl.cluster(min_clust=10, max_clust=25)

# Plot
cl.clusteval.plot()
cl.clusteval.plot_silhouette(results_noise['xycoord'])
cl.clusteval.scatter(results_noise['xycoord'], s=200)

# %%
# =============================================================================
# PLOTS
# =============================================================================
# 1. Waveforms and spectrograms
# 2. Before/after denoising
# 3. Foreground/background separation
# 4. Cluster statistics

from clustaudio import Clustaudio
import visualization as vz

filepath1 = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-07R4Pv.wav'
filepath2 = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-0iyG6P.wav'

vz.plot_waveform(filepath1)
vz.plot_spectrogram(filepath1)
vz.plot_chromagram(filepath1)
vz.plot_audio_comparison(filepath1, filepath2)

# =============================================================================
# PLOT FOREGROUND, BACKGROUND
# =============================================================================
cl = Clustaudio(method='mfcc')
separated = cl.separate_foreground_background(filepath1)
vz.plot_separation_result(separated['results'][0]['pathname'],
                          separated['results'][0]['pathname_foreground'],
                          separated['results'][0]['pathname_background'],
                          )

# =============================================================================
# plot_cluster_audio_grid
# =============================================================================
dirpath = r'D:/REPOS/clustaudio/data/ATC_small'
dirpath = r'D:\REPOS\clustaudio\data\audio_dataset\val'

filepaths = listdir(dirpath)
results = cl.fit_transform(filepaths)
labels = cl.cluster(min_clust=10, max_clust=25)
vz.plot_cluster_audio_grid(filepaths, labels)

# =============================================================================
# Feature importance
# =============================================================================
vz.plot_feature_importance(results['feat'])

# =============================================================================
# plot_denoise_comparison
# =============================================================================
from clustaudio import Clustaudio
cl = Clustaudio(method='mfcc')
cl.import_data(filepath1)
denoised = cl.denoise(filepath1, method='spectral', stationary=False)
vz.plot_denoise_comparison(cl.results['audio'][0], denoised['audio_denoised'][0])

# =============================================================================
# plot_vad_segments
# =============================================================================
from clustaudio import Clustaudio
filepath1 = r'D:\REPOS\clustaudio\data\Air Traffic Control Communication\ACCU-07R4Pv.wav'

cl = Clustaudio(method='mfcc')
segments = cl.extract_voice_segments(filepath1, method='energy')
# cl.fit_transform(segments['pathnames_segments'])
cl.import_data(filepath1)
vz.plot_vad_segments(cl.results['audio'][0], segments['timestamps'][0])

# %%

# Save model
cl.save('my_music_model.pkl', overwrite=True)

# Load later
cl_new = Clustaudio()
cl_new.load('my_music_model.pkl')

# Continue where you left off
similar = cl_new.find('new_song.mp3', k=10)
