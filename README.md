# clustaudio

[![Python](https://img.shields.io/pypi/pyversions/clustaudio)](https://img.shields.io/pypi/pyversions/clustaudio)
[![Pypi](https://img.shields.io/pypi/v/clustaudio)](https://pypi.org/project/clustaudio/)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/clustaudio/)
[![LOC](https://sloc.xyz/github/erdogant/clustaudio/?category=code)](https://github.com/erdogant/clustaudio/)
[![Downloads](https://static.pepy.tech/personalized-badge/clustaudio?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month)](https://pepy.tech/project/clustaudio)
[![Downloads](https://static.pepy.tech/personalized-badge/clustaudio?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/clustaudio)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/clustaudio/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/clustaudio.svg)](https://github.com/erdogant/clustaudio/network)
[![Issues](https://img.shields.io/github/issues/erdogant/clustaudio.svg)](https://github.com/erdogant/clustaudio/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/423822054.svg)](https://zenodo.org/badge/latestdoi/423822054)
[![Medium](https://img.shields.io/badge/Medium-Blog-blue)](https://erdogant.github.io/clustaudio/pages/html/Documentation.html#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg?logo=github%20sponsors)](https://erdogant.github.io/clustaudio/pages/html/Documentation.html#colab-notebook)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/clustaudio/pages/html/Documentation.html#)

<div>
<a href="https://erdogant.github.io/clustaudio/"><img src="https://github.com/erdogant/clustaudio/blob/main/docs/figs/logo.png" width="75" align="left" /></a>
clustaudio is a Python library to detect natural groups or clusters of audio files. Multiple steps are pipelined where audio files are processed, acoustic features extracted, and the clusters evaluated across the feature space.
The optimal number of clusters is determined using methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
clustaudio allows you to determine the most robust clustering by efficiently searching across the parameters and by evaluating the clusters.
Besides clustering of audio, the ``clustaudio`` library can also find the most similar audio files for a new, unseen sample, denoise noisy recordings, separate foreground from background, and extract voice segments. ⭐️Star it if you like it⭐️
</div>

---

<p align="center">
  <img src="https://github.com/erdogant/clustaudio/blob/main/docs/figs/schematic_overview.png" width="1000" />
</p>

---

``clustaudio`` overcomes the following challenges: 

    * 1. Robustly groups similar audio files using acoustic features.
    * 2. Returns the unique representative audio samples per cluster.
    * 3. Finds highly similar audio files for a given input audio.
    * 4. Denoises audio with multiple methods (spectral, wiener, noisereduce).
    * 5. Separates foreground (voice/signal) from background (noise).
    * 6. Extracts voice segments using Voice Activity Detection (VAD).
    * 7. Clusters audio by metadata (artist, album, genre) when available.

``clustaudio`` is fun because:

    * It does not require a learning process.
    * It can group any set of audio files (music, speech, environmental sounds, ATC recordings).
    * It can return only the unique() audio samples per cluster.
    * It can find highly similar audio files given an input audio file.
    * It provides advanced audio processing: denoising, separation, VAD.
    * It handles noisy communications (ATC, radio) with specialized enhancement.
    * It generates playlists and organizes files by cluster.
    * It provides many plots (waveforms, spectrograms, scatter plots) to explore the feature-space.
    * It is built on core audio processing (MFCC, spectral features, chroma) and robust clustering methods.
    * It works out of the box.


# 
**⭐️ Star this repo if you like it ⭐️**
#

### Blogs

* Read the [blog](https://towardsdatascience.com/a-step-by-step-guide-for-clustering-audio-12345) to get a structured overview of how to cluster audio files.

# 

### [Documentation pages](https://erdogant.github.io/clustaudio/)

On the [documentation pages](https://erdogant.github.io/clustaudio/) you can find detailed information about the working of ``clustaudio`` with many examples. 

# 


### Installation

##### It is advisable to create a new environment (e.g. with Conda). 
```bash
conda create -n env_clustaudio python=3.12
conda activate env_clustaudio
```

##### Install clustaudio from PyPI
```bash
pip install clustaudio            # new install
pip install -U clustaudio         # update to latest version
```

##### Install with advanced features (recommended)
```bash
pip install clustaudio[advanced]  # includes denoising, VAD, metadata extraction
```

##### Directly install from GitHub source
```bash
pip install git+https://github.com/erdogant/clustaudio
```  

##### Import clustaudio package

```python
from clustaudio import Clustaudio
```

<hr>

### Quick Start

```python
from clustaudio import Clustaudio

# Initialize with MFCC features
cl = Clustaudio(method='mfcc', sr=22050)

paths = cl.import_example(data='various')
# Cluster audio files from a directory
results = cl.fit_transform(paths)

# Visualize clusters
cl.clusteval.scatter(s=250)

# Plot waveforms for each cluster
cl.plot(labels=[0, 1, 2], show_waveform=True)

# Find similar audio to a query
similar = cl.find(paths[0], k=10)

similar.keys()
# dict_keys(['feat', 'dog_barking_1.wav'])

# Organize files by cluster
cl.move_to_dir(create_playlists=True)
```

<hr>

### Core Features

The results obtained from the clustaudio library is a dictionary containing the following keys:

    * audio         : Preprocessed audio arrays
    * feat          : Acoustic features extracted (MFCC, spectral, chroma, etc.)
    * xycoord       : X and Y coordinates from the embedding (t-SNE, UMAP)
    * pathnames     : Absolute path location to the audio files
    * filenames     : File names of the audio files
    * labels        : Cluster labels
    * sample_rates  : Sample rates of loaded audio
    * durations     : Duration of each audio file


### Feature Extraction Methods

clustaudio supports multiple acoustic feature extraction methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `mfcc` | Mel-Frequency Cepstral Coefficients | General audio, speech recognition |
| `spectral` | Spectral features (centroid, rolloff, contrast) | Timbre-based clustering |
| `chroma` | Chromagram features | Music, pitch-based analysis |
| `mfcc-spectral` | Combined MFCC + spectral | Comprehensive audio analysis |
| `tempogram` | Tempo and rhythm features | Music tempo analysis |
| `fingerprint` | Audio fingerprinting | Duplicate detection |
| `metadata` | ID3, Vorbis, MP4 tags | Organize by artist/album/genre |

<hr>

### Advanced Features

#### Voice Activity Detection (VAD)

Extract voice segments from audio files:

```python
# Extract voice segments from a podcast

from clustaudio import Clustaudio

# Initialize with MFCC features
cl = Clustaudio(method='mfcc', sr=22050)
paths = cl.import_example(data='atc')

segments = cl.extract_voice_segments(
    paths,
    method='webrtcvad',
    min_segment_duration=0.5
)

# Cluster the extracted voice segments
total_segments = [item for sublist in segments['pathnames_segments'] for item in sublist]
results = cl.fit_transform(total_segments)

cl.clusteval.scatter(results['xycoord'], s=200)

```

#### Audio Denoising

Clean up noisy audio recordings:

```python
from clustaudio import Clustaudio

# Initialize with MFCC features
cl = Clustaudio(method='mfcc', sr=22050)
paths = cl.import_example(data='atc')

# Denoise audio files
denoised = cl.denoise(
    pathnames=paths,
    method='spectral',
    save_denoised=True
)

# Cluster denoised audio
results = cl.fit_transform(denoised['pathnames_denoised'])
cl.clusteval.scatter(results['xycoord'], s=200)

```

#### Foreground/Background Separation

Separate signal from noise:

```python

# Initialize with MFCC features
cl = Clustaudio(method='mfcc', sr=22050)
paths = cl.import_example(data='atc')

# Separate foreground (voice) from background (noise)
separated = cl.separate_foreground_background(paths,
    method='spectral_masking',
    denoise=True
)

# Cluster foreground audio
results = cl.fit_transform(separated['pathnames_foreground'])
cl.clusteval.scatter(results['xycoord'], s=200)

```

#### ATC Communication Enhancement

Specialized enhancement for Air Traffic Control recordings:

```python
cl = Clustaudio(method='mfcc', sr=22050)
paths = cl.import_example(data='atc')

# Enhance noisy ATC communications
enhanced = cl.enhance_atc_audio(
    paths,
    method='aggressive'
)

# Process and cluster
results = cl.fit_transform(enhanced['pathnames_enhanced'])
cl.clusteval.scatter(results['xycoord'], s=200)

```

<hr>

### Example: Music Library Organization

```python
from clustaudio import Clustaudio

# Initialize with combined features
cl = Clustaudio(method='mfcc-spectral', sr=22050)
paths = cl.import_example(data='various')

# Cluster your music library
results = cl.fit_transform(paths)

# Create playlists by cluster
playlists = cl.create_playlist(
    format='m3u8',
    savedir='/organized_music/'
)

# Visualize clusters
cl.clusteval.scatter(results['xycoord'], s=200)
cl.plot(labels=[0, 1, 2], show_waveform=True, show_spectrogram=True)

# Find similar songs
similar = cl.find(paths[0], k=20)
cl.plot_find()
```

<hr>

### Example: Podcast Speaker Identification

```python
from clustaudio import Clustaudio

cl = Clustaudio()
paths = cl.import_example(data='atc')

# Extract voice segments
segments = cl.extract_voice_segments(
    paths,
    method='webrtcvad',
    min_segment_duration=1.0
)

# Cluster speakers
total_segments = [item for sublist in segments['pathnames_segments'] for item in sublist]
results = cl.fit_transform(total_segments)

# Each cluster represents a different speaker
cl.plot(labels=[0, 1], show_waveform=True)
```

<hr>

### Example: Environmental Sound Clustering

```python
# Initialize for environmental sounds
cl = Clustaudio()
paths = cl.import_example(data='atc')

# Denoise and cluster
denoised = cl.denoise(
    paths,
    method='noisereduce',
    save_denoised=True
)

results = cl.fit_transform(denoised['pathnames_denoised'])

# Visualize sound clusters
cl.clusteval.scatter(results['xycoord'], s=200)
cl.plot_statistics()
```

<hr>

### Example: Duplicate Audio Detection

```python
# Use fingerprinting for duplicate detection
cl = Clustaudio(method='fingerprint')
paths = cl.import_example(data='atc')


# Find duplicates
results = cl.fit_transform(paths)

# Clusters with multiple files are likely duplicates/covers
for label in np.unique(results['labels']):
    cluster_files = np.array(results['filenames'])[results['labels'] == label]
    if len(cluster_files) > 1:
        print(f"Potential duplicates: {cluster_files}")
```

<hr>

### Example: ATC Communication Analysis

Complete workflow for Air Traffic Control recordings:

```python
from clustaudio import Clustaudio

# Initialize
cl = Clustaudio(method='mfcc', sr=16000)
paths = cl.import_example(data='atc')

# Step 1: Enhance noisy ATC audio
enhanced = cl.enhance_atc_audio(
    paths,
    method='aggressive'
)

# Step 2: Separate foreground (voice) from background (static)
separated = cl.separate_foreground_background(
    enhanced['pathnames_enhanced'],
    method='spectral_masking',
    denoise=True
)

# Step 3: Extract voice segments
segments = cl.extract_voice_segments(
    separated['pathnames_foreground'],
    method='energy',
    min_segment_duration=0.5
)

# Step 4: Cluster communications
total_segments = [item for sublist in segments['pathnames_segments'] for item in sublist]
results = cl.fit_transform(
    total_segments,
    cluster='agglomerative',
    evaluate='silhouette'
)

# Step 5: Visualize and analyze
cl.clusteval.scatter(results['xycoord'], s=200)
cl.plot(labels=[0, 1, 2], show_waveform=True)
cl.plot_statistics()

# Step 6: Organize by cluster
cl.move_to_dir(savedir='/organized_atc/', create_playlists=True)
```

<hr>

### Visualization Examples

##### Scatter Plot with Cluster Labels

```python
# Create scatter plot
from scatterd import scatterd

cl.clusteval.plot()
cl.clusteval.plot_silhouette(results['xycoord'])
cl.clusteval.scatter(results['xycoord'], s=250)
cl.clusteval.dendrogram()
scatterd(results['xycoord'][:, 0], results['xycoord'][:, 1], density=False, labels=results['labels'])
#cl.pca.scatter3d(labels=results['labels'])


```

##### Waveform Visualization per Cluster

```python
# Plot waveforms for specific clusters
cl.plot(labels=[0, 1, 2], show_waveform=True, max_per_cluster=4)
```

##### Spectrogram Visualization

```python
# Plot spectrograms
cl.plot(labels=[0, 1], show_spectrogram=True)
```

##### Denoising Comparison

```python
# Compare before/after denoising
cl.denoise(paths[0])
cl.plot_denoise(index=0)
```

##### Separation Results

```python
# Visualize foreground/background separation
cl.separate_foreground_background(paths[0])
cl.plot_separation(index=0, figsize=(15, 20))
```

##### Cluster Statistics

```python
# Comprehensive cluster statistics
cl.plot_statistics()
```

<hr>

### Supported Audio Formats

clustaudio supports all formats handled by librosa and soundfile:

- **WAV** - Uncompressed (fastest loading)
- **MP3** - Compressed (most common)
- **FLAC** - Lossless compression
- **OGG** - Vorbis compressed
- **M4A** - AAC compressed
- **AAC** - Advanced Audio Coding

<hr>

### Documentation

Detailed documentation with many examples:

##### [Example: Music Library Clustering](https://erdogant.github.io/clustaudio/pages/html/Examples.html#music-library)

##### [Example: Podcast Speaker Clustering](https://erdogant.github.io/clustaudio/pages/html/Examples.html#podcast-speakers)

##### [Example: ATC Communication Processing](https://erdogant.github.io/clustaudio/pages/html/Examples.html#atc-communications)

##### [Example: Environmental Sound Analysis](https://erdogant.github.io/clustaudio/pages/html/Examples.html#environmental-sounds)

##### [Example: Find Similar Audio](https://erdogant.github.io/clustaudio/pages/html/Examples.html#find-similar)

##### [Example: Break Up the Steps](https://erdogant.github.io/clustaudio/pages/html/Examples.html#breaking-up-steps)

<hr>

### Feature Comparison

| Feature | clustimage (Images) | clustaudio (Audio) |
|---------|--------------------|--------------------|
| Core Feature | PCA, HOG | MFCC, Spectral |
| Duplicate Detection | Perceptual hash | Audio fingerprinting |
| Metadata | EXIF (datetime, location) | ID3, Vorbis (artist, album, genre) |
| Advanced Processing | Face detection | VAD, Denoising, Separation |
| Unique Detection | ✅ | ✅ |
| Similarity Search | ✅ | ✅ |
| Clustering | ✅ | ✅ |
| Visualization | Image grids | Waveforms, Spectrograms |

<hr>

### Requirements

**Core:**
- Python ≥ 3.8
- librosa ≥ 0.10.0
- soundfile ≥ 0.12.0
- scikit-learn ≥ 1.0.0
- numpy, pandas, matplotlib
- clusteval, pca, distfit

**Advanced (optional):**
- noisereduce - Advanced denoising
- webrtcvad - Voice activity detection
- mutagen - Metadata extraction
- pyacoustid - Audio fingerprinting

See [requirements.txt](requirements.txt) for full list.

<hr>

### Citation

Please cite ``clustaudio`` if you use it in your research or project:

```bibtex
@software{clustaudio,
  author = {Taskesen, Erdogan},
  title = {clustaudio - Python library for clustering audio files},
  year = {2024},
  url = {https://github.com/erdogant/clustaudio},
}
```

<hr>

### Support

This project needs some love! ❤️ You can help in various ways:

* **Become a Sponsor!** Support continued development
* **Star this repo** at the GitHub page
* **Report bugs** and request features via Issues
* **Contribute code** via Pull Requests
* **Share your use cases** - we'd love to hear how you use clustaudio!

Why sponsor? Read more on the [Sponsor GitHub Page](https://erdogant.github.io/clustaudio/pages/html/Documentation.html#).

<hr>

### License

MIT License. See [LICENSE](LICENSE) for details.

<hr>

### Acknowledgments

Built on top of excellent open-source libraries:
- [librosa](https://librosa.org/) - Audio analysis
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [clusteval](https://github.com/erdogant/clusteval) - Cluster evaluation
- [pca](https://github.com/erdogant/pca) - Dimensionality reduction

<hr>

**⭐️ Star this repo if you like it ⭐️**

Made with ❤ by [Erdogan Taskesen](https://erdogant.github.io)