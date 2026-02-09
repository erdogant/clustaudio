
Abstract
''''''''

**Background**

Audio analysis and classification is a signal processing task for identifying and categorizing sounds, speech, and music.
We can separate audio analysis tasks into two broad categories: supervised and unsupervised approaches.
In the supervised case, we classify audio into a fixed number of learned categories (e.g., speech recognition, music genre classification).
In the unsupervised case, we do not depend on labeled training data but instead interpret the input audio and discover natural groups or clusters.

**Aim**

The aim of ``clustaudio`` is to detect natural groups or clusters of audio files. It works using a multi-step process of carefully pre-processing audio files, extracting features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters are determined using well-known methods such as *silhouette, dbindex, and derivatives* in combination with clustering methods such as *agglomerative, kmeans, dbscan and hdbscan*.
With ``clustaudio`` we aim to determine the most robust clustering by efficiently searching across parameters and evaluating the clusters.
Besides clustering audio files, the ``clustaudio`` model can also be used to find the most similar audio files for a new unseen sample, denoise audio, separate foreground from background, and enhance noisy communications.

**Results**

``clustaudio`` is a Python package for unsupervised clustering of audio files through careful pre-processing, feature extraction, and optimal cluster evaluation in high-dimensional feature space.
``clustaudio`` does not depend on pre-trained neural networks that require many package dependencies but is built on core signal processing methods such as Mel-Frequency Cepstral Coefficients (MFCC), Spectral features, and Chroma features.
The input for the model can be a NxM array where each row is a flattened audio feature vector, or it can be a target directory or the full path to a list of audio files.
Additional advantages of ``clustaudio`` are:

    * It can group any set of audio files by similarity.
    * It can determine the unique() audio samples (representative samples per cluster).
    * It can find similar audio files given an input audio sample.
    * It is built on core signal processing methods (MFCC, Spectral, Chroma) and therefore has minimal dependencies.
    * It provides advanced denoising and audio separation capabilities.
    * It can handle noisy communications (ATC, radio) with specialized enhancement.
    * It includes Voice Activity Detection (VAD) for speech segmentation.
    * It does not require a learning process or labeled training data.
    * It works out of the box.
    * It provides explainable results.
    * It provides many plots (waveforms, spectrograms, cluster visualizations) to improve understanding of the feature space and audio-audio relationships.
    

Schematic overview
'''''''''''''''''''

The schematic overview of the audio clustering approach is as follows:

.. _schematic_overview:

.. figure:: ../figs/schematic_overview_audio.png

   Audio processing pipeline: Audio files → Preprocessing (resample, mono, normalize) → Feature Extraction (MFCC, Spectral, Chroma) → Optional Denoising/Separation → Embedding (t-SNE/UMAP) → Clustering (evaluate optimal k) → Results (labels, visualization, organization)


.. include:: add_bottom.add
