"""
Visualization module for audio clustering and analysis.

This module provides plotting functions for:
- Waveform visualization
- Spectrogram plots
- Cluster scatter plots with audio
- Audio comparison plots
- Feature visualization
- Interactive plots (where supported)

Author: AI Assistant
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display

logger = logging.getLogger(__name__)


def plot_waveform(audio, sr=22050, title=None, figsize=(12, 4), 
                 color='blue', alpha=0.7, show_time=True, ax=None):
    """Plot audio waveform.
    
    Parameters
    ----------
    audio : np.ndarray or str
        Audio waveform or path to audio file
    sr : int
        Sample rate
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
    color : str
        Waveform color
    alpha : float
        Transparency
    show_time : bool
        Show time axis
    ax : matplotlib axis, optional
        Existing axis to plot on
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    # Load audio if path provided
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=sr, mono=True)
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot waveform
    if show_time:
        librosa.display.waveshow(audio, sr=sr, ax=ax, color=color, alpha=alpha)
        ax.set_xlabel('Time (seconds)')
    else:
        ax.plot(audio, color=color, alpha=alpha)
        ax.set_xlabel('Samples')
    
    ax.set_ylabel('Amplitude')
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_spectrogram(audio, sr=22050, n_fft=2048, hop_length=512,
                    title=None, figsize=(12, 6), cmap='viridis',
                    y_axis='mel', x_axis='time', ax=None):
    """Plot audio spectrogram.
    
    Parameters
    ----------
    audio : np.ndarray or str
        Audio waveform or path
    sr : int
        Sample rate
    n_fft : int
        FFT window size
    hop_length : int
        Hop length
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    y_axis : str
        Y-axis type: 'mel', 'log', 'linear'
    x_axis : str
        X-axis type: 'time', 'frames'
    ax : matplotlib axis
        Existing axis
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    # Load audio if path
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=sr, mono=True)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Compute spectrogram
    if y_axis == 'mel':
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    else:
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    
    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length,
        x_axis=x_axis, y_axis=y_axis,
        ax=ax, cmap=cmap
    )
    
    ax.set_ylabel('Frequency' if y_axis == 'linear' else 'Mel Frequency')
    ax.set_xlabel('Time' if x_axis == 'time' else 'Frames')
    
    if title:
        ax.set_title(title)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return fig, ax


def plot_chromagram(audio, sr=22050, hop_length=512, title=None,
                   figsize=(12, 6), cmap='coolwarm', ax=None):
    """Plot chromagram showing pitch content.
    
    Parameters
    ----------
    audio : np.ndarray or str
        Audio waveform or path
    sr : int
        Sample rate
    hop_length : int
        Hop length
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    ax : matplotlib axis
        Existing axis
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    # Load audio if path
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=sr, mono=True)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    
    # Plot
    img = librosa.display.specshow(
        chroma, sr=sr, hop_length=hop_length,
        x_axis='time', y_axis='chroma',
        ax=ax, cmap=cmap
    )
    
    ax.set_ylabel('Pitch Class')
    ax.set_xlabel('Time')
    
    if title:
        ax.set_title(title)
    
    fig.colorbar(img, ax=ax)
    
    return fig, ax


def plot_audio_comparison(audio1, audio2, sr=22050, labels=None,
                         figsize=(14, 10), show_spectrogram=True):
    """Compare two audio signals side by side.
    
    Parameters
    ----------
    audio1, audio2 : np.ndarray or str
        Audio waveforms or paths
    sr : int
        Sample rate
    labels : list of str
        Labels for each audio
    figsize : tuple
        Figure size
    show_spectrogram : bool
        Include spectrograms
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Load audio if needed
    if isinstance(audio1, str):
        audio1, sr = librosa.load(audio1, sr=sr, mono=True)
    if isinstance(audio2, str):
        audio2, sr = librosa.load(audio2, sr=sr, mono=True)
    
    # Default labels
    if labels is None:
        labels = ['Audio 1', 'Audio 2']
    
    # Create figure
    if show_spectrogram:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Waveforms
        ax1 = fig.add_subplot(gs[0, 0])
        plot_waveform(audio1, sr=sr, title=f'{labels[0]} - Waveform', ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        plot_waveform(audio2, sr=sr, title=f'{labels[1]} - Waveform', ax=ax2)
        
        # Spectrograms
        ax3 = fig.add_subplot(gs[1, 0])
        plot_spectrogram(audio1, sr=sr, title=f'{labels[0]} - Spectrogram', ax=ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        plot_spectrogram(audio2, sr=sr, title=f'{labels[1]} - Spectrogram', ax=ax4)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        plot_waveform(audio1, sr=sr, title=labels[0], ax=ax1)
        plot_waveform(audio2, sr=sr, title=labels[1], ax=ax2)
    
    plt.tight_layout()
    
    return fig


def plot_separation_result(original, foreground, background, sr=16000,
                           figsize=(14, 12), snr_db=None):
    """Visualize foreground/background separation results.
    
    Parameters
    ----------
    original : np.ndarray
        Original audio
    foreground : np.ndarray
        Separated foreground
    background : np.ndarray
        Separated background
    sr : int
        Sample rate
    figsize : tuple
        Figure size
    snr_db : float, optional
        SNR to display in title
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Original
    ax1 = fig.add_subplot(gs[0, :])
    title = 'Original Audio'
    if snr_db is not None:
        title += f' (SNR: {snr_db:.2f} dB)'
    plot_waveform(original, sr=sr, title=title, color='gray', ax=ax1)
    
    # Foreground
    ax2 = fig.add_subplot(gs[1, 0])
    plot_waveform(foreground, sr=sr, title='Foreground (Signal)', color='green', ax=ax2)
    
    ax3 = fig.add_subplot(gs[1, 1])
    plot_spectrogram(foreground, sr=sr, title='Foreground Spectrogram', ax=ax3)
    
    # Background
    ax4 = fig.add_subplot(gs[2, 0])
    plot_waveform(background, sr=sr, title='Background (Noise)', color='red', ax=ax4)
    
    ax5 = fig.add_subplot(gs[2, 1])
    plot_spectrogram(background, sr=sr, title='Background Spectrogram', ax=ax5)
    
    plt.tight_layout()
    
    return fig


def plot_cluster_audio_grid(pathnames, labels, max_per_cluster=4,
                            sr=22050, figsize=(16, 12), show_waveform=True):
    """Plot grid of audio waveforms/spectrograms by cluster.
    
    Parameters
    ----------
    pathnames : list
        List of audio file paths
    labels : np.ndarray
        Cluster labels
    max_per_cluster : int
        Maximum samples to show per cluster
    sr : int
        Sample rate
    figsize : tuple
        Figure size
    show_waveform : bool
        Show waveforms (True) or spectrograms (False)
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    fig, axes = plt.subplots(n_clusters, max_per_cluster, 
                            figsize=figsize, squeeze=False)
    
    for i, label in enumerate(unique_labels):
        idx = np.where(labels == label)[0]
        idx_sample = idx[:max_per_cluster]
        
        for j, sample_idx in enumerate(idx_sample):
            ax = axes[i, j]
            
            try:
                if show_waveform:
                    plot_waveform(pathnames[sample_idx], sr=sr, 
                                title=f'Cluster {label}' if j == 0 else '',
                                ax=ax)
                else:
                    plot_spectrogram(pathnames[sample_idx], sr=sr,
                                   title=f'Cluster {label}' if j == 0 else '',
                                   ax=ax)
            except:
                ax.text(0.5, 0.5, 'Load Error', ha='center', va='center')
                ax.axis('off')
        
        # Hide unused subplots
        for j in range(len(idx_sample), max_per_cluster):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    return fig


def plot_feature_importance(features, feature_names=None, top_n=20,
                           figsize=(10, 8), title='Feature Importance'):
    """Plot feature importance or variance.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    feature_names : list, optional
        Names of features
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    # Compute variance as importance measure
    importance = np.var(features, axis=0)
    
    # Sort and get top N
    sorted_idx = np.argsort(importance)[::-1][:top_n]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(features.shape[1])]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(sorted_idx))
    ax.barh(y_pos, importance[sorted_idx])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Variance (Importance)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    return fig, ax


def plot_denoise_comparison(original, denoised, sr=22050, 
                            method_name='Denoised', figsize=(14, 10)):
    """Compare original and denoised audio.
    
    Parameters
    ----------
    original : np.ndarray or str
        Original audio
    denoised : np.ndarray
        Denoised audio
    sr : int
        Sample rate
    method_name : str
        Name of denoising method
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Load original if path
    if isinstance(original, str):
        original, sr = librosa.load(original, sr=sr, mono=True)
    
    fig = plot_audio_comparison(
        original, denoised, sr=sr,
        labels=['Original (Noisy)', method_name],
        figsize=figsize,
        show_spectrogram=True
    )
    
    return fig


def plot_vad_segments(audio, timestamps, sr=16000, figsize=(14, 6)):
    """Visualize voice activity detection segments.
    
    Parameters
    ----------
    audio : np.ndarray or str
        Audio waveform or path
    timestamps : list of tuples
        List of (start, end) timestamps for voice segments
    sr : int
        Sample rate
    figsize : tuple
        Figure size
        
    Returns
    -------
    tuple
        (fig, ax)
    """
    # Load audio if path
    if isinstance(audio, str):
        audio, sr = librosa.load(audio, sr=sr, mono=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot waveform
    times = np.arange(len(audio)) / sr
    ax.plot(times, audio, color='gray', alpha=0.5, label='Audio')
    
    # Highlight voice segments
    for i, (start, end) in enumerate(timestamps):
        ax.axvspan(start, end, alpha=0.3, color='green',
                  label='Voice Segment' if i == 0 else '')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Voice Activity Detection ({len(timestamps)} segments)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax


def plot_cluster_statistics(results, figsize=(14, 10)):
    """Plot comprehensive cluster statistics.
    
    Parameters
    ----------
    results : dict
        Clustering results dictionary
    figsize : tuple
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = results['labels']
    unique_labels = np.unique(labels)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Cluster size distribution
    ax1 = fig.add_subplot(gs[0, 0])
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    ax1.bar(unique_labels, cluster_sizes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Cluster Label')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Cluster Size Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Duration distribution (if available)
    if results.get('durations') is not None:
        ax2 = fig.add_subplot(gs[0, 1])
        durations = results['durations']
        for label in unique_labels:
            idx = labels == label
            ax2.boxplot([durations[idx]], positions=[label], widths=0.6)
        ax2.set_xlabel('Cluster Label')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Audio Duration by Cluster')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Cluster scatter plot
    if results.get('xycoord') is not None:
        ax3 = fig.add_subplot(gs[1, :])
        xycoord = results['xycoord']
        scatter = ax3.scatter(xycoord[:, 0], xycoord[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=50)
        ax3.set_xlabel('Component 1')
        ax3.set_ylabel('Component 2')
        ax3.set_title('Cluster Scatter Plot (t-SNE/UMAP)')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        handles = [mpatches.Patch(color=plt.cm.tab10(i / 10), 
                                 label=f'Cluster {label}')
                  for i, label in enumerate(unique_labels)]
        ax3.legend(handles=handles, loc='best', ncol=2)
    
    plt.tight_layout()
    
    return fig


def save_figure(fig, filepath, dpi=150, bbox_inches='tight'):
    """Save figure to file.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Output path
    dpi : int
        Resolution
    bbox_inches : str
        Bounding box mode
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f'Saved figure to {filepath}')


def close_all():
    """Close all matplotlib figures."""
    plt.close('all')
