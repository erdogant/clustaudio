"""
Complete ATC Audio Clustering Example
======================================

This script demonstrates the full workflow for processing noisy
Air Traffic Control communications with clustaudio.

Usage:
    python atc_example.py --input ./ATC_recordings/ --output ./results/

Requirements:
    pip install librosa soundfile numpy pandas scikit-learn scipy matplotlib tqdm
    pip install pca distfit clustaudio scatterd colourmap datazets pypickle ismember
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import clustaudio
from clustaudio import clustaudio, listdir
import vad
import separation
import visualization as viz


def process_atc_communications(input_dir, output_dir, enhancement_level='aggressive',
                               save_plots=True, verbose=True):
    """
    Process ATC communications with full pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing ATC audio files
    output_dir : str
        Directory for output files
    enhancement_level : str
        'light', 'moderate', or 'aggressive'
    save_plots : bool
        Save visualization plots
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results dictionary with all processing outputs
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'separated'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'segments_voice'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'organized'), exist_ok=True)

    if verbose:
        print("=" * 70)
        print("ATC Audio Clustering Pipeline")
        print("=" * 70)
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Enhancement: {enhancement_level}")
        print("=" * 70)

    # Initialize clustaudio
    if verbose:
        print("\n[1/8] Initializing clustaudio...")

    cl = clustaudio(
        method='mfcc',
        sr=16000,  # Typical for radio communications
        mono=True,
        normalize=True,
        params_mfcc={
            'n_mfcc': 20,
            'n_fft': 2048,
            'hop_length': 512
        },
        verbose='info'
    )

    # Get audio files
    if verbose:
        print("\n[2/8] Scanning for audio files...")

    filepaths = listdir(input_dir, ext=['wav', 'mp3', 'flac', 'ogg', 'm4a'])

    if len(filepaths) == 0:
        print(f"ERROR: No audio files found in {input_dir}")
        return None

    if verbose:
        print(f"Found {len(filepaths)} audio files")

    # Step 1: Enhance ATC audio
    if verbose:
        print(f"\n[3/8] Enhancing audio (method: {enhancement_level})...")

    enhanced = cl.enhance_atc_audio(
        filepaths,
        method=enhancement_level,
        output_dir=os.path.join(output_dir, 'enhanced')
    )

    n_enhanced = len([p for p in enhanced['pathnames_enhanced'] if p is not None])
    if verbose:
        print(f"Enhanced {n_enhanced}/{len(filepaths)} files successfully")

    # Step 2: Separate foreground (voice) from background (noise)
    if verbose:
        print("\n[4/8] Separating voice from noise...")

    separated = cl.separate_foreground_background(
        enhanced['pathnames_enhanced'],
        method='spectral_masking',
        denoise=False,  # Already enhanced
        output_dir=os.path.join(output_dir, 'separated'),
        save_both=True
    )

    avg_snr = np.mean([s for s in separated['snr_db'] if s is not None])
    if verbose:
        print(f"Separated {len(separated['pathnames_foreground'])} files")
        print(f"Average SNR: {avg_snr:.2f} dB")

    # Step 3: Extract voice segments
    if verbose:
        print("\n[5/8] Extracting voice segments...")

    segments_voice = vad.batch_extract_voice_segments(
        separated['pathnames_foreground'],
        output_dir=os.path.join(output_dir, 'segments_voice'),
        sr=16000,
        method='energy',
        energy_threshold=0.015,
        min_silence_duration=0.3,
        min_segment_duration=0.5
    )

    # Collect voice segment paths
    pathnames_voice = [
        p
        for seg in segments_voice
        if seg.get("pathnames_segments")
        for p in seg["pathnames_segments"]
    ]

    if verbose:
        print(f"Extracted {len(pathnames_voice)} voice segments from {len(segments_voice)} files")
        avg_segments = len(pathnames_voice) / max(len(segments_voice), 1)
        print(f"Average segments per file: {avg_segments:.1f}")

    # Step 4: Cluster voice segments
    if verbose:
        print("\n[6/8] Clustering voice segments...")

    if len(pathnames_voice) < 10:
        print("WARNING: Too few voice segments for reliable clustering")
        print(f"Found only {len(pathnames_voice)} segments (minimum 10 recommended)")
        return {
            'enhanced': enhanced,
            'separated': separated,
            'segments_voice': segments_voice,
            'error': 'Too few segments'
        }

    results_voice = cl.fit_transform(pathnames_voice)

    labels_voice = cl.cluster(
        min_clust=max(3, min(10, len(pathnames_voice) // 20)),
        max_clust=min(25, len(pathnames_voice) // 10),
        cluster='agglomerative',
        evaluate='silhouette',
        metric='cosine',
        cluster_space='high'
    )

    n_voice_clusters = len(np.unique(labels_voice))
    if verbose:
        print(f"Found {n_voice_clusters} voice clusters")
        for i in range(n_voice_clusters):
            count = np.sum(labels_voice == i)
            print(f"  Cluster {i}: {count} segments")

    # Step 5: Cluster background noise (optional but useful)
    if verbose:
        print("\n[7/8] Clustering background noise...")

    results_noise = cl.fit_transform(separated['pathnames_background'])

    labels_noise = cl.cluster(
        min_clust=2,
        max_clust=8,
        cluster='dbscan',
        metric='euclidean'
    )

    n_noise_clusters = len(np.unique(labels_noise))
    if verbose:
        print(f"Found {n_noise_clusters} noise types")

    # Step 6: Generate visualizations
    if save_plots and verbose:
        print("\n[8/8] Generating visualizations...")

    plots_dir = os.path.join(output_dir, 'plots')

    if save_plots:
        # Plot cluster statistics
        try:
            fig = cl.plot_statistics()
            if fig:
                viz.save_figure(fig, os.path.join(plots_dir, 'cluster_statistics.png'))
                viz.close_all()
        except Exception as e:
            print(f"Warning: Could not create statistics plot: {e}")

        # Plot separation results (first 3 files)
        for i in range(min(3, len(separated['results']))):
            try:
                fig = cl.plot_separation(index=i)
                if fig:
                    viz.save_figure(fig, os.path.join(plots_dir, f'separation_{i}.png'))
                    viz.close_all()
            except Exception as e:
                print(f"Warning: Could not create separation plot {i}: {e}")

        if verbose:
            print(f"Saved plots to {plots_dir}")

    # Create reports
    if verbose:
        print("\nGenerating reports...")

    # Voice cluster report
    voice_report = pd.DataFrame({
        'segment_path': pathnames_voice,
        'voice_cluster': labels_voice
    })
    voice_report.to_csv(os.path.join(output_dir, 'voice_clusters.csv'), index=False)

    # Combined report
    combined_report = pd.DataFrame({
        'original_file': [os.path.basename(r['pathname']) for r in separated['results']],
        'enhanced_file': [os.path.basename(p) for p in enhanced['pathnames_enhanced']],
        'foreground_file': [os.path.basename(p) if p else None for p in separated['pathnames_foreground']],
        'background_file': [os.path.basename(p) if p else None for p in separated['pathnames_background']],
        'snr_db': separated['snr_db'],
        'noise_cluster': labels_noise
    })
    combined_report.to_csv(os.path.join(output_dir, 'processing_report.csv'), index=False)

    # Summary statistics
    summary = {
        'total_files': len(filepaths),
        'enhanced_files': n_enhanced,
        'separated_files': len(separated['pathnames_foreground']),
        'voice_segments': len(pathnames_voice),
        'voice_clusters': n_voice_clusters,
        'noise_clusters': n_noise_clusters,
        'avg_snr_db': float(avg_snr),
        'avg_segments_per_file': float(len(pathnames_voice) / max(len(segments_voice), 1))
    }

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)

    # Save model
    model_path = os.path.join(output_dir, 'atc_model.pkl')
    cl.save(model_path)

    if verbose:
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"Total files processed: {len(filepaths)}")
        print(f"Voice segments extracted: {len(pathnames_voice)}")
        print(f"Voice clusters found: {n_voice_clusters}")
        print(f"Noise types found: {n_noise_clusters}")
        print(f"Average SNR improvement: {avg_snr:.2f} dB")
        print(f"\nResults saved to: {output_dir}")
        print(f"Model saved to: {model_path}")
        print("=" * 70)

    return {
        'enhanced': enhanced,
        'separated': separated,
        'segments_voice': segments_voice,
        'pathnames_voice': pathnames_voice,
        'results_voice': results_voice,
        'labels_voice': labels_voice,
        'results_noise': results_noise,
        'labels_noise': labels_noise,
        'summary': summary,
        'model_path': model_path,
        'clustaudio': cl
    }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Process ATC communications with clustaudio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python atc_example.py --input ./ATC_recordings/ --output ./results/

  # With custom enhancement
  python atc_example.py --input ./ATC/ --output ./out/ --enhancement aggressive

  # Without plots
  python atc_example.py --input ./ATC/ --output ./out/ --no-plots
        """
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing ATC audio files'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )

    parser.add_argument(
        '--enhancement', '-e',
        default='aggressive',
        choices=['light', 'moderate', 'aggressive'],
        help='Enhancement level (default: aggressive)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory does not exist: {args.input}")
        sys.exit(1)

    # Process
    results = process_atc_communications(
        input_dir=args.input,
        output_dir=args.output,
        enhancement_level=args.enhancement,
        save_plots=not args.no_plots,
        verbose=not args.quiet
    )

    if results is None or results.get('error'):
        sys.exit(1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
