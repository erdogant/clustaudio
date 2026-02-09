import numpy as np
import librosa
import os
from clustaudio import Clustaudio


# ---------------------------------------------------------
# Helpers for generating synthetic audio
# ---------------------------------------------------------

def generate_fake_audio(sr=44100, n=100):
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="wave", n=n), sr


def generate_fake_audio_files(count=4, sr=44100):
    """Generate multiple synthetic audio signals."""
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="atc", n=count)



# ---------------------------------------------------------
# FIT_TRANSFORM TESTS
# ---------------------------------------------------------

class TestFitTransform:

    def test_fit_transform_basic(self, tmp_path):
        paths = generate_fake_audio_files(4)

        cl = Clustaudio(method='mfcc', verbose='warning')
        results = cl.fit_transform(paths, min_clust=2, max_clust=4)

        assert results is not None
        assert 'feat' in results
        assert 'xycoord' in results
        assert 'labels' in results
        assert 'pathnames' in results

        assert results['feat'].shape[0] == len(paths)
        assert len(results['labels']) == len(paths)

    def test_fit_transform_with_synthetic_variety(self, tmp_path):
        paths = generate_fake_audio_files(6)

        cl = Clustaudio(method='mfcc', verbose='warning')
        results = cl.fit_transform(paths, min_clust=2, max_clust=3)

        assert results['feat'].shape[0] == len(paths)
        assert len(results['labels']) == len(paths)

        unique_labels = np.unique(results['labels'])
        assert 2 <= len(unique_labels) <= 3

    def test_fit_transform_stores_results(self, tmp_path):
        paths = generate_fake_audio_files(3)

        cl = Clustaudio(method='mfcc', verbose='warning')
        results = cl.fit_transform(paths, min_clust=2, max_clust=3)

        assert cl.results == results
        assert cl.results['feat'] is not None
        assert cl.results['labels'] is not None


# ---------------------------------------------------------
# CLUSTERING TESTS
# ---------------------------------------------------------

class TestClustering:

    def test_cluster_after_feature_extraction(self, tmp_path):
        paths = generate_fake_audio_files(4)

        cl = Clustaudio(method='mfcc', verbose='warning')

        cl.import_data(paths)
        cl.extract_feat(cl.results)
        cl.embedding(cl.results['feat'])

        labels = cl.cluster(min_clust=2, max_clust=3)

        assert labels is not None
        assert len(labels) == len(paths)
        assert cl.results['labels'] is not None

    def test_cluster_agglomerative(self, tmp_path):
        paths = generate_fake_audio_files(4)

        cl = Clustaudio(method='mfcc', verbose='warning')
        cl.fit_transform(paths, min_clust=2, max_clust=3)

        labels = cl.cluster(cluster='agglomerative', min_clust=2, max_clust=3)

        assert labels is not None
        assert len(labels) == len(cl.results['pathnames'])

    def test_cluster_optimal_selection(self, tmp_path):
        paths = generate_fake_audio_files(5)

        cl = Clustaudio(method='mfcc', verbose='warning')
        cl.fit_transform(paths, min_clust=2, max_clust=4)

        labels = cl.results['labels']
        n_clusters = len(np.unique(labels))

        assert 2 <= n_clusters <= 4


# ---------------------------------------------------------
# UNIQUE REPRESENTATIVE TESTS
# ---------------------------------------------------------

class TestUnique:

    def test_unique_finds_representatives(self, tmp_path):
        paths = generate_fake_audio_files(5)

        cl = Clustaudio(method='mfcc', verbose='warning')
        cl.fit_transform(paths, min_clust=2, max_clust=3)

        unique_results = cl.unique()

        assert 'labels' in unique_results
        assert 'idx' in unique_results
        assert 'xycoord_center' in unique_results
        assert 'pathnames' in unique_results

        n_clusters = len(np.unique(cl.results['labels']))
        assert len(unique_results['labels']) == n_clusters
        assert len(unique_results['idx']) == n_clusters

    def test_unique_representatives_are_valid_indices(self, tmp_path):
        paths = generate_fake_audio_files(4)

        cl = Clustaudio(method='mfcc', verbose='warning')
        cl.fit_transform(paths, min_clust=2, max_clust=3)

        unique_results = cl.unique()
        n_samples = len(cl.results['pathnames'])

        for idx in unique_results['idx']:
            assert 0 <= idx < n_samples

    def test_unique_stored_in_results(self, tmp_path):
        paths = generate_fake_audio_files(4)

        cl = Clustaudio(method='mfcc', verbose='warning')
        cl.fit_transform(paths, min_clust=2, max_clust=3)

        cl.unique()

        assert hasattr(cl, 'results_unique')
        assert cl.results_unique is not None


# ---------------------------------------------------------
# EMBEDDING TESTS
# ---------------------------------------------------------

class TestEmbedding:

    def test_tsne_embedding(self):
        mock_features = np.random.randn(10, 5)

        cl = Clustaudio(embedding='tsne', verbose='warning')
        cl.results['feat'] = mock_features

        coords = cl.embedding(mock_features, embedding='tsne')

        assert coords.shape == (10, 2)
        assert cl.results['xycoord'] is not None
        np.testing.assert_array_equal(cl.results['xycoord'], coords)

    def test_pca_embedding(self):
        mock_features = np.random.randn(12, 6)

        cl = Clustaudio(embedding='pca', verbose='warning')
        cl.results['feat'] = mock_features

        coords = cl.embedding(mock_features, embedding='pca')

        assert coords.shape == (12, 5)

    def test_embedding_no_method(self):
        mock_features = np.random.randn(8, 4)

        cl = Clustaudio(embedding=None, verbose='warning')
        cl.results['feat'] = mock_features

        coords = cl.embedding(mock_features, embedding=None)

        assert coords.shape == (8, 2)
        np.testing.assert_array_equal(coords, mock_features[:, :2])


# ---------------------------------------------------------
# CLUSTERING WITH SYNTHETIC "REAL" AUDIO
# ---------------------------------------------------------

class TestClusteringWithSyntheticAudio:

    def test_cluster_similar_sounds(self, tmp_path):
        # Generate 3 similar synthetic signals
        paths = generate_fake_audio_files(10)

        cl = Clustaudio(method='mfcc', verbose='warning')
        results = cl.fit_transform(paths, min_clust=2, max_clust=2)

        assert results['labels'] is not None
        assert len(results['labels']) == 10

    def test_cluster_atc_audio(self, tmp_path):
        paths = generate_fake_audio_files(10)

        cl = Clustaudio(method='mfcc', sr=44100, verbose='warning')
        results = cl.fit_transform(paths, min_clust=2, max_clust=4)

        assert results['feat'].shape[0] == len(paths)
        assert len(results['labels']) == len(paths)
        assert all(isinstance(label, (int, np.integer)) for label in results['labels'])
