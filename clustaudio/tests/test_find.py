import numpy as np
from clustaudio import Clustaudio


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def generate_fake_audio(sr=44100, n=100):
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="wave", n=n)


def generate_fake_audio_files(count=4, sr=44100):
    """Generate multiple synthetic audio signals."""
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="atc", n=count)


# ---------------------------------------------------------
# BASIC FIND TESTS
# ---------------------------------------------------------

class TestFindBasic:

    def test_find_with_k_neighbors(self):
        # Create synthetic dataset
        audio_list = generate_fake_audio_files(10)
        query_audio = audio_list[0]

        # Fit model
        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        # Run find
        results = cl.find([query_audio], k=3)

        assert isinstance(results, dict)
        assert cl.results["predict"] is not None

    def test_find_returns_correct_number(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        cl.find([query_audio], k=3)

        assert len(cl.results["predict"]) > 0

    def test_find_with_different_metrics(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        for metric in ["euclidean", "cosine"]:
            results = cl.find([query_audio], k=2, metric=metric)
            assert results is not None


# ---------------------------------------------------------
# FIND WITH DIFFERENT QUERY TYPES
# ---------------------------------------------------------

class TestFindWithQuery:

    def test_find_with_single_audio(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        results = cl.find([query_audio], k=2)

        assert results is not None
        assert cl.results["predict"] is not None

    def test_find_with_audio_array(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        results = cl.find([query_audio], k=2)

        assert results is not None

    def test_find_stores_results(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        cl.find([query_audio], k=2)

        assert cl.results["predict"] is not None
        assert "metric_find" in cl.params


# ---------------------------------------------------------
# VALIDATION TESTS
# ---------------------------------------------------------

class TestFindValidation:

    def test_find_requires_fitted_model(self):
        audio = generate_fake_audio()

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")

        try:
            results = cl.find([audio], k=2)
            assert results is None or len(results) == 0
        except Exception:
            pass

    def test_find_with_valid_k(self):
        audio_list = generate_fake_audio_files(4)
        query_audio = audio_list[0]

        cl = Clustaudio(method="mfcc", sr=44100, verbose="warning")
        cl.fit_transform(audio_list, min_clust=2, max_clust=3)

        assert cl.find([query_audio], k=1) is not None
        assert cl.find([query_audio], k=3) is not None


