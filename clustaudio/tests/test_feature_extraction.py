import numpy as np
from clustaudio import Clustaudio



# -----------------------------
# MFCC TESTS
# -----------------------------

def test_mfcc_single_audio():
    # Initialize
    sr = 44100
    cl = Clustaudio(method="mfcc", verbose="warning")
    audio = cl.import_example(data='wave', n=1, freq=440, sr=sr, duration=1.0)

    # Extract features
    features = cl.extract_mfcc(audio, sr=sr)
    assert isinstance(features, np.ndarray)
    expected = cl.params_mfcc["n_mfcc"] * 2
    assert len(features) == expected
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_mfcc_batch():
    # Initialize
    sr = 44100
    cl = Clustaudio(method="mfcc", verbose="warning")
    audio = cl.import_example(data='wave', n=10, freq=440, sr=sr, duration=1.0)

    # Extract features
    features = cl.extract_mfcc(audio, sr=sr)
    assert isinstance(features, np.ndarray)
    expected = cl.params_mfcc["n_mfcc"] * 2
    assert features.shape[1] == expected
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_mfcc_custom_parameters():
    cl = Clustaudio(method="mfcc", params_mfcc={"n_mfcc": 20, "n_fft": 1024, "hop_length": 256})
    audio = cl.import_example(data='wave', n=100)
    features = cl.extract_mfcc(audio)
    assert features.shape[1] == 40


# -----------------------------
# SPECTRAL TESTS
# -----------------------------

def test_spectral_single_audio():
    sr = 44100
    cl = Clustaudio(method="spectral")
    audio = cl.import_example(data='wave', n=10, freq=440, sr=sr, duration=1.0)
    # Extract features
    features = cl.extract_spectral(audio, sr=sr)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 12
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_spectral_batch():
    cl = Clustaudio(method="spectral")
    audio = cl.import_example(data='wave', n=100)
    features = cl.extract_spectral(audio)
    assert features.shape[1] == 12
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


# -----------------------------
# CHROMA TESTS
# -----------------------------

def test_chroma_single_audio():
    cl = Clustaudio(method="chroma")
    audio = cl.import_example(data='wave', n=10)
    # Extract features
    features = cl.extract_chroma(audio)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 24
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

def test_chroma_batch():
    cl = Clustaudio(method="chroma")
    audio = cl.import_example(data='wave', n=100)
    features = cl.extract_chroma(audio)
    assert features.shape[1] == 24
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
    


# -----------------------------
# VALIDATION TESTS
# -----------------------------

def test_extract_feat_empty_list():
    cl = Clustaudio(method="mfcc", verbose="warning")

    result = {"audio": [], "pathnames": [], "filenames": []}
    features = cl.extract_feat(result)

    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 0


def test_same_audio_same_features():
    sr = 44100
    cl = Clustaudio(method="mfcc", sr=sr)
    audio = cl.import_example(data='wave', n=100)

    f1 = cl.extract_mfcc(audio, sr=sr)
    f2 = cl.extract_mfcc(audio, sr=sr)

    np.testing.assert_array_almost_equal(f1, f2)


def test_different_audio_different_features():
    sr = 44100
    cl = Clustaudio(method="mfcc",  sr=sr)
    audio1 = cl.import_example(data='wave', freq=440)
    audio2 = cl.import_example(data='wave', freq=880)

    f1 = cl.extract_mfcc(audio1, sr=sr)
    f2 = cl.extract_mfcc(audio2, sr=sr)

    assert not np.allclose(f1, f2)
