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
# AUDIOREAD TESTS
# ---------------------------------------------------------

class TestAudioRead:

    def test_read_wav_file(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(verbose='warning')
        loaded = cl.audioread(paths[0])

        assert isinstance(loaded, np.ndarray)
        assert len(loaded) > 0
        assert loaded.ndim == 1
        assert np.max(np.abs(loaded)) <= 1.0

    def test_read_nonexistent_file(self):
        cl = Clustaudio(verbose='warning')
        audio, success = cl.audioread("nonexistent.wav", return_success=True)

        assert success is False
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0


# ---------------------------------------------------------
# PREPROCESSING TESTS
# ---------------------------------------------------------

class TestPreprocessing:

    def test_preprocess_single_file(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(verbose='warning')
        result = cl.preprocessing(paths)

        assert 'audio' in result
        assert 'pathnames' in result
        assert 'filenames' in result
        assert 'sample_rates' in result
        assert 'durations' in result

        assert len(result['audio']) == 4
        assert isinstance(result['audio'][0], np.ndarray)
        assert len(result['audio'][0]) > 0

    def test_preprocess_multiple_files(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(verbose='warning')
        result = cl.preprocessing(paths)

        assert len(result['audio']) == len(paths)
        assert len(result['pathnames']) == len(paths)

        for audio in result['audio']:
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            assert audio.ndim == 1

    def test_preprocess_with_duration_limit(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(duration=1.0, verbose='warning')
        result = cl.preprocessing(paths)

        for dur in result['durations']:
            assert dur <= 1.1


# ---------------------------------------------------------
# PREPROCESSING PARAMETER TESTS
# ---------------------------------------------------------

class TestPreprocessingParameters:

    def test_mono_conversion(self, tmp_path):
        audio, sr = generate_fake_audio()
        stereo = np.vstack([audio, audio]).T  # fake stereo
        cl = Clustaudio(mono=True, verbose='warning')

    def test_sample_rate_conversion(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(sr=16000, verbose='warning')
        result = cl.preprocessing(paths)

        assert result['sample_rates'][0] == 16000

    def test_normalization(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(normalize=True, verbose='warning')
        result = cl.preprocessing(paths)

        assert np.max(np.abs(result['audio'][0])) <= 1.0


# ---------------------------------------------------------
# IMPORT_DATA TESTS
# ---------------------------------------------------------

class TestImportData:

    def test_import_from_file_list(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(verbose='warning')
        result = cl.import_data(paths)

        assert len(result['audio']) == len(paths)
        assert len(result['pathnames']) == len(paths)

    def test_import_from_single_file(self, tmp_path):
        paths = generate_fake_audio_files()

        cl = Clustaudio(verbose='warning')
        result = cl.import_data(paths)

        assert len(result['audio']) == 4

