import numpy as np
import librosa
from clustaudio import Clustaudio


# ---------------------------------------------------------
# Helpers (replace fixtures)
# ---------------------------------------------------------

def generate_fake_audio(sr=44100, n=100):
    """Generate synthetic audio using Clustaudio's built‑in example generator."""
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="wave", n=n), sr


def generate_fake_atc_audio(sr=44100, n=200):
    """Simulate ATC audio by generating a longer synthetic waveform."""
    cl = Clustaudio(method="mfcc", sr=sr)
    return cl.import_example(data="wave", n=n), sr


def generate_fake_audio_files(count=2, sr=44100):
    """Simulate multiple audio files by generating multiple synthetic signals."""
    return [generate_fake_atc_audio(sr=sr, n=200) for _ in range(count)]


# ---------------------------------------------------------
# DENOISING TESTS
# ---------------------------------------------------------

class TestDenoising:

    def test_spectral_subtraction_denoising(self):
        cl = Clustaudio()
        audio, sr = generate_fake_audio(n=100)
        audio = np.vstack(audio)

        denoised = cl.separation.denoise_audio(
            audio,
            sr=sr,
            method='spectral',
            stationary=True
        )

        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(audio)
        assert denoised.dtype in (np.float32, np.float64)

    def test_bandpass_filtering(self):
        cl = Clustaudio()
        audio, sr = generate_fake_audio(n=100)
        audio = np.vstack(audio)

        denoised = cl.separation.denoise_audio(
            audio,
            sr=sr,
            method='bandpass',
            lowcut=300,
            highcut=3400
        )

        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(audio)

    def test_denoising_on_atc_audio(self):
        cl = Clustaudio()
        audio, sr = generate_fake_audio(n=100)
        audio = np.vstack(audio)

        denoised = cl.separation.denoise_audio(
            audio,
            sr=sr,
            method='spectral'
        )

        assert isinstance(denoised, np.ndarray)
        assert len(denoised) > 0


# ---------------------------------------------------------
# SEPARATION TESTS
# ---------------------------------------------------------

class TestSeparation:

    def test_spectral_masking_separation(self):
        cl = Clustaudio()
        audio, sr = generate_fake_audio(n=100)
        audio = np.vstack(audio)

        result = cl.separation.separate_foreground_background(
            audio,
            sr=sr,
            method='spectral_masking'
        )

        assert 'foreground' in result
        assert 'background' in result
        assert 'foreground_mask' in result
        assert 'snr_db' in result

        assert isinstance(result['foreground'], np.ndarray)
        assert isinstance(result['background'], np.ndarray)
        assert len(result['foreground']) == len(audio)
        assert len(result['background']) == len(audio)

    def test_harmonic_percussive_separation(self):
        cl = Clustaudio()
        audio, sr = generate_fake_audio(n=100)
        audio = np.vstack(audio)

        result = cl.separation.separate_foreground_background(
            audio,
            sr=sr,
            method='harmonic_percussive'
        )

        assert 'foreground' in result
        assert 'background' in result
        assert 'snr_db' in result
        assert isinstance(result['snr_db'], np.float32)

    def test_separation_on_atc_audio(self):
        cl = Clustaudio()
        audio, sr = generate_fake_atc_audio()
        audio = np.vstack(audio)

        result = cl.separation.separate_foreground_background(
            audio,
            sr=sr,
            method='spectral_masking',
            noise_profile_duration=0.5
        )

        assert len(result['foreground']) > 0
        assert len(result['background']) > 0


# ---------------------------------------------------------
# ATC ENHANCEMENT TESTS
# ---------------------------------------------------------

class TestATCEnhancement:

    def test_light_atc_enhancement(self):
        audio, sr = generate_fake_atc_audio()
        audio = np.vstack(audio)
        cl = Clustaudio()

        enhanced = cl.separation.enhance_atc_audio(
            audio,
            sr=sr,
            method='light'
        )

        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) > 0

    def test_aggressive_atc_enhancement(self):
        audio, sr = generate_fake_atc_audio(n=10)
        audio = np.vstack(audio)
        cl = Clustaudio()

        enhanced = cl.separation.enhance_atc_audio(
            audio,
            sr=sr,
            method='aggressive'
        )

        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) > 0


# ---------------------------------------------------------
# CLUSTAUDIO INTEGRATION TESTS
# ---------------------------------------------------------

class TestClustaudioIntegration:

    def test_clustaudio_denoise_method(self, tmp_path):
        cl = Clustaudio(sr=44100)
        filepaths = cl.import_example(data='atc', n=3)
        results = cl.denoise(pathnames=filepaths, method='spectral', save_denoised=True,)

        assert 'audio_denoised' in results
        assert 'pathnames_denoised' in results
        assert np.all(list(map(lambda x: x is not None, results['pathnames_denoised'])))

    def test_clustaudio_separation_method(self, tmp_path):
        cl = Clustaudio(sr=44100)
        filepaths = cl.import_example(data='atc', n=3)

        results = cl.separate_foreground_background(
            pathnames=filepaths,
            method='spectral_masking',
            save_both=True,
        )

        assert 'pathnames_foreground' in results
        assert 'pathnames_background' in results
        assert 'snr_db' in results
        assert np.all(list(map(lambda x: x is not None, results['pathnames_foreground'])))

    def test_clustaudio_atc_enhancement(self, tmp_path):
        cl = Clustaudio(sr=44100)
        filepaths = cl.import_example(data='atc', n=3)

        results = cl.enhance_atc_audio(
            pathnames=filepaths,
            method='moderate',
        )

        assert 'pathnames_enhanced' in results
        assert np.all(list(map(lambda x: x is not None, results['pathnames_enhanced'])))
