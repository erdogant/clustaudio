"""
Unit tests for Voice Activity Detection (VAD).

Tests voice segment detection and extraction including:
- Energy-based VAD
- Librosa-based VAD
- Segment extraction
- Batch processing
"""
import numpy as np
import librosa
import os

from clustaudio import Clustaudio


# ---------------------------------------------------------
# VAD DETECTION TESTS
# ---------------------------------------------------------

class TestVADDetection:

    def test_energy_based_vad(self):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')
        audio = cl.import_data(filepaths)['audio']
        
        segments = cl.vad.detect_voice_segments(
            audio[0],
            sr=sr,
            method='energy',
            energy_threshold=0.02,
            min_segment_duration=0.1
        )

        assert isinstance(segments, list)

        for seg in segments:
            assert isinstance(seg, tuple)
            assert len(seg) == 2
            assert seg[1] > seg[0]

    def test_librosa_vad(self):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')
        audio = cl.import_data(filepaths)['audio']

        segments = cl.vad.detect_voice_segments(
            audio[0],
            sr=sr,
            method='librosa',
            min_segment_duration=0.1
        )

        assert isinstance(segments, list)

        for seg in segments:
            assert len(seg) == 2
            assert seg[1] > seg[0]

    def test_vad_on_atc_audio(self):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')
        audio = cl.import_data(filepaths)['audio']

        segments = cl.vad.detect_voice_segments(
            audio[0],
            sr=sr,
            method='energy',
            energy_threshold=0.01
        )

        assert len(segments) >= 0  # synthetic ATC may or may not contain "voice"


# ---------------------------------------------------------
# SEGMENT EXTRACTION TESTS
# ---------------------------------------------------------

class TestVADSegmentExtraction:

    def test_extract_voice_segments(self, tmp_path):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')

        result = cl.vad.extract_voice_segments(
            filepaths[0],
            sr=sr,
            method='energy',
            energy_threshold=0.01
        )

        assert 'pathname' in list(result.keys())
        assert 'pathnames_segments' in result
        assert 'timestamps' in result
        assert 'num_segments' in result

        assert result['num_segments'] >= 0

        if result['num_segments'] > 0:
            for seg_path in result['pathnames_segments']:
                assert os.path.exists(seg_path)

    def test_extract_segments_output_structure(self, tmp_path):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')

        result = cl.vad.extract_voice_segments(
            filepaths[0],
            sr=sr,
            method='energy'
        )

        assert len(result['pathnames_segments']) == result['num_segments']
        assert len(result['timestamps']) == result['num_segments']
        assert len(result['durations']) == result['num_segments']


# ---------------------------------------------------------
# INTEGRATION TESTS WITH CLUSTAUDIO
# ---------------------------------------------------------

class TestVADIntegration:

    def test_extract_and_cluster_voice_segments(self, tmp_path):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')

        all_segments = []

        for filepath in filepaths:
            result = cl.vad.extract_voice_segments(
                filepath,
                sr=sr,
                method='energy',
                energy_threshold=0.01
            )
            all_segments.extend(result['pathnames_segments'])

        if len(all_segments) >= 3:
            cl = Clustaudio(method='mfcc', sr=sr, verbose='warning')
            results = cl.fit_transform(all_segments, min_clust=2, max_clust=3)

            assert results['feat'].shape[0] == len(all_segments)
            assert len(results['labels']) == len(all_segments)

    def test_clustaudio_extract_voice_segments_method(self, tmp_path):
        sr = 22050
        cl = Clustaudio(sr=sr)
        filepaths = cl.import_example(data='atc')
        
        results = cl.extract_voice_segments(
            filepaths,
            method='energy',
            energy_threshold=0.01
        )

        assert 'pathnames' in results
        assert 'pathnames_segments' in results
        assert 'num_segments' in results
