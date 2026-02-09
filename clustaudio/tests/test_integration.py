# =============================================================================
# Integration tests
# =============================================================================
from clustaudio import Clustaudio
import numpy as np

class TestPipelines:
    """Test basic pipeline functionality"""

    def test_pipeline_run(self):
        cl = Clustaudio()
        dirpath = cl.import_example(data='atc')
        
        # Use combined features
        cl = Clustaudio()
        # Cluster with custom parameters
        cl.fit_transform(dirpath)
        # Check output variables
        assert set(cl.results.keys()) == {'audio', 'durations', 'feat', 'filenames', 'labels', 'pathnames', 'sample_rates', 'url', 'xycoord'}

    def test_imports(self):
        cl = Clustaudio()
        dirpath = cl.import_example(data='atc', n=10)
        # Cluster with custom parameters
        cl.fit_transform(dirpath)
        # Check output variables
        assert len(cl.results['pathnames']) == 10
        assert len(cl.results['filenames']) == 10
        assert cl.results['xycoord'].shape == (10, 2)
        assert len(cl.results['feat']) == 10
        assert len(cl.results['audio']) == 10
        assert len(cl.results['labels']) == 10
        assert len(cl.results['sample_rates']) == 10
        assert len(cl.results['durations']) == 10

    def test_sampleRateConversion(self):
        cl = Clustaudio()
        dirpath = cl.import_example(data='atc', n=3)
        # Use combined features
        for sr in [22050, 44100, 48000]:
            cl = Clustaudio(sr=sr)
            # Cluster with custom parameters
            cl.fit_transform(dirpath)
            # Check output variables
            assert np.all(list(map(lambda x: x==sr, cl.results['sample_rates'])))
            
    def test_checkDuration(self):
        cl = Clustaudio()
        dirpath = cl.import_example(data='atc', n=3)
        # Cluster with custom parameters
        cl.fit_transform(dirpath)
        # Check output variables
        assert np.all(list(map(lambda x: x>0, cl.results['durations'])))
