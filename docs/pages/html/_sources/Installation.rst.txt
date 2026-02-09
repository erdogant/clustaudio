Installation
################


Pypi
****

.. code-block:: console

   pip install clustaudio


Github source
**************

.. code-block:: console

   git clone https://github.com/erdogant/clustaudio.git
   cd clustaudio
   pip install -U .


Dependencies
*************

``clustaudio`` requires the following core dependencies:

**Audio Processing (Required)**

.. code-block:: console

   pip install librosa soundfile


**Core Dependencies (Required)**

.. code-block:: console

   pip install numpy pandas scikit-learn scipy matplotlib tqdm


**Original Library Dependencies (Required)**

.. code-block:: console

   pip install pca distfit clusteval scatterd colourmap datazets pypickle ismember


**Phase 2 Dependencies (Optional but Recommended)**

For enhanced metadata extraction:

.. code-block:: console

   pip install mutagen tinytag


For advanced Voice Activity Detection:

.. code-block:: console

   pip install webrtcvad


For audio fingerprinting:

.. code-block:: console

   pip install pyacoustid


For metadata extraction (alternative):

.. code-block:: console

   pip install eyed3


**Phase 3 Dependencies (Optional)**

For advanced noise reduction:

.. code-block:: console

   pip install noisereduce


For format conversion:

.. code-block:: console

   pip install pydub


Complete Installation
**********************

To install all dependencies at once:

.. code-block:: console

   # Core only
   pip install clustaudio

   # With recommended optional packages
   pip install clustaudio mutagen tinytag webrtcvad noisereduce


Supported Audio Formats
*************************

``clustaudio`` supports the following audio formats:

* **WAV** (.wav) - Uncompressed, lossless
* **MP3** (.mp3) - Compressed, lossy
* **FLAC** (.flac) - Compressed, lossless
* **OGG** (.ogg) - Compressed, lossy
* **M4A** (.m4a) - Compressed (AAC), lossy
* **AAC** (.aac) - Compressed, lossy

.. note::
   For best performance and compatibility, WAV and FLAC formats are recommended.
   MP3 and other compressed formats may have slight quality degradation.


System Requirements
********************

**Minimum Requirements:**

* Python 3.7 or higher
* 4 GB RAM
* 500 MB disk space

**Recommended Requirements:**

* Python 3.9 or higher
* 8 GB RAM or more
* 2 GB disk space for caching
* Multi-core CPU (for faster processing)


Troubleshooting
****************

**librosa installation issues:**

On some systems, you may need to install additional audio codecs:

.. code-block:: console

   # Ubuntu/Debian
   sudo apt-get install libsndfile1

   # macOS (using Homebrew)
   brew install libsndfile

   # Windows
   # Usually works out of the box with pip install


**soundfile issues:**

If soundfile fails to load audio:

.. code-block:: console

   pip install soundfile --upgrade


**webrtcvad compilation issues:**

If webrtcvad fails to install:

.. code-block:: console

   # Try installing build tools first
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev

   # macOS
   xcode-select --install


**noisereduce installation:**

.. code-block:: console

   pip install noisereduce==2.0.0


Verification
*************

To verify your installation:

.. code-block:: python

   import clustaudio
   print(clustaudio.__version__)

   # Check if audio can be loaded
   cl = clustaudio.clustaudio(method='mfcc')
   print("Installation successful!")


To check optional dependencies:

.. code-block:: python

   # Check librosa
   import librosa
   print(f"librosa version: {librosa.__version__}")

   # Check if advanced features are available
   try:
       import webrtcvad
       print("WebRTC VAD available: Yes")
   except ImportError:
       print("WebRTC VAD available: No")

   try:
       import noisereduce
       print("Noisereduce available: Yes")
   except ImportError:
       print("Noisereduce available: No")

   try:
       import mutagen
       print("Mutagen (metadata) available: Yes")
   except ImportError:
       print("Mutagen (metadata) available: No")


Getting Started
****************

After installation, see the Examples section for usage tutorials, or try the quick start:

.. code-block:: python

   from clustaudio import clustaudio

   # Initialize
   cl = clustaudio(method='mfcc', sr=22050)

   # Cluster audio files
   results = cl.fit_transform('/path/to/audio/files/')

   # Plot results
   cl.plot_statistics()

For more examples, see: :ref:`Examples`


.. include:: add_bottom.add
