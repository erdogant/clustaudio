clustaudio's documentation!
===========================

|python| |pypi| |docs| |stars| |LOC| |downloads_month| |downloads_total| |license| |forks| |open issues| |project status| |medium| |colab| |DOI| |repo-size| |donate|

.. tip::
	`Medium Blog: Unsupervised Audio Clustering Guide <https://towardsdatascience.com/unsupervised-audio-clustering-guide>`_


.. |fig1| image:: ../figs/schematic_overview_audio.png

.. table::
   :align: center

   +----------+
   | |fig1|   |
   +----------+

-----------------------------------

The aim of ``clustaudio`` is to detect natural groups or clusters of audio files.

Many audio analysis tasks rely on supervised machine learning or deep neural networks to classify sounds into pre-defined categories. However, not all tasks require supervised approaches. With an unsupervised approach, we can discover natural groups or clusters of audio files without being constrained to fixed categories. This library enables clustering of music, speech, environmental sounds, and other audio data using advanced feature extraction (MFCC, Spectral, Chroma) and robust clustering evaluation methods. All results are derived using the Python library clustaudio.

``clustaudio`` is a comprehensive approach for unsupervised audio clustering and addresses the following challenges:

    * 1. Robustly groups similar audio files.
    * 2. Returns the unique audio samples.
    * 3. Waveforms, spectrograms, and cluster plots for deeper exploration.
    * 4. Finds highly similar audio files for a given input sample.
    * 5. Advanced denoising and audio separation capabilities.
    * 6. Specialized processing for noisy communications (ATC, radio).


-----------------------------------

.. note::
	**Your ❤️ is important to keep maintaining this package.** You can `support <https://erdogant.github.io/clustaudio/pages/html/Documentation.html>`_ in various ways, have a look at the `sponsor page <https://erdogant.github.io/clustaudio/pages/html/Documentation.html>`_.
	Report bugs, issues and feature extensions at `github <https://github.com/erdogant/clustaudio/>`_ page.

	.. code-block:: console

	   pip install clustaudio

-----------------------------------



Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Background

   Abstract


.. toctree::
   :maxdepth: 1
   :caption: Installation

   Installation


.. toctree::
  :maxdepth: 1
  :caption: Core functionalities

  core_functions

.. toctree::
  :maxdepth: 1
  :caption: Feature Extraction

  Feature_Extraction


.. toctree::
  :maxdepth: 1
  :caption: Cluster Evaluation

  Cluster_Evaluation

.. toctree::
  :maxdepth: 1
  :caption: Find/Predict

  Find


.. toctree::
  :maxdepth: 1
  :caption: Performance

  Performance

.. toctree::
  :maxdepth: 1
  :caption: Save and Load

  Save_and_Load


.. toctree::
  :maxdepth: 1
  :caption: Documentation

  clustaudio.clustaudio




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



.. |repo-size| image:: https://img.shields.io/github/repo-size/erdogant/clustaudio
    :alt: repo-size
    :target: https://img.shields.io/github/repo-size/erdogant/clustaudio

.. |stars| image:: https://img.shields.io/github/stars/erdogant/clustaudio
    :alt: Stars
    :target: https://img.shields.io/github/stars/erdogant/clustaudio

.. |python| image:: https://img.shields.io/pypi/pyversions/clustaudio.svg
    :alt: |Python
    :target: https://erdogant.github.io/clustaudio/

.. |pypi| image:: https://img.shields.io/pypi/v/clustaudio.svg
    :alt: |Python Version
    :target: https://pypi.org/project/clustaudio/

.. |docs| image:: https://img.shields.io/badge/Sphinx-Docs-blue.svg
    :alt: Sphinx documentation
    :target: https://erdogant.github.io/clustaudio/

.. |LOC| image:: https://sloc.xyz/github/erdogant/clustaudio/?category=code
    :alt: lines of code
    :target: https://github.com/erdogant/clustaudio

.. |downloads_month| image:: https://static.pepy.tech/personalized-badge/clustaudio?period=month&units=international_system&left_color=grey&right_color=brightgreen&left_text=PyPI%20downloads/month
    :alt: Downloads per month
    :target: https://pepy.tech/project/clustaudio

.. |downloads_total| image:: https://static.pepy.tech/personalized-badge/clustaudio?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads
    :alt: Downloads in total
    :target: https://pepy.tech/project/clustaudio

.. |license| image:: https://img.shields.io/badge/license-MIT-green.svg
    :alt: License
    :target: https://github.com/erdogant/clustaudio/blob/master/LICENSE

.. |forks| image:: https://img.shields.io/github/forks/erdogant/clustaudio.svg
    :alt: Github Forks
    :target: https://github.com/erdogant/clustaudio/network

.. |open issues| image:: https://img.shields.io/github/issues/clustaudio.svg
    :alt: Open Issues
    :target: https://github.com/erdogant/clustaudio/issues

.. |project status| image:: http://www.repostatus.org/badges/latest/active.svg
    :alt: Project Status
    :target: http://www.repostatus.org/#active

.. |medium| image:: https://img.shields.io/badge/Medium-Blog-green.svg
    :alt: Medium Blog
    :target: https://erdogant.github.io/clustaudio/pages/html/Documentation.html#medium-blog

.. |donate| image:: https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors
    :alt: donate
    :target: https://erdogant.github.io/clustaudio/pages/html/Documentation.html#

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Colab example
    :target: https://erdogant.github.io/clustaudio/pages/html/Documentation.html#colab-notebook

.. |DOI| image:: https://zenodo.org/badge/423822054.svg
    :alt: Cite
    :target: https://zenodo.org/badge/latestdoi/423822054


.. include:: add_bottom.add
