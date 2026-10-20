import logging

from clustaudio.clustaudio import Clustaudio

import clustaudio.vad as vad
import clustaudio.separation as separation
import clustaudio.visualization as visualization
import clustaudio.visualization as visualization
import clustaudio.fingerprint as fingerprint
import clustaudio.metadata as metadata

from clustaudio.clustaudio import (
    import_example,
    listdir,
    set_logger,
    move_files,
    )


__author__ = 'Erdogan Tasksen'
__email__ = 'erdogant@gmail.com'
__version__ = '0.1.1'

# Setup root logger
_logger = logging.getLogger('clustaudio')
_log_handler = logging.StreamHandler()
_fmt = '[{asctime}] [{name}] [{levelname}] {msg}'
_formatter = logging.Formatter(fmt=_fmt, style='{', datefmt='%d-%m-%Y %H:%M:%S')
_log_handler.setFormatter(_formatter)
_log_handler.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)
_logger.propagate = False

# module level doc-string
__doc__ = """
clustaudio
=====================================================================

Python package clustaudio is to detect natural groups or clusters of images.

The aim of ``clustaudio`` is to detect natural groups or clusters of images. It works using a multi-step proces of carefully pre-processing the images, extracting the features, and evaluating the optimal number of clusters across the feature space.
The optimal number of clusters are determined using well known methods suchs as *silhouette, dbindex, and derivatives* in combination with clustering methods, such as *agglomerative, kmeans, dbscan and hdbscan*.
With ``clustaudio`` we aim to determine the most robust clustering by efficiently searching across the parameter and evaluation the clusters.
Besides clustering of images, the ``clustaudio`` model can also be used to find the most similar images for a new unseen sample.

Example
-------
>>> from clustaudio import clustaudio
>>>
>>> # Init with default settings
>>> cl = clustaudio(method='pca')
>>>
>>> # load example dataset
>>> X, y = cl.import_example(data='mnist')
>>>
>>> # Cluster digits
>>> results = cl.fit_transform(X)
>>>
>>> # Cluster evaluation
>>> cl.clustaudio.plot()
>>> cl.clustaudio.scatter(cl.results['xycoord'])
>>> cl.pca.plot()
>>>
>>> # Unique
>>> cl.plot_unique(img_mean=False)
>>> cl.results_unique.keys()
>>>
>>> # Scatter
>>> cl.scatter(img_mean=False, zoom=3)
>>>
>>> # Plot clustered images
>>> cl.plot(labels=8)
>>>
>>> # Plot dendrogram
>>> cl.dendrogram()
>>>
>>> # Find images
>>> results_find = cl.find(X[0,:], k=None, alpha=0.05)
>>> cl.plot_find()
>>> cl.scatter()
>>>

References
----------
https://github.com/erdogant/clustaudio

"""
