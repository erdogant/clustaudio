Installation
################

Create environment
**********************

If desired, install ``clustaudio`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_clustaudio python=3.8
    conda activate env_clustaudio


Pypi
**********************

.. code-block:: console

    # Install from Pypi:
    pip install clustaudio

    # Force update to latest version
    pip install -U clustaudio


Github source
************************************

.. code-block:: console

    # Install directly from github
    pip install git+https://github.com/erdogant/clustaudio


Uninstalling
################

Remove environment
**********************

.. code-block:: console

   # List all the active environments. clustaudio should be listed.
   conda env list

   # Remove the clustaudio environment
   conda env remove --name clustaudio

   # List all the active environments. clustaudio should be absent.
   conda env list


Remove installation
**********************

Note that the removal of the environment will also remove the ``clustaudio`` installation.

.. code-block:: console

    # Install from Pypi:
    pip uninstall clustaudio



Quickstart
**********************

A quick example how to learn a model on a given dataset.

.. code:: python

    # Import library
    from clustaudio import clustaudio

    # init with default parameters
    cl = clustaudio()

    # load example with flowers
    path_to_imgs = cl.import_example(data='flowers')

    # Run the model to find the optimal clusters
    results = cl.fit_transform(path_to_imgs, min_clust=10)

    # Cluster evaluation plot
    cl.clustaudio.plot()
    
    # Unique images
    cl.results_unique.keys()
    cl.plot_unique(img_mean=False)

    # Scatter
    cl.scatter(dotsize=50, img_mean=False)

    # Plot clustered images
    cl.plot(labels=0)

    # Plot dendrogram
    cl.dendrogram()

    # Predict
    results_find = cl.find(path_to_imgs[0:5], k=None, alpha=0.05)
    cl.plot_find()
    cl.scatter()




.. include:: add_bottom.add