Saving
##########

Saving and loading models can be used to restore previous results.
In order to accomplish this, we created two functions: function :func:`clustaudio.clustaudio.clustaudio.save`. and function :func:`clustaudio.clustaudio.clustaudio.load`.
Below we illustrate how to save and load models.

Saving the model with the results:

.. code:: python

    from clustaudio import clustaudio

    # Initialize
    cl = clustaudio(method='hog')
    # Load example data

    X, y = cl.import_example(data='mnist')
    # Preprocessing, feature extraction and cluster evaluation
    results = cl.fit_transform(X)

    # Load model
    cl.save('clustaudio_model', overwrite=True)


Loading
##########

Loading a learned model can be done using the function :func:`clustaudio.load`:

.. code:: python

    from clustaudio import clustaudio

    # Load model
    cl.load('clustaudio_model')




.. include:: add_bottom.add