============
Usage
============

Quick start usage
----------------------

.. code-block:: python

    import netlsd
    import networkx as nx

    g = nx.erdos_renyi_graph(100, 0.01) # create a random graph with 100 nodes
    descriptor = netlsd.heat(g) # compute NetLSD signature

That's it! Then, signatures of two graphs can be compared easily::

.. code-block:: python

    import netlsd

    distance = netlsd.compare(sig1, sig2) # compare the signatures using l2 distance

or, equivalently::

.. code-block:: python

    import numpy as np

    distance = np.linalg.norm(sig1 - sig2) # compare the signatures using l2 distance in numpy

Advanced usage
----------------------

Here we outline different ways to get more out of NetLSD.

Try the wave kernel
~~~~~~~~

In the paper, we introduce two kernels: heat and wave.
You can simply replace ``netlsd.heat`` with ``netlsd.wave`` to switch to wave kernel.
Wave kernel is known to preserve symmetries and structures as it acts as a band-pass filter on the spectrum.

Supply adjacency matrix directly
~~~~~~~~

You do not need to use python's graph libraries to interface with NetLSD.
One option is to use any type of a sparse matrix from scipy:

.. code-block:: python

    import netlsd
    import scipy.sparse as sps

    A = sps.random(1000, 1000) # create a random adjacency matrix
    A = A + A.T # make sure it is undirected
    descriptor = netlsd.heat(A) # compute NetLSD signature

In case you have already constructed a Laplacian, just pass it to the function.

Scale things up with custom eigensolvers
~~~~~~~~

If you want to use a different eigensolver routine, such as SLEPc, you can directly supply eigenvalues to NetLSD:

.. code-block:: python

    import netlsd
    import fancy_eigensolver

    eigenvalues = fancy_eigensolver(graph)
    descriptor = netlsd.heat(eigenvalues) # compute NetLSD signature