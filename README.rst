===============================
NetLSD
===============================

NetLSD is a family of spectral graph descriptros. Given a graph, NetLSD computes a low-dimensional vector representation that can be used for different tasks.

Quick start
-----------

.. code-block:: python

    import netlsd
    import networkx as nx

    g = nx.erdos_renyi_graph(100, 0.01) # create a random graph with 100 nodes
    descriptor = netlsd.heat(g) # compute the signature

That's it! Then, signatures of two graphs can be compared easily. NetLSD supports `networkx <http://networkx.github.io/>`_, `graph_tool <https://graph-tool.skewed.de/>`_, and `igraph <http://igraph.org/python/>`_ packages natively.

.. code-block:: python

    import netlsd
    import numpy as np

    distance = netlsd.compare(desc1, desc2) # compare the signatures using l2 distance
    distance = np.linalg.norm(desc1 - desc2) # equivalent


For more advanced usage, check out `online documentation <http://netlsd.readthedocs.org/>`_.


Requirements
------------
* numpy
* scipy


Installation
------------
#. cd netlsd
#. pip install -r requirements.txt 
#. python setup.py install

Or simply ``pip install netlsd``

Citing
------
If you find NetLSD useful in your research, we ask that you cite the following paper::

    @inproceedings{Tsitsulin:2018:KDD,
     author={Tsitsulin, Anton and Mottin, Davide and Karras, Panagiotis and Bronstein, Alex and M{\"u}ller, Emmanuel},
     title={NetLSD: Hearing the Shape of a Graph},
     booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
     series = {KDD '18},
     year = {2018},
    } 

Misc
----

NetLSD - Hearing the shape of graphs.

* MIT license
* Documentation: http://netlsd.readthedocs.org