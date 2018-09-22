PySPPL
======

.. toctree::
   :glob:
   :maxdepth: 4
   :caption: Contents:

The Pyfo Language
~~~~~~~~~~~~~~~~~~~~~


`Pyfo` is built on top of a mathematically robust compiler for first-order graphical models.
Models written in `pyfo` are checked by the simple probabilistic programming language (SPPL) to ensure that the model
written is a valid model.

.. automodule:: pyfo.pyppl

SPPL
~~~~


base model
----------
.. autoclass:: pyfo.pyppl.ppl_base_model.base_model
    :members:
    :undoc-members:
    :show-inheritance:

Distribtuion Types in SPPL
--------------------------

.. autoclass:: pyfo.pyppl.distributions.DistributionType
    :members:
    :undoc-members:
    :show-inheritance:


Distributions in SPPL
---------------------

.. autoclass:: pyfo.pyppl.distributions.Distribution
    :members:
    :undoc-members:
    :show-inheritance:


Backend
~~~~~~~

Computational Graph
-------------------

.. autoclass:: pyfo.pyppl.backend.ppl_graph_codegen.GraphCodeGenerator
    :undoc-members:
    :show-inheritance:

Graph Factory
-------------

.. autoclass:: pyfo.pyppl.backend.ppl_graph_factory._ConditionCollector
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyfo.pyppl.backend.ppl_graph_factory.GraphFactory
    :undoc-members:
    :show-inheritance:

