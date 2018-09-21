Distributions
=============

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Contents:

PyTorch Distributions
~~~~~~~~~~~~~~~~~~~~~

Most distributions in Pyro are thin wrappers around PyTorch distributions.
For details on the PyTorch distribution interface, see
:class:`torch.distributions.distribution.Distribution`.
For differences between the Pfro and PyTorch interfaces, see
:class:`~pyfo.distributions.distribution_pyro.Distribution`.

.. automodule:: pyfo.distributions

Pyfo Distributions
~~~~~~~~~~~~~~~~~~


Exponential
-----------
.. autoclass:: pyfo.distributions.Exponential
    :members:
    :undoc-members:
    :show-inheritance:

Binomial
--------

.. autoclass:: pyfo.distributions.Binomial
    :members:
    :undoc-members:
    :show-inheritance:

