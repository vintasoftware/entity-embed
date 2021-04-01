============
Installation
============

Entity Embed can be installed from PyPI with tools like ``pip``:

.. code-block:: bash

    $ pip install entity-embed

Requirements
------------

System
~~~~~~

- MacOS or Linux (tested on latest MacOS and Ubuntu via GitHub Actions)
- Entity Embed can train and run on a powerful laptop. Tested on a system with 32 GBs of RAM, RTX 2070 Mobile (8 GB VRAM), i7-10750H (12 threads). With batch sizes smaller than 32 and few field types, it's possible to train and run even with 2 GB of VRAM.

Libraries
~~~~~~~~~

- Python: >= 3.6
- `Numpy <https://numpy.org/>`_: >= 1.19.0
- `PyTorch <https://pytorch.org/>`_: >= 1.7.1
- `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_: >= 1.1.6
- `N2 <https://github.com/kakao/n2/>`_: >= 0.1.7

And others, see requirements.txt.
