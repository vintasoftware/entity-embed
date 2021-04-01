# Entity Embed

[![PyPi version](https://img.shields.io/pypi/v/entity-embed.svg)](https://pypi.python.org/pypi/entity-embed)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-lightning)](https://pypi.org/project/pytorch-lightning/)
[![Documentation Status](https://readthedocs.org/projects/entity-embed/badge/?version=latest)](https://entity-embed.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/vintasoftware/entity-embed/branch/main/graph/badge.svg?token=4BW63JN071)](https://codecov.io/gh/vintasoftware/entity-embed)
[![License: MIT](https://img.shields.io/github/license/vintasoftware/django-react-boilerplate.svg)](LICENSE.txt)

Transform entities like companies, products, etc. into vectors to support **scalable Entity Resolution using Approximate Nearest Neighbors.**

Using Entity Embed, you can train a deep learning model to transform entities into vectors in an N-dimensional embedding space. Thanks to a contrastive loss, those vectors are organized to keep similar entities close and dissimilar entities far apart in this embedding space. Embedding entities enables [scalable ANN search](http://ann-benchmarks.com/index.html), which means finding thousands of candidate duplicate pairs of entities per second per CPU.

Entity Embed achieves Recall of ~0.99 with Pair-Entity ratio below 100 on a variety of datasets. **Entity Embed aims for high recall at the expense of precision. Therefore, this library is suited for the Blocking/Indexing stage of an Entity Resolution pipeline.**  A scalabale and noise-tolerant Blocking procedure is often the main bottleneck for performance and quality on Entity Resolution pipelines, so this library aims to solve that. Note the ANN search on embedded entities returns several candidate pairs that must be filtered to find the best matching pairs, possibly with a pairwise classifier.

Entity Embed is based on and is a special case of the [AutoBlock model described by Amazon](https://www.amazon.science/publications/autoblock-a-hands-off-blocking-framework-for-entity-matching).

**⚠️ Warning: this project is under heavy development.**

![Embedding Space Example](https://user-images.githubusercontent.com/397989/113318040-689a2d00-92e6-11eb-8373-29477d57d29e.png)

## Documentation

https://entity-embed.readthedocs.io

## Requirements

### System

- MacOS or Linux (tested on latest MacOS and Ubuntu via GitHub Actions).
- Entity Embed can train and run on a powerful laptop. Tested on a system with 32 GBs of RAM, RTX 2070 Mobile (8 GB VRAM), i7-10750H (12 threads). With batch sizes smaller than 32 and few field types, it's possible to train and run even with 2 GB of VRAM.

### Libraries

- **Python**: >= 3.6
- **[Numpy](https://numpy.org/)**: >= 1.19.0
- **[PyTorch](https://pytorch.org/)**: >= 1.7.1
- **[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)**: >= 1.1.6
- **[N2](https://github.com/kakao/n2/)**: >= 0.1.7

And others, see [requirements.txt](/requirements.txt).

## Installation

```
pip install entity-embed
```

## Examples

Run:

```
pip install -r requirements-examples.txt
```

Then check the example Jupyter Notebooks:

- Deduplication, when you have a single dirty dataset with duplicates: [notebooks/Deduplication-Example.ipynb](/notebooks/Deduplication-Example.ipynb)
- Record Linkage, when you have multiple clean datasets you need to link: [notebooks/Record-Linkage-Example.ipynb](/notebooks/Record-Linkage-Example.ipynb)

## Releases

See [CHANGELOG.md](/CHANGELOG.md).

## Credits

This project is maintained by [open-source contributors](/AUTHORS.rst) and [Vinta Software](https://www.vintasoftware.com/).

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage) project template.


## Commercial Support

[Vinta Software](https://www.vintasoftware.com/) is always looking for exciting work, so if you need any commercial support, feel free to get in touch: contact@vinta.com.br


## Citations

If you use Entity Embed in your research, please consider citing it.

BibTeX entry:

```
@software{entity-embed,
  title = {{Entity Embed}: Scalable Entity Resolution using Approximate Nearest Neighbors.},
  author = {Juvenal, Flávio and Vieira, Renato},
  url = {https://github.com/vintasoftware/entity-embed},
  version = {0.0.1},
  date = {2021-03-30},
  year = {2021}
}
```
