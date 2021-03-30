# Entity Embed

[![PyPi version](https://img.shields.io/pypi/v/entity-embed.svg)](https://pypi.python.org/pypi/entity-embed)
[![Documentation Status](https://readthedocs.org/projects/entity-embed/badge/?version=latest)](https://entity-embed.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/vintasoftware/entity-embed/shield.svg)](https://pyup.io/repos/github/vintasoftware/entity-embed/)
[![License: MIT](https://img.shields.io/github/license/vintasoftware/django-react-boilerplate.svg)](LICENSE.txt)

Transform entities like companies, products, etc. into vectors to support scalable Record Linkage / Entity Resolution using Approximate Nearest Neighbors.

**⚠️ Warning: this project is under heavy development.**

## Documentation

https://entity-embed.readthedocs.io

## Requirements

- **Python**: >= 3.6
- **[Numpy](https://numpy.org/)**: >= 1.19.0
- **[PyTorch](https://pytorch.org/)**: >= 1.7.1
- **[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)**: >= 1.1.6
- **[N2](https://github.com/kakao/n2/)**: >= 0.1.7

And others, see [requirements.txt](/requirements.txt)

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
  title = {{Entity Embed}: Transform entities like companies, products, etc. into vectors.},
  author = {Juvenal, Flávio and Vieira, Renato},
  url = {https://github.com/vintasoftware/entity-embed},
  version = {0.0.1},
  date = {2021-03-30},
  year = {2021}
}
```
