#!/usr/bin/env python

"""The setup script."""

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = [str(req) for req in parse_requirements(requirements_file.readlines())]

short_description = (
    "Transform entities like companies, products, etc. into vectors to support scalable "
    "Record Linkage / Entity Resolution using Approximate Nearest Neighbors."
)

setup(
    author="Flávio Juvenal (Vinta Software)",
    author_email="flavio@vinta.com.br",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description=short_description,
    entry_points={
        "console_scripts": [
            "entity_embed_train=entity_embed.cli:train",
            "entity_embed_predict=entity_embed.cli:predict",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="record linkage,entity resolution,deduplication,embedding",
    name="entity-embed",
    packages=find_packages(include=["entity_embed", "entity_embed.*"]),
    url="https://github.com/vintasoftware/entity-embed",
    version="0.0.5",
    zip_safe=False,
)
