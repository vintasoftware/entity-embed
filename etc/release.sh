#!/bin/bash

# Backup previous release
if [[ (-d "dist") || (-d "build") ]]; then
    previous_release=release-backup-$(date --iso=seconds)
    mkdir $previous_release
    [[ (-d "dist") ]] && mv dist $previous_release
    [[ (-d "build") ]] && mv build $previous_release
fi

# Delete current build archives
rm -rf build dist

# Make sure you're running this inside a virtual environment
pip install -r ./requirements.txt

# Build the artifacts
python setup.py sdist bdist_wheel --universal

# This will upload the artifacts to PyPi (add your credentials to .pypirc for convenience)
twine upload dist/*
