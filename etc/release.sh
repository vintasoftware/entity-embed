#!/bin/bash

# Backup previous release
if [[ (-d "dist") || (-d "build") ||  (-d "release") ]]; then
    previous_release=release-backup-$(date --iso=seconds)
    mkdir $previous_release
    [[ (-d "dist") ]] && mv dist $previous_release
    [[ (-d "build") ]] && mv build $previous_release
    [[ (-d "release") ]] && mv release $previous_release
fi

# Delete current build archives
rm -rf build dist
mkdir release

# Make sure you're running this inside a virtual environment
pip install -r ./requirements.txt

# Build wheel
python setup.py sdist bdist_wheel

# Build tarball
python setup.py sdist --formats=xztar

# Build zip (for windows)
python setup.py sdist --formats=zip

mv dist release/pypi
