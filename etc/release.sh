#!/bin/bash

function backup_release {
    if [[ (-d "dist") || (-d "build") ||  (-d "release") ]]; then
        previous_release=release-backup-$(date --iso=seconds)
        mkdir $previous_release
        [[ (-d "dist") ]] && mv dist $previous_release
        [[ (-d "build") ]] && mv build $previous_release
        [[ (-d "release") ]] && mv release $previous_release
    fi
}

function clean_build {
    rm -rf build dist
}

backup_release
clean_build
mkdir release

pip install -r ./requirements.txt

python setup.py sdist bdist_wheel

mv dist release/pypi
