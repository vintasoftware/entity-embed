#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Flávio Juvenal da Silva Junior",
    author_email='flavio@vinta.com.br',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Transform entities like companies, products, etc. into vectors to support scalable Record Linkage / Entity Resolution using Approximate Nearest Neighbors.",
    entry_points={
        'console_scripts': [
            'entity_embed=entity_embed.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='record linkage,entity resolution,deduplication,embedding',
    name='entity-embed',
    packages=find_packages(include=['entity_embed', 'entity_embed.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/vintasoftware/entity-embed',
    version='0.0.1',
    zip_safe=False,
)
