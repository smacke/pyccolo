#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import versioneer

pkg_name = 'pyccolo'


def read_file(fname):
    with open(fname, 'r', encoding='utf8') as f:
        return f.read()


history = read_file('HISTORY.rst')
requirements = read_file('requirements.txt').strip().split()

setup(
    name=pkg_name,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Stephen Macke',
    author_email='stephen.macke@gmail.com',
    description='Embedded instrumentation for Python.',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/smacke/pyccolo',
    packages=find_packages(exclude=[
        'binder',
        'docs',
        'scratchspace',
        'notebooks',
        'img',
        'test',
        'scripts',
        'versioneer.py',
    ]),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'pyc = pyccolo.__main__:main',
            'pyccolo = pyccolo.__main__:main',
        ],
    },
    license='BSD-3-Clause',
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

# python setup.py sdist bdist_wheel --universal
# twine upload dist/*
