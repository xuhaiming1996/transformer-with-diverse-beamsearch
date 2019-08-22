from os import path

from setuptools import setup, find_packages



setup(
    name='page_serving_client',
    version='1.0',  # noqa
    packages=find_packages(),
    zip_safe=False,
    classifiers=(
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),

)
