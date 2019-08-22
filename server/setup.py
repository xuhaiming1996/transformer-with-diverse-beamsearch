from os import path

from setuptools import setup, find_packages




setup(
    name='page_serving_server',
    version="1.1",
    author='HaimingXu',
    packages=find_packages(),
    license='MIT',
    zip_safe=False,
    install_requires=[
        'flask',
        'flask-compress',
        'flask-cors',
        'flask-json',
        'numpy',
        'six',
        'pyzmq>=17.1.0',
        'GPUtil>=1.3.0',
        'termcolor>=1.1'
    ],
    extras_require={
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json', 'page-serving-client']
    },

    entry_points={
        'console_scripts': ['page-serving-start=bert_serving.server.cli:main',
                            'page-serving-benchmark=bert_serving.server.cli:benchmark',
                            'page-serving-terminate=bert_serving.server.cli:terminate'],
    },

)
