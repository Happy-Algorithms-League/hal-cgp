# encoding: utf8
from setuptools import setup

setup(
    name='python-gp',
    version='0.1',
    author='Jakob Jordan, Maximilian Schmidt',
    author_email='jakobjordan@posteo.de',
    description=('Cartesian genetic programming in Python.'),
    license='GPLv3',
    keywords='genetic programming',
    url='https://github.com/jakobj/python-gp',
    python_requires='>=3.6, <4',
    install_requires=['sympy', 'torch', 'numpy', ],
    packages=['gp', 'gp.ea'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
)
