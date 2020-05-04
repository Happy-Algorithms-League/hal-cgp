# encoding: utf8
from setuptools import setup


def read_requirements():
    with open("./requirements.txt") as f:
        requirements = f.read()
    return requirements


def read_extra_requirements():

    extra_requirements = {}
    extra_requirements["all"] = []
    with open("./extra-requirements.txt") as f:
        for l in f:
            req = l.replace("\n", " ")
            extra_requirements[req] = [req]
            extra_requirements["all"].append(req)

    return extra_requirements


setup(
    name="python-gp",
    version="0.1",
    author="Jakob Jordan, Maximilian Schmidt",
    author_email="jakobjordan@posteo.de",
    description=("Cartesian Genetic Programming in Python."),
    license="GPLv3",
    keywords="genetic programming",
    url="https://github.com/jakobj/python-gp",
    python_requires=">=3.6, <4",
    install_requires=read_requirements(),
    extras_require=read_extra_requirements(),
    packages=["gp", "gp.ea", "gp.local_search"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Utilities",
    ],
)
