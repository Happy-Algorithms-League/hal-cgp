# encoding: utf8
import re
from setuptools import setup


def read_version():
    with open("./cgp/__version__.py") as f:
        line = f.read()
        match = re.findall(r"[0-9]+\.[0-9]+\.[0-9]+", line)
        return match[0]


def read_requirements():
    with open("./requirements.txt") as f:
        requirements = f.read()
    return requirements


def read_extra_requirements():

    extra_requirements = {}
    extra_requirements["all"] = []
    with open("./extra-requirements.txt") as f:
        for dep in f:
            req = dep.replace("\n", " ")
            extra_requirements[req] = [req]
            extra_requirements["all"].append(req)

    extra_requirements[":python_version == '3.6'"] = ["dataclasses"]
    return extra_requirements


setup(
    name="hal-cgp",
    version=read_version(),
    author="Jakob Jordan, Maximilian Schmidt",
    description=("Cartesian Genetic Programming in pure Python."),
    license="GPLv3",
    keywords="genetic programming",
    url="https://github.com/Happy-Algorithms-League/hal-cgp",
    python_requires=">=3.6, <4",
    install_requires=read_requirements(),
    extras_require=read_extra_requirements(),
    packages=["cgp", "cgp.ea", "cgp.local_search"],
    long_description=open("long_description.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Typing :: Typed",
    ],
)
