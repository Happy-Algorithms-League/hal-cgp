# encoding: utf8
import re
from collections import defaultdict

from setuptools import setup


def _cut_version_number_from_requirement(req):
    return req.split()[0]


def read_metadata(metadata_str):
    """
    Find __"meta"__ in init file.
    """
    with open("./cgp/__init__.py", "r") as f:
        meta_match = re.search(fr"^__{metadata_str}__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError(f"Unable to find __{metadata_str}__ string.")


def read_requirements():
    requirements = []
    with open("./requirements.txt") as f:
        for req in f:
            req = req.replace("\n", " ")
            requirements.append(req)
    return requirements


def read_extra_requirements():
    extra_requirements = defaultdict(lambda: [])
    collect_key = None
    collect_keys = []
    with open("./extra-requirements.txt") as f:
        for req in f:
            req = req.replace("\n", " ")
            if req.startswith("#"):  # new block in requirements file
                collect_key = req.split(" ")[1]
                collect_keys.append(collect_key)
                continue
            # We only list extra dependencies required for using the
            # library as individual options, not dev or doc
            # dependencies
            if collect_key == "extra":
                extra_requirements[_cut_version_number_from_requirement(req)] = [req]
            extra_requirements[collect_key].append(req)
    # Collect all requirements into the 'all' key
    for key in collect_keys:
        extra_requirements["all"] += extra_requirements[key]
    extra_requirements[":python_version == '3.6'"] = ["dataclasses"]
    return extra_requirements


def read_long_description():
    with open("README.rst", "r") as f:
        descr = f.read()
    ind = descr.find(".. long-description-end")
    return descr[:ind]


setup(
    name="hal-cgp",
    version=read_metadata("version"),
    maintainer=read_metadata("maintainer"),
    author=read_metadata("author"),
    description=(read_metadata("description")),
    license=read_metadata("license"),
    keywords=["cartesian genetic programming", "evolutionary algorithm", "symbolic regression"],
    url=read_metadata("url"),
    python_requires=">=3.6, <4",
    install_requires=read_requirements(),
    extras_require=read_extra_requirements(),
    packages=["cgp", "cgp.ea", "cgp.local_search"],
    long_description=read_long_description(),
    long_description_content_type="text/x-rst",
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
