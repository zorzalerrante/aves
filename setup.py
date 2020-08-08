import io
import os
import re

from setuptools import find_packages
from setuptools import setup

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(["setup.cfg"])
metadata = dict(conf.items("metadata"))


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


# Metadata
PACKAGENAME = metadata.get("package_name", "aves")
VERSION = metadata.get("version", "0.1.0")
URL = metadata.get("url", "https://github.com/carnby/aves")
LICENSE = metadata.get("license", "MIT")
AUTHOR = metadata.get("author_name", "Eduardo Graells-Garrido")
AUTHOR_EMAIL = metadata.get("author_email", "eduardo.graells@bsc.es")
DESCRIPTION = metadata.get("description", "Module to support my Information Visualization course.")


setup(
    name=PACKAGENAME,
    version=VERSION,
    url=URL,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read("README.md"),
    packages=find_packages(where="src", exclude=("tests",)),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
