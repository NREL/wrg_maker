"""
setup.py
"""
import logging
import os
import shlex
from codecs import open
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.develop import develop

import wrg_maker

logger = logging.getLogger(__name__)

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

version = wrg_maker.__version__


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            logger.warning(f"Unable to run 'pre-commit install': {e}")

        develop.run(self)


with open("requirements.txt") as f:
    install_requires = f.readlines()

test_requires = [
    "pytest>=5.2",
]

setup(
    name="wrg_maker",
    version=version,
    description="Make .WRG files from HDF5 formatted wind resource datasets",
    long_description=readme,
    author="Michael Rossol",
    author_email="mrossol@gmail.com",
    url="https://github.com/moptis/wrg_maker",
    packages=find_packages(),
    package_dir={"wrg_maker": "wrg_maker"},
    entry_points={
        "console_scripts": [
            "wrg_maker=wrg_maker.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="wrg_maker",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    install_requires=install_requires,
    extras_require={"test": test_requires, "dev": test_requires + ["isort", "pre-commit", "black"]},
    cmdclass={"develop": PostDevelopCommand},
)
