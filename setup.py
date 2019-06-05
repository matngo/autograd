from setuptools import setup, find_packages

NAME = 'autograd'
DESCRIPTION = 'A homemade autograd library'
REQUIRED = ["numpy>=1.16.4"]
EXTRAS_REQUIRED = {
    "dev": ["ipython"],
    "tests": ["pytest"]
}

setup(
    name=NAME,
    description=DESCRIPTION,
    install_requires=REQUIRED,
    extras_require=EXTRAS_REQUIRED,
    packages=find_packages()
)
