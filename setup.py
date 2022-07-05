from setuptools import setup, find_packages


setup(
    name='ldsim',
    version='0.4.0',
    author='Vyacheslav Golovin',
    description='Tools for modelling laser diode DC operation',
    packages=find_packages(exclude=['tests'],),
    install_requires=[
        'scipy',
        'matplotlib',
    ],
)
