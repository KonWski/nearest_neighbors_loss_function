from setuptools import setup, find_packages
setup(
    name='nearest_neighbors_loss_function',
    version='0.1.0',
    install_requires=["skfp"],
    packages=find_packages(include=['nearest_neighbors_loss_function', 'nearest_neighbors_loss_function.*'])
)