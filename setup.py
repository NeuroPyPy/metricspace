from setuptools import setup, find_packages

setup(
    name='metricspace',
    version='0.6.0',
    description='A python translation of code originally theorized in: Metric-space analysis of spike trains: theory, algorithms, and application Jonathan D. Victor and Keith Purpura Network 8, 127-164 (1997)',
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    author='Flynn OConnell, Jonathan D Victor',
    author_email='flynnoconnell@gmail.com, jdvicto@med.cornell.edu',
    url='https://github.com/NeuroPyPy/metricspace',
    classifiers=[
        "Programming Language :: Rust",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        "setuptools>=40.8.0",
        "wheel"
    ],
    license_files = ("LICENSE",),
)