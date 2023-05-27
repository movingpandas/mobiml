import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Packages that MovingPandas uses explicitly:
INSTALL_REQUIRES = [
    "matplotlib",
    "geopandas",
    "fiona",
    "rtree",
    "geopy",
]

setuptools.setup(
    name="movingml",
    version="0.1.0",
    author="Anita Graser",
    author_email="anitagraser@gmx.at",
    description="MovingML - a library for machine learning with movement data",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/movingpandas/movingml",
    packages=[
        "movingml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
)
