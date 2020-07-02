import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepmorpheus",
    version="0.2.0",
    author="Mees Gelein, Jeroen Offerijns",
    description="Morphological tagger for Ancient Greek using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/offerijns/deepmorpheus",
    packages=["deepmorpheus"],
    install_requires=["torch==1.5.0", "pyconll==2.2.1", "requests==2.20.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
