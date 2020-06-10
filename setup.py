import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepmorpheus", # Replace with your own username
    version="0.0.1",
    author="Mees Gelein, Jeroen Offerijns",
    author_email="author@example.com",
    description="Morphological tagger for Ancient Greek using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/offerijns/deepmorpheus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
