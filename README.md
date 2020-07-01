# Deep Morpheus
Trying to parse Ancient Greek texts using state of the art neural network techniques.

This packages requires Python 3.6+, as well as PyTorch 1.0+. To get started, install the package from PyPI:
```
pip install deepmorpheus
```

Then, you can tag the contents of a text file like this:
```
deepmorpheus.tag_from_file(filepath, "ancient-greek")
```