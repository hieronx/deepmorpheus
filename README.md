# Deep Morpheus
Morphological tagger for Ancient Greek and Latin using deep learning.

## Installation
Make sure you are running Python 3.6 or higher, and that you have already installed PyTorch >= 1.3.0. You can install the package from PyPI:

```shell
pip install deepmorpheus
```

## Usage
Import the library:

```python
import deepmorpheus
```

To tag a `.txt` file::

```python

deepmorpheus.tag_from_file("input.txt", "ancient-greek")
```

To tag a string directly:
```python
deepmorpheus.tag_from_lines("τὰ γὰρ πρὸ αὐτῶν καὶ τὰ ἔτι παλαίτερα σαφῶς μὲν εὑρεῖν διὰ χρόνου πλῆθος ἀδύνατα ἦν", "ancient-greek")
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
