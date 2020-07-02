# Deep Morpheus
Morphological tagger for Ancient Greek and Latin using deep learning.

## Getting started
Make sure you are running Python 3.6 or higher. You can install the package from PyPI:

```shell
pip install deepmorpheus
```

To tag a `.txt` file, simply run:

```python
import deepmorpheus

deepmorpheus.tag_from_file("input.txt", "ancient-greek")
```

Or to tag a string directly, run:
```python
import deepmorpheus

deepmorpheus.tag_from_file("τὰ γὰρ πρὸ αὐτῶν καὶ τὰ ἔτι παλαίτερα σαφῶς μὲν εὑρεῖν διὰ χρόνου πλῆθος ἀδύνατα ἦν", "ancient-greek")
```