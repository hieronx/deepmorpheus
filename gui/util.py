"""
Contains multiple useful utility functions, this prevents lots of code doubling
as well as just being plain useful and not necessarily tied to a specific class
"""
from tkinter import filedialog

def configure_rows(tkinter_object, weights):
    """Configures multiple rows in a simple oneliner"""
    for i, wght in enumerate(weights):
        tkinter_object.rowconfigure(i, weight=wght)

def configure_cols(tkinter_object, weights):
    """Configures multiple columns in a simple oneliner"""
    for i, wght in enumerate(weights):
        tkinter_object.columnconfigure(i, weight=wght)

def read_file_as_str(url):
    """Tries to read the provided url as a utf8 encoded text file. Returns None if it fails."""
    if url is None or len(url.strip()) == 0: return None
    try:
        with open(url, 'r', encoding='utf8') as f:
            return "".join(f.readlines())
    except:
        return None

def get_input_filename():
    """Uses tkinter filedialog to get a filename for an opening file"""
    name = filedialog.askopenfilename()
    if filename is None or len(filename.strip()) == 0: return None
    else: return name

def text_is_empty(text):
    """Returns if the provided instance of tkinter.Text is empty"""
    return len(text.get('1.0').strip()) > 0
