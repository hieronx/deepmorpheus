from tkinter import *
from window import Window
from const import WINDOW_SIZE, WINDOW_TITLE

def setup():
    """Starts running the GUI application"""
    print("Starting Tkinter version: %s" % TkVersion)
    root = Tk()
    root.option_add("*tearOff", False)
    root.wm_title(WINDOW_TITLE)
    root.geometry(WINDOW_SIZE)
    app = Window(root)
    print("Initialized window, starting mainloop...")
    root.mainloop()

if __name__ == '__main__':
    """Checks if this is the main script that is running, if so start the setup"""
    setup()