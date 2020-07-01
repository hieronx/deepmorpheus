from tkinter import *
from application import Application
from const import WINDOW_SIZE, WINDOW_TITLE

def setup():
    """Starts running the GUI application"""
    print("Starting Tkinter version: %s" % TkVersion)
    root = Tk()
    root.option_add("*tearOff", False)
    root.wm_title(WINDOW_TITLE)
    root.geometry(WINDOW_SIZE)
    app = Application(root)
    print("Initialized window, starting mainloop...")
    app.mainloop()

if __name__ == '__main__':
    """Checks if this is the main script that is running, if so start the setup"""
    setup()