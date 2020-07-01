from tkinter import *

class Window(Frame):
    """The window class that subclasses the TKinter Frame, this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master