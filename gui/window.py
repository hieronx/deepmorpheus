from tkinter import *
from tkinter import filedialog

class Window(Frame):
    """The window class that subclasses the TKinter Frame, this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=True)
        self.create_menu()

        # Example on how to place buttons in relative positions
        # button = Button(text='Click to Open File', command=self.open_file)
        # button.pack(side=RIGHT, padx=5, pady=5)

    def create_menu(self):
        """Creates the menu on the top border of the window"""
        menu = Menu(self.master)
        self.master.config(menu=menu)

        file_menu = Menu(menu)
        file_menu.add_command(label="Open File...", command=self.open_file)
        file_menu.add_command(label="Exit", command=self.exit_program)
        menu.add_cascade(label="File", menu=file_menu)

    def open_file(self):
        """Opens the file dialog window letting us point to a file to be opened as input"""
        print(filedialog.askopenfilename())

    def exit_program(self):
        """Attempts a graceful shutdown of the Tkinter mainloop"""
        print("Attempting graceful Tkinter environment halt...")
        self.master.destroy()
