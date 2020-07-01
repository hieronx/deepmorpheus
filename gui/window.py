from tkinter import *
from tkinter import filedialog

class Window(Frame):
    """The window class that subclasses the TKinter Frame, this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.grid(column=0, row=0, sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.create_dropdown()
        self.create_menu()

    def create_dropdown(self):
        """Creates the dropdown menu that shows our language selection"""
        self.language = StringVar(self.master)
        options = ['Ancient Greek', 'Latin']
        self.language.set(options[0])
        self.language.trace('w', self.change_dropdown)

        Label(self.master, text='Input Language: ').grid(row=0, column=0)
        dropdown = OptionMenu(self.master, self.language, *options)
        dropdown.grid(row=0, column=1)

    def change_dropdown(self, *args):
        """Called whenever the dropdown changes language"""
        print(self.language.get())

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
