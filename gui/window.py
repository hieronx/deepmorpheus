from tkinter import *
from tkinter import filedialog

class Window(Frame):
    """The window class that subclasses the TKinter Frame, this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.grid(column=0, row=0, sticky='nsew', padx=8, pady=8)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.open_button()
        self.create_dropdown()
        self.create_menu()

    def open_button(self):
        """Creates the "open file" button"""
        open_file_button = Button(self.master, command=self.open_file, text='Open File...')
        open_file_button.grid(row=0, column=0, padx=8, pady=8)

    def create_dropdown(self):
        """Creates the dropdown menu that shows our language selection"""
        self.language = StringVar(self.master)
        options = ['Ancient Greek', 'Latin']
        self.language.set(options[0])
        self.language.trace('w', self.change_dropdown)

        Label(self.master, text='Input Language: ').grid(row=0, column=2)
        dropdown = OptionMenu(self.master, self.language, *options)
        dropdown.grid(row=0, column=3, padx=8, pady=8)

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
