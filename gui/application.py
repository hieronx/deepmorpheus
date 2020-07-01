from tkinter import *
from tkinter import filedialog
from const import *

class Application(Frame):
    """The window class this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.configure_grid()
        self.menu = self.create_menu()
        self.settings_frame, self.contents_frame, self.buttons_frame = self.create_frames()

        self.populate_settings()

    def populate_settings(self):
        """Populates the settings menu"""
        for i in range(3): self.settings_frame.columnconfigure(i, weight = 1)

        # Create the open file button
        open_file_button = Button(self.settings_frame, command=self.open_file, text='Open File...')
        open_file_button.grid(row=0, column=0, sticky=NESW)

        # First set up the TKVar that tracks the Dropdown input
        self.language = StringVar(self.master)
        options = ['Ancient Greek', 'Latin']
        self.language.set(options[0])
        self.language.trace('w', self.change_dropdown)

        # Now make the label and the dropdown
        Label(self.settings_frame, text='Input Language: ').grid(row=0, column=1, sticky=NES)
        dropdown = OptionMenu(self.settings_frame, self.language, *options)
        dropdown.grid(row=0, column=2, sticky=NESW)
        
    def create_frames(self):
        """Creates the 3 frames that make up the application"""
        settings = Frame(self.master)
        settings.grid(row=0, column=0, rowspan=1, columnspan=10, sticky=NESW)

        contents = Frame(self.master, bg='blue')
        contents.grid(row=1, column=0, rowspan=1, columnspan=10, sticky=NESW)

        buttons = Frame(self.master, bg='green')
        buttons.grid(row=2, column=0, rowspan=1, columnspan=10, sticky=NESW)

        return settings, contents, buttons

    def configure_grid(self):
        """Handles the necessary setup for the grid"""
        self.grid()
        for col in range(10):
            self.master.columnconfigure(col, weight=1)
        self.master.rowconfigure(0, weight=0)
        self.master.rowconfigure(1, weight=1)
        self.master.rowconfigure(2, weight=0)

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

        return menu

    def open_file(self):
        """Opens the file dialog window letting us point to a file to be opened as input"""
        print(filedialog.askopenfilename())

    def exit_program(self):
        """Attempts a graceful shutdown of the Tkinter mainloop"""
        print("Attempting graceful Tkinter environment halt...")
        self.master.destroy()
