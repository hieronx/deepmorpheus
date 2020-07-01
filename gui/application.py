from tkinter import *
from tkinter import filedialog
from const import *

class Application(Frame):
    """The window class this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.input_file = StringVar(self.master)
        self.input_file.set("No file loaded...")
        self.configure_grid()
        self.menu = self.create_menu()
        self.settings_frame, self.contents_frame, self.buttons_frame = self.create_frames()

        self.populate_settings()
        self.populate_contents()
        self.populate_buttons()

    def populate_contents(self):
        """Populates the middle frame"""
        self.contents_frame.columnconfigure(0, weight=1)
        self.contents_frame.columnconfigure(1, weight=0)
        self.contents_frame.rowconfigure(0, weight=1)
        self.contents_frame.rowconfigure(1, weight=0)
        self.contents_frame.rowconfigure(2, weight=0)

        self.input_text = Text(self.contents_frame, wrap=NONE)
        self.input_text.grid(row = 0, column = 0, sticky=NESW)

        file_label = Label(self.master, textvariable = self.input_file)
        file_label.grid(row=2, column=0, columnspan=2, sticky=NESW)

        yscroll = Scrollbar(self.contents_frame)
        yscroll.grid(row=0, column=1, sticky=NESW)
        yscroll.config(command=self.input_text.yview)

        xscroll = Scrollbar(self.contents_frame, orient=HORIZONTAL)
        xscroll.grid(row=1, column=0, sticky=NESW)
        xscroll.config(command=self.input_text.xview)

        self.input_text.config(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    def populate_buttons(self):
        """Populates the bottom frame"""
        self.buttons_frame.columnconfigure(0, weight = 1)
        self.buttons_frame.columnconfigure(1, weight = 1)
        self.buttons_frame.columnconfigure(2, weight = 1)

        # Create the start conversion button
        start_button = Button(self.buttons_frame, text='Start Analysis')
        start_button.grid(row=0, column=1, sticky=NESW)


    def populate_settings(self):
        """Populates the settings menu"""
        self.settings_frame.columnconfigure(0, weight = 1)
        self.settings_frame.columnconfigure(1, weight = 1)
        self.settings_frame.columnconfigure(2, weight = 1)

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
        settings.grid(row=0, column=0, rowspan=1, columnspan=10, sticky=NESW, padx=4, pady=4)

        contents = Frame(self.master)
        contents.grid(row=1, column=0, rowspan=1, columnspan=10, sticky=NESW, padx=4, pady=4)

        buttons = Frame(self.master)
        buttons.grid(row=2, column=0, rowspan=1, columnspan=10, sticky=NESW, padx=4, pady=4)

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
        filename = filedialog.askopenfilename()
        if filename is None or len(filename.strip()) == 0: return

        contents = ""
        try:
            with open(filename, 'r', encoding='utf8') as f:
                contents = "".join(f.readlines())
            self.input_file.set(filename)
        except:
            self.input_file.set("Could not open file: %s" % filename)
            
        self.input_text.delete('1.0', END)
        self.input_text.insert('1.0', contents)


    def exit_program(self):
        """Attempts a graceful shutdown of the Tkinter mainloop"""
        print("Attempting graceful Tkinter environment halt...")
        self.master.destroy()
