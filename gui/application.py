from tkinter import *
from tkinter import filedialog
from const import *
from util import *

class Application(Frame):
    """The window class this holds the entire program"""

    def __init__(self, master=None):
        """Creates the new window with a reference to its master"""
        Frame.__init__(self, master)
        self.master = master
        self.mode = INPUT
        self.filename = StringVar(self.master)
        self.input_file = "No file loaded..."
        self.filename.set(self.input_file)

        self.configure_grid()
        self.master.config(menu=self.create_menu())
        self.settings_frame, self.contents_frame, self.buttons_frame = self.create_frames()

        self.populate_settings()
        self.populate_contents()
        self.populate_buttons()

    def populate_contents(self):
        """Populates the middle frame"""
        configure_cols(self.contents_frame, (1, 0))
        configure_rows(self.contents_frame, (0, 1, 0))

        self.file_label = Label(self.contents_frame, textvariable = self.filename)
        self.file_label.grid(row=0, column=0, columnspan=2, sticky=NESW)

        self.input_text = Text(self.contents_frame, wrap=NONE)
        self.input_text.bind_all('<Key>', self.check_input)
        self.output_text = Text(self.contents_frame, wrap=NONE, bg='black', fg='white')
        
        self.yscroll = Scrollbar(self.contents_frame)
        self.yscroll.grid(row=1, column=1, sticky=NESW)

        self.xscroll = Scrollbar(self.contents_frame, orient=HORIZONTAL)
        self.xscroll.grid(row=2, column=0, sticky=NESW)
        self.register_text_field(self.input_text)

    def check_input(self, *args):
        """Checks if we are ready to enable/disable the start analysis button"""
        self.start_button['state'] = DISABLED if text_is_empty(self.input_text) == 0 else NORMAL

    def register_text_field(self, textfield):
        """Registers the textfield to the scrollbars"""
        self.yscroll.config(command=textfield.yview)
        self.xscroll.config(command=textfield.xview)
        textfield.config(yscrollcommand=self.yscroll.set, xscrollcommand=self.xscroll.set)
        textfield.grid(row = 1, column = 0, sticky=NESW)

    def populate_buttons(self):
        """Populates the bottom frame"""
        configure_cols(self.buttons_frame, (1, 1, 1))
        configure_rows(self.buttons_frame, (1, 0))

        # Create the start conversion button
        self.start_button = Button(self.buttons_frame, text='Start Analysis')
        self.start_button.grid(row=0, column=2, sticky=NESW, padx=4, pady=4)
        self.check_input()

        # Create the toggle output buttons
        self.toggle_button = Button(self.buttons_frame, text='Textfield: Show Output', command=self.toggle_text_mode)
        self.toggle_button.grid(row=0, column = 1, sticky=NESW, padx=4, pady=4)

    def toggle_text_mode(self):
        """Toggles the text mode from input to output and vice versa"""
        self.mode = INPUT if self.mode == OUTPUT else OUTPUT
        self.toggle_button['text'] =  'Textfield: Show Output' if self.mode == INPUT else 'Textfield: Show Input'

        self.register_text_field(self.input_text if self.mode == INPUT else self.output_text)
        (self.output_text if self.mode == INPUT else self.input_text).grid_forget()
        self.filename.set(self.input_file if self.mode == INPUT else '-- CONSOLE OUTPUT -- ')

    def populate_settings(self):
        """Populates the settings menu"""
        configure_cols(self.settings_frame, (1, 1, 1))

        open_file_button = Button(self.settings_frame, command=self.open_file, text='Open File...')
        open_file_button.grid(row=0, column=0, sticky=NESW)

        self.language = StringVar(self.master)
        options = ['Ancient Greek', 'Latin']
        self.language.set(options[0])
        self.language.trace('w', self.change_dropdown)

        Label(self.settings_frame, text='Input Language: ').grid(row=0, column=1, sticky=NES)
        dropdown = OptionMenu(self.settings_frame, self.language, *options)
        dropdown.grid(row=0, column=2, sticky=NESW)
        
    def create_frames(self):
        """Creates the 3 frames that make up the application"""
        frames = [Frame(self.master), Frame(self.master), Frame(self.master)]
        for i, frame in enumerate(frames):
            frame.grid(row=i, column=0, rowspan=1, columnspan=10, sticky=NESW, padx=4, pady=4)
        return frames

    def configure_grid(self):
        """Handles the necessary setup for the grid"""
        self.grid()
        self.master.columnconfigure(0, weight=1)
        configure_rows(self.master, (0, 1, 0))

    def change_dropdown(self, *args):
        """Called whenever the dropdown changes language"""
        print(self.language.get())

    def create_menu(self):
        """Creates the menu on the top border of the window"""
        menu = Menu(self.master)
        file_menu = Menu(menu)
        file_menu.add_command(label="Open File...", command=self.open_file)
        file_menu.add_command(label="Exit", command=self.master.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        return menu

    def open_file(self):
        """Opens the file dialog window letting us point to a file to be opened as input"""
        filename = get_input_filename()
        contents = read_file_as_str(filename)
        if contents is None:
            self.input_file = "Could not open file: %s" % filename
            set_text(self.input_text, 'ERROR')
        else: 
            self.input_file = filename
            self.filename.set(self.input_file)
            self.start_button['state'] = NORMAL
            set_text(self.input_text, contents)
