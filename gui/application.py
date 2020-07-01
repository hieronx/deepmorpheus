from tkinter import *
from tkinter import filedialog
from const import *

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
        self.menu = self.create_menu()
        self.settings_frame, self.contents_frame, self.buttons_frame = self.create_frames()

        self.populate_settings()
        self.populate_contents()
        self.populate_buttons()

    def populate_contents(self):
        """Populates the middle frame"""
        self.contents_frame.columnconfigure(0, weight=1)
        self.contents_frame.columnconfigure(1, weight=0)
        self.contents_frame.rowconfigure(0, weight=0)
        self.contents_frame.rowconfigure(1, weight=1)
        self.contents_frame.rowconfigure(2, weight=0)

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
        """Checks if we are ready to enable/disable the start button"""
        input_text = self.input_text.get('1.0').strip()
        new_state =  DISABLED if len(input_text) == 0 else NORMAL
        self.start_button['state'] = new_state

    def register_text_field(self, textfield):
        """Registers the textfield to the scrollbars"""
        self.yscroll.config(command=textfield.yview)
        self.xscroll.config(command=textfield.xview)
        textfield.config(yscrollcommand=self.yscroll.set, xscrollcommand=self.xscroll.set)
        textfield.grid(row = 1, column = 0, sticky=NESW)

    def populate_buttons(self):
        """Populates the bottom frame"""
        self.buttons_frame.columnconfigure(0, weight = 1)
        self.buttons_frame.columnconfigure(1, weight = 1)
        self.buttons_frame.columnconfigure(2, weight = 1)
        self.buttons_frame.rowconfigure(0, weight = 1)
        self.buttons_frame.rowconfigure(1, weight = 0)

        # Create the start conversion button
        self.start_button = Button(self.buttons_frame, text='Start Analysis')
        self.start_button.grid(row=0, column=2, sticky=NESW, padx=4, pady=4)
        self.start_button['state'] = DISABLED

        # Create the toggle output buttons
        self.toggle_button = Button(self.buttons_frame, text=self.get_toggle_text(), command=self.toggle_text_mode)
        self.toggle_button.grid(row=0, column = 1, sticky=NESW, padx=4, pady=4)

    def toggle_text_mode(self):
        """Toggles the text mode from input to output and vice versa"""
        self.mode = INPUT if self.mode == OUTPUT else OUTPUT
        self.toggle_button['text'] = self.get_toggle_text()

        if self.mode == INPUT:
            self.register_text_field(self.input_text)
            self.output_text.grid_forget()
            self.filename.set(self.input_file)
        else:
            self.register_text_field(self.output_text)
            self.input_text.grid_forget()
            self.filename.set('-- CONSOLE OUTPUT --')

    def get_toggle_text(self):
        """Returns the correct output for the toggle button"""
        return 'Textfield: Show Output' if self.mode == INPUT else 'Textfield: Show Input'

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
            self.input_file = filename
            self.filename.set(self.input_file)
            self.start_button['state'] = NORMAL
        except:
            self.input_file = "Could not open file: %s" % filename
            
        self.input_text.delete('1.0', END)
        self.input_text.insert('1.0', contents)


    def exit_program(self):
        """Attempts a graceful shutdown of the Tkinter mainloop"""
        print("Attempting graceful Tkinter environment halt...")
        self.master.destroy()
