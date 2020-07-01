import tkinter as tk

root = tk.Tk()

button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.BOTTOM)

reset_button = tk.Button(button_frame, text='Reset')
run_button = tk.Button(button_frame, text='Run')

button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

reset_button.grid(row=0, column=0, sticky=tk.W+tk.E)
run_button.grid(row=0, column=1, sticky=tk.W+tk.E)

root.mainloop()