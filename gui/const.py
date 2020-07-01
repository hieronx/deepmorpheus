"""
This file contains all the constants used in development of the Deepmorpheus GUI
"""
import tkinter as tk

APPLICATION_NAME = "Deepmorpheus"
APPLICATION_VERSION = "v0.1.0"
WINDOW_TITLE = "%s - %s" % (APPLICATION_NAME, APPLICATION_VERSION)
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 450
WINDOW_SIZE = "%dx%d" % (WINDOW_WIDTH, WINDOW_HEIGHT)
INPUT = 0
OUTPUT = 1
NS = tk.N + tk.S
NE = tk.E + tk.N
NW = tk.W + tk.N
EW = tk.E + tk.W
SW = tk.S + tk.W
SE = tk.S + tk.E
NES = NE + tk.S
NWS = NW + tk.S
SEW = SE + tk.W
NEW = NE + tk.W
NESW = NS + EW