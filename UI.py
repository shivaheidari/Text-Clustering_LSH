from tkinter import *
import tkinter
from tkinter import filedialog

def directoryBox(self, title=None, dirName=None):
    self.topLevel.update_idletasks()
    options = {}
    options['initialdir'] = dirName
    options['title'] = title
    options['mustexist'] = False
    fileName = filedialog.askdirectory(**options)
    if fileName == "":
        return None
    else:
        return fileName


