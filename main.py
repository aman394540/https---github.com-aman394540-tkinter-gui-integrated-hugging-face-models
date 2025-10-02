import tkinter as tk
from .gui import AppGUI

def main():
    root = tk.Tk()
    AppGUI(root)
    root.geometry("900x700")
    root.mainloop()

if __name__ == "__main__":
    main()
