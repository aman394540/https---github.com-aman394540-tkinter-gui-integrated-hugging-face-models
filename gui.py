import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk  # for image preview

from .models import (
    make_text_sentiment_runner,
    make_image_classifier_runner,
)
from .oop_demo import get_explanations


class AppGUI(ttk.Frame):
    """
    UI matches the provided sample but:
      • Image results are Top-1 (closest) by default.
      • Selected image is previewed in the left pane when in Image mode.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master.title("Tkinter AI GUI")
        self.master.geometry("980x720")

        # ttk style
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", padding=(2, 2))
        style.configure("TButton", padding=(6, 4))
        style.configure("TLabelframe", padding=(10, 8))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))

        self.pack(fill=tk.BOTH, expand=True)

        # Runners (Model 1 = Text, Model 2 = Image)
        self.text_runner = make_text_sentiment_runner()
        self.image_runner = make_image_classifier_runner()

        # State
        self.input_mode = tk.StringVar(value="text")   # "text" or "image"
        self._selected_image_path = tk.StringVar(value="")
        self._preview_photo = None  # keep ref to avoid GC

        # Menus
        self._build_menubar()

        # Top row: Model selection
        self._build_model_selection_row()

        # Middle: I/O columns
        self._build_io_columns()

        # Bottom: info & OOP
        self._build_info_panel()

        # Notes
        self._build_notes()

        self.model_select.current(0)
        self._update_selected_model_info()
        self._update_input_view()

    # ------------- Menubar -------------
    def _build_menubar(self):
        menubar = tk.Menu(self.master)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        models_menu = tk.Menu(menubar, tearoff=0)
        models_menu.add_command(label="Load Model 1 (Text)", command=lambda: self._load_runner("model1"))
        models_menu.add_command(label="Load Model 2 (Image)", command=lambda: self._load_runner("model2"))
        menubar.add_cascade(label="Models", menu=models_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                "HIT137 Assignment 3\nTwo free Hugging Face models\n• Model 1: Text Sentiment (Text)\n• Model 2: Image Classification (Vision)",
            ),
        )
        menubar.add_cascade(label="Help", menu=help_menu)

        self.master.config(menu=menubar)

    # ------------- Top row -------------
    def _build_model_selection_row(self):
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=12, pady=(10, 4))

        ttk.Label(row, text="Model Selection:").pack(side=tk.LEFT, padx=(0, 8))

        self.model_select = ttk.Combobox(
            row,
            state="readonly",
            values=[
                "Model 1 — Text Sentiment (Text)",
                "Model 2 — Image Classification (Vision)",
            ],
            width=40,
        )
        self.model_select.pack(side=tk.LEFT)
        self.model_select.bind("<<ComboboxSelected>>", lambda e: self._update_selected_model_info())

        ttk.Label(row, text="  ").pack(side=tk.LEFT)  # spacer
        ttk.Button(row, text="Load Model", command=self._load_selected_model).pack(side=tk.LEFT, padx=(8, 0))

    # ------------- I/O columns -------------
    def _build_io_columns(self):
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left: User Input Section
        left = ttk.Labelframe(mid, text="User Input Section")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Mode + Browse row
        radios = ttk.Frame(left)
        radios.pack(fill=tk.X, pady=(2, 6))
        tk.Radiobutton(
            radios, text="Text", variable=self.input_mode, value="text",
            command=self._update_input_view, anchor="w"
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            radios, text="Image", variable=self.input_mode, value="image",
            command=self._update_input_view, anchor="w"
        ).pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(radios, text="Browse", command=self._pick_image).pack(side=tk.RIGHT)

        # Stack for text vs image content
        self.input_stack = ttk.Frame(left)
        self.input_stack.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # Text input container
        self.text_container = ttk.Frame(self.input_stack)
        self.input_text = tk.Text(self.text_container, height=10, wrap="word")
        self.input_text.pack(fill=tk.BOTH, expand=True)

        # Image preview container
        self.image_container = ttk.Frame(self.input_stack)
        self.preview_label = ttk.Label(self.image_container, text="No image selected", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Buttons row
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(btns, text="Run Model 1", command=self._run_model1_threaded).pack(side=tk.LEFT)
        ttk.Button(btns, text="Run Model 2", command=self._run_model2_threaded).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cl", width=4, command=self._clear_io).pack(side=tk.LEFT)

        # Right: Output section
        right = ttk.Labelframe(mid, text="Model Output Section")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        ttk.Label(right, text="Output Display:").pack(anchor="w", padx=6, pady=(2, 2))

        out_frame = ttk.Frame(right)
        out_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.output_text = tk.Text(out_frame, height=16, wrap="word")
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(out_frame, orient="vertical", command=self.output_text.yview)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=yscroll.set)

    # ------------- Info / OOP panel -------------
    def _build_info_panel(self):
        bottom = ttk.Labelframe(self, text="Model Information & Explanation")
        bottom.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 8))

        left = ttk.Frame(bottom)
        right = ttk.Frame(bottom)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # Selected Model Info
        ttk.Label(left, text="Selected Model Info:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self.info_model_name = ttk.Label(left, text="• Model Name: —")
        self.info_model_cat = ttk.Label(left, text="• Category (Text, Vision, Audio): —")
        self.info_model_desc = ttk.Label(left, text="• Short Description: —", wraplength=420, justify="left")
        self.info_model_name.pack(anchor="w", padx=4, pady=1)
        self.info_model_cat.pack(anchor="w", padx=4, pady=1)
        self.info_model_desc.pack(anchor="w", padx=4, pady=1)

        # OOP Concepts
        ttk.Label(right, text="OOP Concepts Explanation:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        oop_text = tk.Text(right, height=7, wrap="word")
        oop_text.pack(fill=tk.BOTH, expand=True, padx=4)
        oop_text.insert(
            "1.0",
            "• Where Multiple Inheritance is used: model runners inherit from BaseModelRunner + LoggingMixin.\n"
            "• Why Encapsulation was applied: private-like attributes (_pipe, _initialized) with a public property.\n"
            "• How Polymorphism & Method Overriding are shown: both runners implement run() and override describe().\n"
            "• Where Multiple Decorators are applied: @timeit and @ensure_initialized wrap load()/run().\n"
        )
        oop_text.configure(state="disabled")
        self._oop_text_widget = oop_text

    # ------------- Notes -------------
    def _build_notes(self):
        notes = ttk.Frame(self)
        notes.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Label(notes, text="Notes").pack(side=tk.LEFT)
        self.notes_entry = ttk.Entry(notes)
        self.notes_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

    # ------------- Helpers -------------
    def _update_input_view(self):
        """Switch between text input and image preview areas."""
        for child in self.input_stack.winfo_children():
            child.pack_forget()
        if self.input_mode.get() == "text":
            self.text_container.pack(fill=tk.BOTH, expand=True)
        else:
            self.image_container.pack(fill=tk.BOTH, expand=True)

    def _clear_io(self):
        self.input_text.delete("1.0", tk.END)
        self._selected_image_path.set("")
        self.preview_label.configure(text="No image selected", image="")
        self._preview_photo = None
        self._set_output("")

    def _set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def _pick_image(self):
        if self.input_mode.get() != "image":
            self.input_mode.set("image")
            self._update_input_view()
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
        )
        if not path:
            return
        self._selected_image_path.set(path)
        try:
            img = Image.open(path).convert("RGB")
            # Fit nicely in the container
            img.thumbnail((520, 300))
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _selected_runner(self):
        idx = self.model_select.current()
        return self.text_runner if idx == 0 else self.image_runner

    def _update_selected_model_info(self):
        runner = self._selected_runner()
        meta = runner.meta()
        self.info_model_name.config(text=f"• Model Name: {meta.get('name', '—')}")
        self.info_model_cat.config(text=f"• Category (Text, Vision, Audio): {meta.get('category', '—')}")
        self.info_model_desc.config(text=f"• Short Description: {meta.get('short_description', '—')}")

    def _load_selected_model(self):
        idx = self.model_select.current()
        self._load_runner("model1" if idx == 0 else "model2")

    def _load_runner(self, which: str):
        try:
            if which == "model1":
                self.text_runner.load()
            else:
                self.image_runner.load()
            messagebox.showinfo("Model", "Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------- Run (threaded) -------------
    def _run_model1_threaded(self):
        threading.Thread(target=self._run_model1, daemon=True).start()

    def _run_model2_threaded(self):
        threading.Thread(target=self._run_model2, daemon=True).start()

    def _run_model1(self):
        if self.input_mode.get() != "text":
            messagebox.showwarning("Input", "Switch input mode to Text to run Model 1.")
            return
        txt = self.input_text.get("1.0", tk.END).strip()
        if not txt:
            messagebox.showwarning("Input", "Please enter some text.")
            return
        try:
            result = self.text_runner.run(txt)
            self._set_output(self._format_text_result_top1(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_model2(self):
        if self.input_mode.get() != "image":
            messagebox.showwarning("Input", "Switch input mode to Image to run Model 2.")
            return
        path = self._selected_image_path.get().strip()
        if not path:
            messagebox.showwarning("Input", "Please browse and select an image.")
            return
        try:
            # Top-1: we ask for a single best label
            result = self.image_runner.run(path, top_k=1)
            self._set_output(self._format_image_result_top1(result))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------- Formatting (Top-1 only) -------------
    def _format_text_result_top1(self, result_dict):
        try:
            results = result_dict.get("results", [])
            if isinstance(results, list) and results:
                r = results[0]
                label = r.get("label", "?")
                score = r.get("score", 0.0)
                rt = result_dict.get("runtime_s", 0.0)
                return f"Label: {label}\nScore: {score:.4f}\nRuntime: {rt:.3f}s"
            return json.dumps(result_dict, indent=2)
        except Exception:
            return json.dumps(result_dict, indent=2)

    def _format_image_result_top1(self, result_dict):
        try:
            results = result_dict.get("results", [])
            if isinstance(results, list) and results:
                r = results[0]
                label = r.get("label", "?")
                score = r.get("score", 0.0)
                rt = result_dict.get("runtime_s", 0.0)
                return f"Top-1 Prediction: {label}\nScore: {score:.4f}\nRuntime: {rt:.3f}s"
            return json.dumps(result_dict, indent=2)
        except Exception:
            return json.dumps(result_dict, indent=2)
