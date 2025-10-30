# app.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from analyzer import analizar_y_triplos

EJEMPLO = (
    "E$ Pedro , Juan , Luis , i , j , Sofia1;\n"
    "Pedro = 9;\n"
    "Juan = 2;\n"
    "Sofia1 = Pedro / 3 + 5;\n"
    "Juan = Sofia1 - 2;\n"
    "do{\n"
    "  Luis = Luis + 1 * 2;\n"
    "  Pedro = Pedro + 3 / 2;\n"
    "  do{ i = i + 1; } while (i < 5);\n"
    "} while (Pedro < 10 || Sofia1 > 8);\n"
)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini-Compilador Automatas II")
        self.geometry("1650x950")
        self._last_triplos = []

        # ====== Editor ======
        left = ttk.LabelFrame(self, text="Entrada del programa")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        editor_wrap = ttk.Frame(left)
        editor_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.line_nums = tk.Text(editor_wrap, width=5, padx=6, takefocus=0, border=0,
                                 background="#f3f3f3", state="disabled", wrap="none",
                                 font=("Consolas", 12))
        self.line_nums.pack(side=tk.LEFT, fill=tk.Y)

        self.txt = tk.Text(editor_wrap, wrap="none", undo=True, font=("Consolas", 12))
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        yscroll = ttk.Scrollbar(editor_wrap, orient="vertical")
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        yscroll.config(command=self._sync_yview)
        self.txt.config(yscrollcommand=lambda *a: self._on_yscroll(*a, widget=self.txt))
        self.line_nums.config(yscrollcommand=lambda *a: self._on_yscroll(*a, widget=self.line_nums))

        xscroll = ttk.Scrollbar(left, orient="horizontal", command=self.txt.xview)
        xscroll.pack(fill=tk.X, padx=6)
        self.txt.config(xscrollcommand=xscroll.set)

        self.txt.insert("1.0", EJEMPLO)

        self.txt.bind("<KeyRelease>", self._update_line_numbers)
        self.txt.bind("<MouseWheel>", self._update_line_numbers)
        self.txt.bind("<ButtonRelease-1>", self._update_line_numbers)
        self.txt.bind("<<Modified>>", self._on_modified)
        self.bind("<Configure>", self._update_line_numbers)

        # ====== Panel derecho ======
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        btns = ttk.Frame(right); btns.pack(fill=tk.X)
        ttk.Button(btns, text="Analizar (2 pasadas)", command=self.analizar).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Limpiar", command=self.limpiar).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Exportar triplos (TXT)", command=self.exportar_triplos_txt).pack(side=tk.LEFT, padx=4, pady=4)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(8, 8))

        frame_sym = ttk.Frame(self.notebook); self.notebook.add(frame_sym, text="Símbolos")
        self.tree_sym = ttk.Treeview(frame_sym, columns=("lex", "tipo"), show="headings", height=20)
        self.tree_sym.heading("lex", text="Lexema"); self.tree_sym.heading("tipo", text="Tipo de dato")
        self.tree_sym.column("lex", width=260); self.tree_sym.column("tipo", width=120)
        self.tree_sym.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        frame_err = ttk.Frame(self.notebook); self.notebook.add(frame_err, text="Errores")
        self.tree_err = ttk.Treeview(frame_err,
            columns=("token","linea","col","lex","desc"), show="headings", height=20)
        for h, w in (("token",130),("linea",60),("col",60),("lex",200),("desc",420)):
            self.tree_err.heading(h, text=h.capitalize())
            self.tree_err.column(h, width=w, anchor="e" if h in ("linea","col") else "w")
        self.tree_err.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        frame_tri = ttk.Frame(self.notebook); self.notebook.add(frame_tri, text="Triplos")
        self.tree_tri = ttk.Treeview(frame_tri, columns=("N","O","DO","DF"), show="headings", height=20)
        self.tree_tri.heading("N", text="N"); self.tree_tri.column("N", width=60, anchor="e")
        for h, w in (("O",140),("DO",220),("DF",220)):
            self.tree_tri.heading(h, text=h if h!="DO" else "D.O")
            self.tree_tri.column(h, width=w)
        self.tree_tri.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        help_lbl = ttk.Label(right, justify="left",
            text=("Autores:\n"
                  "• Roger Fernando Gonzalez Pereira.\n"
                  "• Aurora Vanessa Madera Canul.\n"
                  "• Cristian Eduardo Medina Pech.\n"))
        help_lbl.pack(fill=tk.X, pady=8)

        self._update_line_numbers()

    # ===== Helpers GUI =====
    def _on_modified(self, _=None):
        self.txt.edit_modified(False); self._update_line_numbers()

    def _sync_yview(self, *args):
        self.txt.yview(*args); self.line_nums.yview(*args)

    def _on_yscroll(self, *args, widget=None):
        if widget is self.txt: self.line_nums.yview_moveto(args[0])
        else: self.txt.yview_moveto(args[0])

    def _update_line_numbers(self, _=None):
        total = int(self.txt.index("end-1c").split(".")[0])
        nums = "\n".join(str(i) for i in range(1, total + 1))
        self.line_nums.configure(state="normal"); self.line_nums.delete("1.0", "end")
        self.line_nums.insert("1.0", nums); self.line_nums.configure(state="disabled")

    # ===== Lógica =====
    def limpiar(self):
        for t in (self.tree_sym, self.tree_err, self.tree_tri):
            t.delete(*t.get_children())
        self._last_triplos = []

    def analizar(self):
        try:
            self.limpiar()
            texto = self.txt.get("1.0", tk.END)
            tabla, errores, triplos = analizar_y_triplos(texto)
            self._last_triplos = triplos

            for lex, tipo in tabla:
                self.tree_sym.insert("", tk.END, values=(lex, tipo))
            for e in errores:
                self.tree_err.insert("", tk.END, values=(e.get('token',''), e.get('linea',''),
                                                         e.get('col',''), e.get('lex',''),
                                                         e.get('desc','')))
            for n, o, do, df in triplos:
                self.tree_tri.insert("", tk.END, values=(n, o, do, df))

            self.notebook.select(2)
            messagebox.showinfo("Éxito", f"Análisis completo.\nTriplos generados: {len(triplos)}")
        except Exception as ex:
            messagebox.showerror("Error", f"Ocurrió un error durante el análisis:\n{ex}")

    def exportar_triplos_txt(self):
        if not self._last_triplos:
            messagebox.showwarning("Sin datos", "Primero ejecuta 'Analizar' para generar triplos.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Texto", "*.txt")],
                                            title="Guardar triplos como TXT")
        if not path: return

        try:
            headers = ["N", "O", "D.O", "D.F"]
            rows = [[str(n), str(o), str(do), str(df)] for (n, o, do, df) in self._last_triplos]

            widths = [len(h) for h in headers]
            for r in rows:
                for idx, cell in enumerate(r):
                    widths[idx] = max(widths[idx], len(cell))

            def sep(): return "+" + "+".join("-"*(w+2) for w in widths) + "+\n"
            def row(vals):
                out=[]
                for i, c in enumerate(vals):
                    out.append(" " + (c.rjust(widths[i]) if i==0 else c.ljust(widths[i])) + " ")
                return "|" + "|".join(out) + "|\n"

            with open(path, "w", encoding="utf-8") as f:
                f.write(sep()); f.write(row(headers)); f.write(sep())
                for r in rows: f.write(row(r))
                f.write(sep())
            messagebox.showinfo("OK", f"Triplos exportados en:\n{path}")
        except Exception as ex:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{ex}")

if __name__ == "__main__":
    App().mainloop()
