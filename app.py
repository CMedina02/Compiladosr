import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from analyzer import analizar_y_triplos, optimizar_codigo_fuente
from codegen8086 import generar_ensamblador_8086

EJEMPLO = (
    "E$ Pedro , Juan , Luis , i , Sofia1;\n"
    "Pedro = 9;\n"
    "Juan = 2;\n"
    "Sofia1 = Pedro / 3 + 5;\n"
    "Juan = Sofia1 - 2;\n"
    "do{\n"
    "  Luis = Luis + 1 * 2;\n"
    "  Pedro = Pedro + 3 / 2;\n"
    "  do{\n"
    "    i = i + 1;\n"
    "  } while (i < 5);\n"
    "} while (Pedro < 10 || Sofia1 > 8);\n"
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini-Compilador Automatas II")
        self.geometry("1750x980")
        self._last_triplos = []

        # ==========================================================
        # ============ EDITOR IZQUIERDO: CÓDIGO OPTIMIZADO ==========
        # ==========================================================
        left = ttk.LabelFrame(self, text="Código optimizado (resultado)")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        editor_wrap = ttk.Frame(left)
        editor_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.line_nums = tk.Text(
            editor_wrap, width=5, padx=6, takefocus=0, border=0,
            background="#f3f3f3", state="disabled", wrap="none",
            font=("Consolas", 12)
        )
        self.line_nums.pack(side=tk.LEFT, fill=tk.Y)

        # Text principal: SIEMPRE muestra el CÓDIGO OPTIMIZADO
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

        # Código inicial (como ejemplo)
        self.txt.insert("1.0", EJEMPLO)

        # Eventos numeración
        self.txt.bind("<KeyRelease>", self._update_line_numbers)
        self.txt.bind("<MouseWheel>", self._update_line_numbers)
        self.txt.bind("<ButtonRelease-1>", self._update_line_numbers)
        self.txt.bind("<<Modified>>", self._on_modified)
        self.bind("<Configure>", self._update_line_numbers)

        # ==========================================================
        # ======================= PANEL DERECHO =====================
        # ==========================================================
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Analizar (2 pasadas)", command=self.analizar).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Limpiar tablas", command=self.limpiar).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Exportar triplos (TXT)", command=self.exportar_triplos_txt).pack(side=tk.LEFT, padx=4, pady=4)

        # ==================== NOTEBOOK ====================
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=6)

        # ---------- CÓDIGO ORIGINAL ----------
        frame_orig = ttk.Frame(self.notebook)
        self.notebook.add(frame_orig, text="Código de entrada (original)")

        self.txt_orig = tk.Text(frame_orig, wrap="none", font=("Consolas", 12), state="disabled")
        self.txt_orig.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        scroll_orig = ttk.Scrollbar(frame_orig, orient="vertical", command=self.txt_orig.yview)
        scroll_orig.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_orig.config(yscrollcommand=scroll_orig.set)

        # ---------- SÍMBOLOS ----------
        frame_sym = ttk.Frame(self.notebook)
        self.notebook.add(frame_sym, text="Símbolos")

        self.tree_sym = ttk.Treeview(frame_sym, columns=("lex", "tipo"), show="headings", height=20)
        self.tree_sym.heading("lex", text="Lexema")
        self.tree_sym.heading("tipo", text="Tipo")
        self.tree_sym.column("lex", width=260)
        self.tree_sym.column("tipo", width=160)
        self.tree_sym.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------- ERRORES ----------
        frame_err = ttk.Frame(self.notebook)
        self.notebook.add(frame_err, text="Errores")

        self.tree_err = ttk.Treeview(
            frame_err,
            columns=("token", "linea", "col", "lex", "desc"),
            show="headings", height=20
        )
        for h, w in (("token",120),("linea",60),("col",60),("lex",200),("desc",420)):
            self.tree_err.heading(h, text=h.capitalize())
            self.tree_err.column(h, width=w)
        self.tree_err.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------- TRIPLOS ----------
        frame_tri = ttk.Frame(self.notebook)
        self.notebook.add(frame_tri, text="Triplos")

        self.tree_tri = ttk.Treeview(
            frame_tri, columns=("N", "O", "DO", "DF"), show="headings", height=20
        )
        self.tree_tri.heading("N", text="N")
        self.tree_tri.column("N", width=60, anchor="e")
        for h, w in (("O",150),("DO",180),("DF",180)):
            self.tree_tri.heading(h, text=h if h != "DO" else "D.O")
            self.tree_tri.column(h, width=w)
        self.tree_tri.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ---------- ASM 8086 ----------
        frame_asm = ttk.Frame(self.notebook)
        self.notebook.add(frame_asm, text="ASM 8086 (Código Objeto)")

        self.txt_asm = tk.Text(frame_asm, wrap="none", font=("Consolas", 12))
        self.txt_asm.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        scroll_asm = ttk.Scrollbar(frame_asm, orient="vertical", command=self.txt_asm.yview)
        scroll_asm.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_asm.config(yscrollcommand=scroll_asm.set)

        self._update_line_numbers()

    # ==========================================================
    #                 ACTUALIZADORES DE GUI
    # ==========================================================
    def _on_modified(self, _=None):
        self.txt.edit_modified(False)
        self._update_line_numbers()

    def _sync_yview(self, *args):
        self.txt.yview(*args)
        self.line_nums.yview(*args)

    def _on_yscroll(self, *args, widget=None):
        if widget is self.txt:
            self.line_nums.yview_moveto(args[0])
        else:
            self.txt.yview_moveto(args[0])

    def _update_line_numbers(self, _=None):
        total = int(self.txt.index("end-1c").split(".")[0])
        nums = "\n".join(str(i) for i in range(1, total+1))
        self.line_nums.config(state="normal")
        self.line_nums.delete("1.0", "end")
        self.line_nums.insert("1.0", nums)
        self.line_nums.config(state="disabled")

    # ==========================================================
    #                         LÓGICA
    # ==========================================================
    def limpiar(self):
        for t in (self.tree_sym, self.tree_err, self.tree_tri):
            t.delete(*t.get_children())
        self._last_triplos = []
        self.txt_asm.delete("1.0", tk.END)

    def analizar(self):
        try:
            self.limpiar()

            # 1) Leer código que EL USUARIO escribió (entrada)
            codigo_entrada = self.txt.get("1.0", tk.END)

            # 2) Guardar el código de entrada en la pestaña "original"
            self.txt_orig.config(state="normal")
            self.txt_orig.delete("1.0", tk.END)
            self.txt_orig.insert("1.0", codigo_entrada)
            self.txt_orig.config(state="disabled")

            # 3) OPTIMIZAR primero
            codigo_opt = optimizar_codigo_fuente(codigo_entrada)

            # 4) Mostrar código OPTIMIZADO en el editor izquierdo
            self.txt.delete("1.0", tk.END)
            self.txt.insert("1.0", codigo_opt)
            self._update_line_numbers()

            # 5) Analizar SIEMPRE el código optimizado
            tabla, errores, triplos = analizar_y_triplos(codigo_opt)
            self._last_triplos = triplos

            # 6) Poblar símbolos
            for lex, tipo in tabla:
                self.tree_sym.insert("", tk.END, values=(lex, tipo))

            # 7) Poblar errores
            for e in errores:
                self.tree_err.insert(
                    "", tk.END,
                    values=(e.get("token",""), e.get("linea",""),
                            e.get("col",""), e.get("lex",""), e.get("desc",""))
                )

            # 8) Poblar triplos
            for n, o, do, df in triplos:
                self.tree_tri.insert("", tk.END, values=(n, o, do, df))

            # 9) Generar ASM 8086 A PARTIR DEL CÓDIGO OPTIMIZADO
            asm = generar_ensamblador_8086(codigo_opt)
            self.txt_asm.delete("1.0", tk.END)
            self.txt_asm.insert("1.0", asm)

            # Por comodidad, saltar a la pestaña de ASM
            self.notebook.select(4)

        except Exception as ex:
            messagebox.showerror("Error", f"Ocurrió un error:\n{ex}")

    # ==========================================================
    #              EXPORTAR TRIPLOS A TXT
    # ==========================================================
    def exportar_triplos_txt(self):
        if not self._last_triplos:
            messagebox.showwarning("Sin datos", "Primero ejecuta 'Analizar'.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt")],
            title="Guardar triplos"
        )
        if not path:
            return

        try:
            headers = ["N", "O", "D.O", "D.F"]
            rows = [[str(n), str(o), str(do), str(df)] for (n, o, do, df) in self._last_triplos]

            widths = [len(h) for h in headers]
            for r in rows:
                for idx, cell in enumerate(r):
                    widths[idx] = max(widths[idx], len(cell))

            def sep():
                return "+" + "+".join("-"*(w+2) for w in widths) + "+\n"

            def row(vals):
                out=[]
                for i,c in enumerate(vals):
                    out.append(" " + (c.rjust(widths[i]) if i==0 else c.ljust(widths[i])) + " ")
                return "|" + "|".join(out) + "|\n"

            with open(path, "w", encoding="utf-8") as f:
                f.write(sep())
                f.write(row(headers))
                f.write(sep())
                for r in rows: 
                    f.write(row(r))
                f.write(sep())

            messagebox.showinfo("OK", f"Archivo guardado en:\n{path}")

        except Exception as ex:
            messagebox.showerror("Error", f"No se pudo guardar:\n{ex}")


if __name__ == "__main__":
    App().mainloop()
