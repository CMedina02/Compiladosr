import re
import tkinter as tk
from tkinter import ttk, messagebox

# ----------------------------
# Reglas léxicas (expresiones regulares)
# ----------------------------
RE_TIPO = r'(E\$|C\$|F\$)'  # E$: entero, C$: cadena, F$: flotante
RE_IDENT = r'[A-Za-z][A-Za-z0-9]*'  # Identificadores: letra seguido de letras/dígitos
RE_FLOAT = r'\d+\.\d+'              # Reales (al menos 1 dígito . 1 dígito)
RE_INT = r'\d+'                     # Enteros
RE_STR = r'\"[^"\n]*\"|\'.*?\''     # Cadenas "..." o '...'

# Operadores y signos que aceptamos (para tokenizar; no entran a la tabla con tipo)
RE_OP = r'(\+|-|\*|/|%|==|!=|<=|>=|<|>|=|\(|\)|\{|\}|\[|\]|,|;|\.)'
# Palabras clave (opcionales) soportadas
RE_KW = r'\b(if|else|while|for|do)\b'

# Mapeo de tipo de declaración -> etiqueta de tipo humana
TIPO_HUMANO = {
    'E$': 'Entero',
    'C$': 'Cadena',
    'F$': 'Flotante'
}

# ----------------------------
# Funciones de "fases" simples
# ----------------------------

def analizar_lexico(texto):
    """
    Tokeniza de forma simple. Devuelve una lista de tuplas (tipo, lexema).
    Tipos posibles: TIPO, IDENT, INT, FLOAT, STR, KW, OP, DESCONOCIDO
    """
    tokens = []
    # Patrón global que captura en orden de preferencia
    patron = re.compile(
        rf'\s*('
        rf'{RE_TIPO}'              # 1: tipo
        rf'|{RE_FLOAT}'            # 2: float
        rf'|{RE_INT}'              # 3: int
        rf'|{RE_STR}'              # 4: string
        rf'|{RE_KW}'               # 5: keyword
        rf'|{RE_IDENT}'            # 6: identificador
        rf'|{RE_OP}'               # 7: operador/signo
        rf')',
        re.IGNORECASE
    )

    i = 0
    while i < len(texto):
        m = patron.match(texto, i)
        if not m:
            # Carácter inesperado ⇒ token DESCONOCIDO de 1 char
            tokens.append(('DESCONOCIDO', texto[i]))
            i += 1
            continue

        lex = m.group(1)
        i = m.end()

        # Clasificamos el lexema por prioridad
        if re.fullmatch(RE_TIPO, lex):
            tokens.append(('TIPO', lex))
        elif re.fullmatch(RE_FLOAT, lex):
            tokens.append(('FLOAT', lex))
        elif re.fullmatch(RE_INT, lex):
            tokens.append(('INT', lex))
        elif re.fullmatch(RE_STR, lex):
            tokens.append(('STR', lex))
        elif re.fullmatch(RE_KW, lex, flags=re.IGNORECASE):
            tokens.append(('KW', lex))
        elif re.fullmatch(RE_IDENT, lex):
            tokens.append(('IDENT', lex))
        else:
            tokens.append(('OP', lex))  # operadores, signos, etc.

    return tokens


def extraer_declaraciones(texto):
    """
    Busca declaraciones del estilo:
        E$ id1, id2, id3;
        C$ nombre;
        F$ x, y;
    Devuelve dict {identificador: tipo_humano}
    """
    simbolos = {}
    # Regla: tipo + lista de identificadores separados por comas + ;
    patron_decl = re.compile(
        rf'\b({RE_TIPO})\s+({RE_IDENT}(?:\s*,\s*{RE_IDENT})*)\s*;',
        re.IGNORECASE
    )
    for m in patron_decl.finditer(texto):
        tipo = m.group(1)
        lista_ids = re.findall(RE_IDENT, m.group(2))
        for ident in lista_ids:
            simbolos[ident] = TIPO_HUMANO[tipo.upper()]
    return simbolos


def construir_tabla_simbolos(texto):
    """
    Aplica "fases":
      - léxico: tokeniza
      - sintáctico/semántico (muy simple): 
          * recoge declaraciones y asigna tipos a IDs declarados
          * tipa constantes por clase (int/float/str)
      - arma tabla de símbolos sin duplicados.
    Retorna lista de filas: [(lexema, tipo_o_vacio), ...]
    """
    tokens = analizar_lexico(texto)
    declarados = extraer_declaraciones(texto)

    # Diccionario de salida preservando orden de aparición
    tabla = {}

    for tk, lex in tokens:
        if tk == 'IDENT':
            # Si fue declarado, asignamos su tipo; si no, tipo vacío
            tipo = declarados.get(lex, '')
            if lex not in tabla:
                tabla[lex] = tipo

        elif tk in ('INT', 'FLOAT', 'STR'):
            # Constantes se tipan por su clase
            if tk == 'INT':
                tipo = 'Entero'
            elif tk == 'FLOAT':
                tipo = 'Flotante'
            else:
                tipo = 'Cadena'
            if lex not in tabla:
                tabla[lex] = tipo

        # Otros tokens (operadores, keywords) no se agregan a la tabla
        # porque no llevan tipo en la tabla de símbolos solicitada.

    # Convertimos a lista de filas
    return [(lex, tabla[lex]) for lex in tabla]


# ----------------------------
# Interfaz gráfica
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mini-Compilador (demo de fases y Tabla de Símbolos)")
        self.geometry("980x620")

        # --- Entrada (editor simple) ---
        frm_in = ttk.LabelFrame(self, text="Entrada del programa")
        frm_in.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.txt = tk.Text(frm_in, wrap="word", font=("Consolas", 12))
        self.txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Texto de ejemplo (puedes borrarlo)
        ejemplo = (
            "E$ ITA5, ITA67, ITA87;\n"
            "C$ nombre;\n"
            "F$ valor, x1;\n"
            "ITA87 = 123 * x1;\n"
            "if (ITA5 > 0) { nombre = \"hola\"; }\n"
            "while (ITA5 != 0) { ITA5 = ITA5 - 1; }\n"
        )
        self.txt.insert("1.0", ejemplo)

        # --- Panel derecho: botones + tabla ---
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=8, pady=8)

        # Botones
        btns = ttk.Frame(right)
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Analizar", command=self.analizar).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Limpiar", command=self.limpiar).pack(side=tk.LEFT, padx=4, pady=4)

        # Tabla de símbolos
        frm_tab = ttk.LabelFrame(right, text="Tabla de símbolos")
        frm_tab.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.tree = ttk.Treeview(frm_tab, columns=("lex", "tipo"), show="headings", height=24)
        self.tree.heading("lex", text="Lexema")
        self.tree.heading("tipo", text="Tipo de dato")
        self.tree.column("lex", width=240)
        self.tree.column("tipo", width=160)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Pie: guía rápida de la gramática aceptada
        ayuda = ttk.Label(
            right,
            justify="left",
            text=(
                "Reglas aceptadas (demo):\n"
                "• Declaración:  E$|C$|F$  id (, id)* ;\n"
                "• Identificador: [A-Za-z][A-Za-z0-9]*\n"
                "• Constantes: 123   3.14   \"texto\"  'texto'\n"
                "• Otras sentencias (if/else/while/for/do, asignaciones, aritmética)\n"
                "  se tokenizan, pero la tabla sólo tipa IDs/constantes."
            )
        )
        ayuda.pack(fill=tk.X, pady=8)

    def limpiar(self):
        self.tree.delete(*self.tree.get_children())

    def analizar(self):
        try:
            self.limpiar()
            texto = self.txt.get("1.0", tk.END)
            tabla = construir_tabla_simbolos(texto)
            if not tabla:
                messagebox.showinfo("Análisis", "No se encontraron identificadores ni constantes.")
                return
            for lex, tipo in tabla:
                self.tree.insert("", tk.END, values=(lex, tipo))
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error durante el análisis:\n{e}")


if __name__ == "__main__":
    App().mainloop()
