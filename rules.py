# rules.py
import re

#} Tipos del lenguaje (mostrar siempre como E$ / C$ / F$)
TYPE_TOKENS = ("E$", "C$", "F$")       # Entero, Cadena, Flotante
TOK_INT, TOK_STR, TOK_FLOAT = TYPE_TOKENS  # alias léxicos

# Expresiones regulares de tokens
RE_TIPO      = r'(E\$|C\$|F\$)'
RE_IDENT     = r'[A-Za-z][A-Za-z0-9]*'
RE_FLOAT     = r'\d+\.\d+'
RE_INT       = r'\d+'
RE_STR       = r'"[^"\n]*"'
RE_CHAR      = r"'[^'\n]'"

# Palabras clave simples
RE_KW        = r'\b(if|else|while|for|do|break|continue|func|return)\b'

# Palabra reservada compuesta (un solo token)
RE_SYSPRINT  = r'System\.out\.println'

# Operadores y separadores (incluye '.')
RE_OP        = r'(\+|-|\*|/|%|==|!=|<=|>=|<|>|=|,|;|\(|\)|\{|\}|!|&&|\|\||\.)'

# Patrón maestro
TOKEN_PATTERN = re.compile(
    rf'\s*('
    rf'{RE_TIPO}|'
    rf'{RE_SYSPRINT}|'
    rf'{RE_FLOAT}|{RE_INT}|{RE_STR}|{RE_CHAR}|'
    rf'{RE_KW}|'
    rf'{RE_IDENT}|'
    rf'{RE_OP}'
    rf')',
    re.IGNORECASE
)

def idx_a_line_col(texto: str, idx: int):
    """Convierte índice absoluto en (línea, columna) 1-based."""
    line = texto.count('\n', 0, idx) + 1
    last_nl = texto.rfind('\n', 0, idx)
    col = (idx + 1) if last_nl == -1 else (idx - last_nl)
    return line, col
