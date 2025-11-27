# rules.py (ACTUALIZADO CON PRIORIDAD DE OPERADORES)
import re

# Tipos del lenguaje (mostrar siempre como E$ / C$ / F$)
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
RE_KW_SINGLE = r'\b(if|else|while|for|do|break|continue|func|return)\b'

# Palabra reservada compuesta (un solo token)
RE_SYSPRINT  = r'System\.out\.println'

# Operadores de 2 caracteres (RELACIONALES y LÓGICOS)
RE_OP2       = r'(==|!=|<=|>=|&&|\|\|)'
# Operadores de 1 caracter (ARITMÉTICOS, ASIGNACIÓN, SEPARADORES, etc.)
RE_OP1       = r'(\+|-|\*|/|%|<|>|=|;|\(|\)|\{|\}|!|,|\.)'

# Patrón maestro (Orden de prioridad: SYSPRINT > OP2 > Literales > KW > IDENT > OP1)
TOKEN_PATTERN = re.compile(
    rf'\s*('
    rf'{RE_SYSPRINT}|'
    rf'{RE_OP2}|'
    rf'{RE_TIPO}|'
    rf'{RE_FLOAT}|{RE_INT}|{RE_STR}|{RE_CHAR}|'
    rf'{RE_KW_SINGLE}|'
    rf'{RE_IDENT}|'
    rf'{RE_OP1}'
    rf')',
    re.IGNORECASE
)

def idx_a_line_col(texto: str, idx: int):
    """Convierte índice absoluto en (línea, columna) 1-based."""
    line = texto.count('\n', 0, idx) + 1
    last_nl = texto.rfind('\n', 0, idx)
    col = (idx + 1) if last_nl == -1 else (idx - last_nl)
    return line, col