# rules.py
# Reglas y patrones de tokens
import re

TOKEN_SPEC = [
    ("WS",      r"[ \t]+"),
    ("NEWLINE", r"\r?\n"),
    ("COMMENT", r"//[^\n]*"),
    ("STR",     r"\"([^\"\\]|\\.)*\""),
    ("FLOAT",   r"\d+\.\d+"),
    ("INT",     r"\d+"),
    ("OP",      r"==|!=|<=|>=|\|\||&&|[+\-*/%<>]=?|="),
    ("PUNC",    r"[(){};,]"),
    ("TYPE",    r"(E\$|F\$|C\$)"),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
]

MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))
KEYWORDS = {"do", "while"}
