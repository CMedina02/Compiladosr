# codegen8086.py
# -----------------------------------------
# Generador de código ensamblador 8086
# directo desde el CÓDIGO OPTIMIZADO.
#
# - NO usa temporales de triplos.
# - Usa AX como acumulador y BX como
#   segundo operando.
# - Traduce:
#     * Asignaciones con +, -, *, /, % y ()
#     * do { ... } while (cond_relacional);
# -----------------------------------------

from typing import List, Tuple

# ============================================================
# Utilidades generales
# ============================================================

def _es_entero(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _op_txt(op) -> str:
    """Normaliza operando a texto."""
    if op is None:
        return ""
    return str(op).strip()


def _linea_asm(etq: str, mnem: str, oper: str = "") -> str:
    """
    Formato: [Etiqueta] [Operación] [Operandos]
    Sin comentarios para que sea limpio.
    """
    parts = []
    if etq:
        parts.append(f"{etq}:")
    else:
        parts.append("    ")

    if mnem:
        # si hay etiqueta, dejo un espacio; si no, indentación
        if etq:
            parts.append(f" {mnem:<6}")
        else:
            parts.append(f"{mnem:<6}")
        if oper:
            parts.append(f" {oper}")
    return "".join(parts).rstrip()


# ============================================================
#  Tokenizador y parser de expresiones (shunting-yard)
# ============================================================

OPERADORES = {"+", "-", "*", "/", "%"}
PRECEDENCIA = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
}
ASOCIATIVA_IZQ = {"+", "-", "*", "/", "%"}


def _tokenizar_expr(expr: str) -> List[str]:
    """
    Tokeniza una expresión en:
      - identificadores (variables)
      - números
      - operadores + - * / %
      - paréntesis ( )
    """
    tokens = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c.isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif c in "+-*/%()":
            tokens.append(c)
            i += 1
        else:
            # carácter desconocido, lo agregamos tal cual
            tokens.append(c)
            i += 1
    return tokens


def _a_postfijo(tokens: List[str]) -> List[str]:
    """
    Shunting-yard: infijo -> postfijo.
    Maneja + - * / % y paréntesis.
    """
    salida = []
    pila = []
    for tok in tokens:
        if tok in OPERADORES:
            while pila and pila[-1] in OPERADORES:
                top = pila[-1]
                if (PRECEDENCIA[top] > PRECEDENCIA[tok]) or (
                    PRECEDENCIA[top] == PRECEDENCIA[tok] and tok in ASOCIATIVA_IZQ
                ):
                    salida.append(pila.pop())
                else:
                    break
            pila.append(tok)
        elif tok == "(":
            pila.append(tok)
        elif tok == ")":
            while pila and pila[-1] != "(":
                salida.append(pila.pop())
            if pila and pila[-1] == "(":
                pila.pop()
        else:
            # identificador o número
            salida.append(tok)
    while pila:
        salida.append(pila.pop())
    return salida


# ============================================================
#  Generación de código ASM para una expresión
# ============================================================

def _emit_expr_to_ax(expr: str, asm_lines: List[str]) -> None:
    """
    Genera instrucciones que dejan el valor de 'expr' en AX.
    Usa AX como acumulador y BX como segundo operando.
    NO crea temporales en memoria.
    """
    expr = expr.strip()
    if not expr:
        asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
        return

    tokens = _tokenizar_expr(expr)
    post = _a_postfijo(tokens)

    # Elementos en la pila: dict(kind='var'|'imm'|'regAX', value=...)
    stack = []

    def load_to_ax(elem):
        """Carga elem en AX si no está ya en AX."""
        if elem["kind"] == "regAX":
            return
        if elem["kind"] == "imm":
            asm_lines.append(_linea_asm("", "MOV", f"AX, {elem['value']}"))
        else:  # var
            asm_lines.append(_linea_asm("", "MOV", f"AX, {elem['value']}"))

    def operand_txt(elem):
        if elem["kind"] == "imm":
            return elem["value"]
        elif elem["kind"] == "var":
            return elem["value"]
        else:  # regAX
            return "AX"

    for tok in post:
        if tok in OPERADORES:
            # Operador binario: pop derecha e izquierda
            if len(stack) < 2:
                # expresión mal formada, simplificamos a 0
                asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
                stack = [{"kind": "regAX"}]
                break

            right = stack.pop()
            left = stack.pop()

            # Suma / Resta
            if tok in ("+", "-"):
                load_to_ax(left)
                op2 = operand_txt(right)
                mnem = "ADD" if tok == "+" else "SUB"
                asm_lines.append(_linea_asm("", mnem, f"AX, {op2}"))
                stack.append({"kind": "regAX"})
                continue

            # Multiplicación
            if tok == "*":
                # Regla: AL * BL → AX
                # Ponemos left en AL, right en BL
                if left["kind"] == "regAX":
                    # ya está en AX, usamos AL directamente
                    pass
                else:
                    op1 = operand_txt(left)
                    asm_lines.append(_linea_asm("", "MOV", f"AL, {op1}"))
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "MUL", "BL"))
                # resultado en AX
                stack.append({"kind": "regAX"})
                continue

            # División
            if tok == "/":
                # AX = dividendo, BL = divisor, DIV BL
                load_to_ax(left)
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "DIV", "BL"))
                # Cociente en AL, pero tratamos AX como resultado
                stack.append({"kind": "regAX"})
                continue

            # Módulo
            if tok == "%":
                load_to_ax(left)
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "DIV", "BL"))
                # Residuo en AH -> lo pasamos a AL/AX
                asm_lines.append(_linea_asm("", "MOV", "AL, AH"))
                stack.append({"kind": "regAX"})
                continue

        else:
            # operando
            if _es_entero(tok):
                stack.append({"kind": "imm", "value": tok})
            else:
                stack.append({"kind": "var", "value": tok})

    # Al final, el tope debe estar en AX
    if not stack:
        asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
    else:
        top = stack.pop()
        load_to_ax(top)


# ============================================================
#  Generación de ASM para asignaciones y do-while
# ============================================================

REL_OPS = ["<=", ">=", "==", "!=", "<", ">"]

JCC_MAP = {
    "==": "JE",
    "!=": "JNE",
    ">":  "JG",
    "<":  "JL",
    ">=": "JGE",
    "<=": "JLE",
}


def _partir_relacional(cond: str):
    """
    Separa una condición simple tipo:
      expr OP expr
    donde OP ∈ REL_OPS.
    Devuelve (left_expr, op, right_expr) o (cond, None, None) si no encuentra.
    """
    cond = cond.strip()
    for op in REL_OPS:
        pos = cond.find(op)
        if pos != -1:
            left = cond[:pos].strip()
            right = cond[pos + len(op):].strip()
            return left, op, right
    return cond, None, None


def generar_ensamblador_8086(codigo_opt: str) -> str:
    """
    Recibe el CÓDIGO OPTIMIZADO como texto
    y genera el código ensamblador 8086.

    - Sin temporales de triplos.
    - Usa etiquetas para do-while.
    """

    lineas = codigo_opt.splitlines()
    asm: List[str] = []

    loop_stack: List[str] = []   # pila de etiquetas de inicio de do-while
    loop_counter = 0

    for linea in lineas:
        raw = linea.rstrip("\n")
        txt = raw.strip()

        if not txt:
            continue

        # -------------------------------------------------
        # Declaraciones: E$, F$, C$ ... -> por ahora se omiten
        # -------------------------------------------------
        if txt.startswith("E$") or txt.startswith("F$") or txt.startswith("C$"):
            # Podrías generar defs de variables aquí si lo requieren,
            # por ahora dejamos vacío.
            continue

        # -------------------------------------------------
        # do {  -> etiqueta de inicio de ciclo
        # -------------------------------------------------
        if txt.startswith("do") and "{" in txt:
            loop_counter += 1
            etq = f"ET_DO{loop_counter}"
            loop_stack.append(etq)
            asm.append(_linea_asm(etq, "", ""))
            continue

        # -------------------------------------------------
        # } while (cond);
        # -------------------------------------------------
        if txt.startswith("}") and "while" in txt:
            if not loop_stack:
                continue  # estructura rota, ignoramos

            etq_inicio = loop_stack.pop()

            # extraer condición entre paréntesis
            # ejemplo: } while (X * H <= 400);
            s = txt
            ini = s.find("(")
            fin = s.rfind(")")
            if ini != -1 and fin != -1 and fin > ini:
                cond = s[ini + 1:fin]
            else:
                cond = ""

            left, op_rel, right = _partir_relacional(cond)

            # si no hay relacional, no generamos nada
            if op_rel is None:
                continue

            # Evaluar left y right
            _emit_expr_to_ax(left, asm)          # AX = left
            # Right en BX o inmediato
            # Si es número, usamos inmediato en CMP, si no, cargamos a BX
            right = right.strip()
            if _es_entero(right):
                asm.append(_linea_asm("", "CMP", f"AX, {right}"))
            else:
                asm.append(_linea_asm("", "MOV", f"BX, {right}"))
                asm.append(_linea_asm("", "CMP", "AX, BX"))

            jcc = JCC_MAP.get(op_rel, "JMP")
            asm.append(_linea_asm("", jcc, etq_inicio))

            continue

        # -------------------------------------------------
        # Llaves sueltas { o } sin while -> no producen código
        # -------------------------------------------------
        if txt == "{" or txt == "}":
            continue

        # -------------------------------------------------
        # Asignaciones: ID = expr;
        # -------------------------------------------------
        # quitamos ';' final si existe
        linea_sin_puntoycoma = txt
        if linea_sin_puntoycoma.endswith(";"):
            linea_sin_puntoycoma = linea_sin_puntoycoma[:-1]

        # Buscar patrón ID = expr
        if "=" in linea_sin_puntoycoma:
            partes = linea_sin_puntoycoma.split("=", 1)
            lhs = partes[0].strip()
            rhs = partes[1].strip()

            if lhs:
                # Evaluar RHS en AX
                _emit_expr_to_ax(rhs, asm)
                # Mover resultado a la variable destino
                asm.append(_linea_asm("", "MOV", f"{lhs}, AX"))
                continue

        # -------------------------------------------------
        # Cualquier otra línea se ignora en esta versión
        # -------------------------------------------------
        # (si tuvieras otras estructuras, podrías manejarlas aquí)

    return "\n".join(asm)
