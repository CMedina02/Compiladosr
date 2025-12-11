from typing import List

def _es_entero(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False

def _linea_asm(etq: str, mnem: str, oper: str = "") -> str:
    label_txt = f"{etq}:" if etq else ""
    mnem_txt = mnem or ""
    oper_txt = oper or ""
    return f"{label_txt:<8} {mnem_txt:<8} {oper_txt}".rstrip()

# =========================
# Simplificación muy básica (para condiciones y asignaciones)
# =========================
def _simplificar_expr(expr: str) -> str:
    e = expr.strip()
    if not e:
        return e
    # no tocar strings/chars
    if '"' in e or "'" in e:
        return e

    old = None
    while old != e:
        old = e
        e = e.replace(" ", "")
        # /1, *1, 1*, +0, 0+, -0
        e = e.replace("/1", "")
        e = e.replace("*1", "")
        # 1*X -> X (solo patrón simple)
        if e.startswith("1*"):
            e = e[2:]
        e = e.replace("+0", "")
        if e.startswith("0+"):
            e = e[2:]
        e = e.replace("-0", "")
        # volver a una forma con espacios mínimos no es necesario para tokenizar
    return e

# =========================
# Tokenizador / Postfijo
# =========================
OPERADORES = {"+", "-", "*", "/", "%"}
PRECEDENCIA = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2}
ASOCIATIVA_IZQ = {"+", "-", "*", "/", "%"}

def _tokenizar_expr(expr: str) -> List[str]:
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
            continue
        if c.isdigit():
            j = i + 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
            continue
        if c in "+-*/%()":
            tokens.append(c)
            i += 1
            continue
        # carácter desconocido
        tokens.append(c)
        i += 1
    return tokens

def _a_postfijo(tokens: List[str]) -> List[str]:
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
            salida.append(tok)
    while pila:
        salida.append(pila.pop())
    return salida

# =========================
# Generación ASM para expresión -> AX
# =========================
def _emit_expr_to_ax(expr: str, asm_lines: List[str]) -> None:
    expr = _simplificar_expr(expr)
    expr = expr.strip()
    if not expr:
        asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
        return

    tokens = _tokenizar_expr(expr)
    post = _a_postfijo(tokens)

    stack = []

    def load_to_ax(elem):
        if elem["kind"] == "regAX":
            return
        if elem["kind"] == "imm":
            asm_lines.append(_linea_asm("", "MOV", f"AX, {elem['value']}"))
        else:
            asm_lines.append(_linea_asm("", "MOV", f"AX, {elem['value']}"))

    def operand_txt(elem):
        if elem["kind"] == "imm":
            return elem["value"]
        if elem["kind"] == "var":
            return elem["value"]
        return "AX"

    for tok in post:
        if tok in OPERADORES:
            if len(stack) < 2:
                asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
                stack = [{"kind": "regAX"}]
                break

            right = stack.pop()
            left = stack.pop()

            if tok in ("+", "-"):
                load_to_ax(left)
                op2 = operand_txt(right)
                mnem = "ADD" if tok == "+" else "SUB"
                asm_lines.append(_linea_asm("", mnem, f"AX, {op2}"))
                stack.append({"kind": "regAX"})
                continue

            if tok == "*":
                if left["kind"] != "regAX":
                    asm_lines.append(_linea_asm("", "MOV", f"AL, {operand_txt(left)}"))
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "MUL", "BL"))
                stack.append({"kind": "regAX"})
                continue

            if tok == "/":
                load_to_ax(left)
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "DIV", "BL"))
                stack.append({"kind": "regAX"})
                continue

            if tok == "%":
                load_to_ax(left)
                op2 = operand_txt(right)
                asm_lines.append(_linea_asm("", "MOV", f"BL, {op2}"))
                asm_lines.append(_linea_asm("", "DIV", "BL"))
                asm_lines.append(_linea_asm("", "MOV", "AL, AH"))
                stack.append({"kind": "regAX"})
                continue

        else:
            if _es_entero(tok):
                stack.append({"kind": "imm", "value": tok})
            else:
                stack.append({"kind": "var", "value": tok})

    if not stack:
        asm_lines.append(_linea_asm("", "MOV", "AX, 0"))
    else:
        top = stack.pop()
        load_to_ax(top)

# =========================
# Relacionales
# =========================
REL_OPS = ["<=", ">=", "==", "!=", "<", ">"]

REL_JMP = {
    "==": "EQ",
    "!=": "NE",
    ">":  "GT",
    "<":  "LT",
    ">=": "GE",
    "<=": "LE",
}

def _partir_relacional(cond: str):
    cond = cond.strip()
    for op in REL_OPS:
        pos = cond.find(op)
        if pos != -1:
            left = cond[:pos].strip()
            right = cond[pos + len(op):].strip()
            return left, op, right
    return cond, None, None

def _es_atomico(expr: str) -> bool:
    """Si expr es solo un número o un identificador simple."""
    e = expr.strip()
    if not e:
        return False
    if _es_entero(e):
        return True
    # identificador simple
    return all(ch.isalnum() or ch == "_" for ch in e) and (e[0].isalpha() or e[0] == "_")

# =========================
# Generador principal
# =========================
def generar_ensamblador_8086(codigo_opt: str) -> str:
    lineas = codigo_opt.splitlines()
    asm: List[str] = []

    loop_stack: List[str] = []
    loop_counter = 0

    for linea in lineas:
        txt = linea.strip()
        if not txt:
            continue

        # omitir declaraciones
        if txt.startswith("E$") or txt.startswith("F$") or txt.startswith("C$"):
            continue

        # do {
        if txt.startswith("do") and "{" in txt:
            loop_counter += 1
            etq = f"ET_DO{loop_counter}"
            loop_stack.append(etq)
            asm.append(_linea_asm(etq, "", ""))
            continue

        # } while (cond);
        if txt.startswith("}") and "while" in txt:
            if not loop_stack:
                continue
            etq_inicio = loop_stack.pop()

            ini = txt.find("(")
            fin = txt.rfind(")")
            cond = txt[ini + 1:fin] if (ini != -1 and fin != -1 and fin > ini) else ""

            left, op_rel, right = _partir_relacional(cond)
            if op_rel is None:
                continue

            left = _simplificar_expr(left)
            right = _simplificar_expr(right)

            # AX = left
            _emit_expr_to_ax(left, asm)

            # --- Arreglado: si right es expresión, se calcula en AX y se pasa a BX ---
            if _es_entero(right):
                asm.append(_linea_asm("", "CMP", f"AX, {right}"))
            elif _es_atomico(right):
                asm.append(_linea_asm("", "MOV", f"BX, {right}"))
                asm.append(_linea_asm("", "CMP", "AX, BX"))
            else:
                asm.append(_linea_asm("", "PUSH", "AX"))
                _emit_expr_to_ax(right, asm)          # AX = right
                asm.append(_linea_asm("", "MOV", "BX, AX"))
                asm.append(_linea_asm("", "POP", "AX"))
                asm.append(_linea_asm("", "CMP", "AX, BX"))

            jmp_cond = REL_JMP.get(op_rel, None)
            if jmp_cond:
                asm.append(_linea_asm("", jmp_cond, etq_inicio))
            continue

        if txt == "{" or txt == "}":
            continue

        # asignaciones
        linea_sin_puntoycoma = txt[:-1] if txt.endswith(";") else txt
        if "=" in linea_sin_puntoycoma:
            lhs, rhs = linea_sin_puntoycoma.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if lhs:
                _emit_expr_to_ax(rhs, asm)
                asm.append(_linea_asm("", "MOV", f"{lhs}, AX"))
                continue

    return "\n".join(asm)
