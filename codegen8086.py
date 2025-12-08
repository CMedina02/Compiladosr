# codegen8086.py
# -----------------------------------------
# Traduce triplos (N, O, D.O, D.F) a código
# ensamblador 8086 (modo real) siguiendo
# las reglas académicas dadas.
# -----------------------------------------

from typing import List, Tuple

Triplo = Tuple[int, str, str, str]

# -------------------------------------------------
# Utilidades de formato
# -------------------------------------------------

def _es_entero(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def _op_to_asm(op: str) -> str:
    """
    Convierte un operando del triplo a operando de ASM.
    Si es número → inmediato, si no, se deja como identificador.
    """
    if op is None:
        return ""
    op = str(op).strip()
    if op == "":
        return ""
    if _es_entero(op):
        return op
    return op


def _linea_asm(etq: str, mnem: str, oper: str, comentario: str = "") -> str:
    """
    Formato uniforme: [Etiqueta] [Operación] [Operando(s)]
    (Los comentarios se IGNORAN a propósito para que el ASM salga limpio)
    """
    parts = []

    if etq:
        parts.append(f"{etq}:")
    else:
        parts.append("    ")

    if mnem:
        if not etq:
            parts.append(f"{mnem:<6}")
        else:
            parts.append(f" {mnem:<6}")
        if oper:
            parts.append(f" {oper}")

    # NO agregamos comentarios al ensamblador
    return "".join(parts).rstrip()


# -------------------------------------------------
# Traducción de triplos a ensamblador
# -------------------------------------------------

REL_OPS = {"<", ">", "<=", ">=", "==", "!="}

JCC_MAP = {
    "==": "JE",
    "!=": "JNE",
    ">":  "JG",
    "<":  "JL",
    ">=": "JGE",
    "<=": "JLE",
}


def generar_ensamblador_8086(triplos: List[Triplo]) -> str:
    """
    Recibe la tabla de triplos [(N, O, DO, DF), ...]
    y regresa un string con el código ensamblador 8086.

    Suposiciones:
    - Triplos aritméticos tipo 3-direcciones:
        =  dest  src        → dest = src
        +  dest  src        → dest = dest + src
        -  dest  src        → dest = dest - src
        *  dest  src        → dest = dest * src
        /  dest  src        → dest = dest / src
        %  dest  src        → dest = dest % src
    - Triplos relacionales vienen en bloques: [REL] / TRUE / FALSE
        N:   op_rel   T1   K
        N+1: TRUE     destV   -
        N+2: FALSE    destF   -
    """

    # 1) Detectar qué N son destino de saltos (para poner etiquetas)
    destinos = set()
    for n, o, do, df in triplos:
        if o in ("TRUE", "FALSE"):
            d = str(do).strip()
            if d.isdigit():
                destinos.add(int(d))

    asm_lines: List[str] = []
    i = 0
    n2index = {n: idx for idx, (n, _, _, _) in enumerate(triplos)}

    while i < len(triplos):
        n, op, do, df = triplos[i]
        op = str(op).strip() if op is not None else ""
        do = str(do).strip() if do is not None else ""
        df = str(df).strip() if df is not None else ""

        etiqueta_actual = f"ET{n}" if n in destinos else ""

        # -------------------------
        #  BLOQUE RELACIONAL: op / TRUE / FALSE
        # -------------------------
        if op in REL_OPS:
            if i + 2 < len(triplos):
                n_true, o_true, do_true, _ = triplos[i + 1]
                n_false, o_false, do_false, _ = triplos[i + 2]

                if o_true == "TRUE" and o_false == "FALSE":
                    left = _op_to_asm(do)
                    right = _op_to_asm(df)

                    dest_true = str(do_true).strip()
                    dest_false = str(do_false).strip()

                    etq_true = f"ET{dest_true}" if dest_true.isdigit() else dest_true
                    etq_false = f"ET{dest_false}" if dest_false.isdigit() else dest_false

                    asm_lines.append(
                        _linea_asm(
                            etiqueta_actual,
                            "MOV",
                            f"AX, {left}"
                        )
                    )
                    asm_lines.append(
                        _linea_asm(
                            "",
                            "CMP",
                            f"AX, {right}"
                        )
                    )

                    jcc = JCC_MAP.get(op, "JMP")
                    asm_lines.append(
                        _linea_asm(
                            "",
                            jcc,
                            etq_true
                        )
                    )
                    asm_lines.append(
                        _linea_asm(
                            "",
                            "JMP",
                            etq_false
                        )
                    )

                    i += 3
                    continue

            left = _op_to_asm(do)
            right = _op_to_asm(df)
            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "CMP",
                    f"{left}, {right}"
                )
            )
            i += 1
            continue

        # TRUE / FALSE: sólo marcadores de salto
        if op in ("TRUE", "FALSE"):
            if etiqueta_actual:
                asm_lines.append(_linea_asm(etiqueta_actual, "", ""))
            i += 1
            continue

        # -------------------------
        #  Asignación simple (=)
        # -------------------------
        if op == "=":
            dest = _op_to_asm(do)
            src = _op_to_asm(df)
            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "MOV",
                    f"{dest}, {src}"
                )
            )
            i += 1
            continue

        # -------------------------
        #  Suma / Resta: dest = dest (+|-) src
        # -------------------------
        if op in ("+", "-"):
            dest = _op_to_asm(do)
            src = _op_to_asm(df)
            mnem = "ADD" if op == "+" else "SUB"

            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "MOV",
                    f"AX, {dest}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    mnem,
                    f"AX, {src}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"{dest}, AX"
                )
            )
            i += 1
            continue

        # -------------------------
        #  Multiplicación: dest = dest * src
        # -------------------------
        if op == "*":
            dest = _op_to_asm(do)
            src = _op_to_asm(df)

            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "MOV",
                    f"AL, {dest}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"BL, {src}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MUL",
                    "BL"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"{dest}, AX"
                )
            )
            i += 1
            continue

        # -------------------------
        #  División: dest = dest / src
        # -------------------------
        if op == "/":
            dest = _op_to_asm(do)
            src = _op_to_asm(df)

            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "MOV",
                    f"AX, {dest}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"BL, {src}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "DIV",
                    "BL"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"{dest}, AL"
                )
            )
            i += 1
            continue

        # -------------------------
        #  Módulo: dest = dest % src
        # -------------------------
        if op == "%":
            dest = _op_to_asm(do)
            src = _op_to_asm(df)

            asm_lines.append(
                _linea_asm(
                    etiqueta_actual,
                    "MOV",
                    f"AX, {dest}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"BL, {src}"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "DIV",
                    "BL"
                )
            )
            asm_lines.append(
                _linea_asm(
                    "",
                    "MOV",
                    f"{dest}, AH"
                )
            )
            i += 1
            continue

        # -------------------------
        #  Operador desconocido
        # -------------------------
        asm_lines.append(
            _linea_asm(
                etiqueta_actual,
                "NOP",
                ""
            )
        )
        i += 1

    return "\n".join(asm_lines)
