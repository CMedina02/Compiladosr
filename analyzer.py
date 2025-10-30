# analyzer.py
import re

# ===== Lexer =====
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
KW = {"do", "while"}

REL_OPS = {"==", "!=", "<", "<=", ">", ">="}
OPA     = {"+", "-", "*", "/", "%"}

TOK_INT, TOK_FLOAT, TOK_STR = "E$", "F$", "C$"


def lex(texto: str):
    tokens, line, col, pos = [], 1, 1, 0
    for m in MASTER_RE.finditer(texto):
        typ, lexeme, start = m.lastgroup, m.group(), m.start()
        while pos < start:
            if texto[pos] == "\n":
                line += 1; col = 1
            else:
                col += 1
            pos += 1
        if typ in ("WS", "COMMENT"):
            for ch in lexeme:
                if ch == "\n": line += 1; col = 1
                else: col += 1
            pos = m.end(); continue
        if typ == "NEWLINE":
            line += 1; col = 1; pos = m.end(); continue
        tok = {"tipo": typ, "lex": lexeme, "linea": line, "col": col}
        if typ == "IDENT" and lexeme in KW:
            tok["tipo"] = "KW"
        tokens.append(tok)
        for ch in lexeme:
            if ch == "\n": line += 1; col = 1
            else: col += 1
        pos = m.end()
    return tokens


# ===== 1ª pasada: Tabla de símbolos =====
def analizar_primera_pasada(texto: str):
    toks = lex(texto)
    i, n = 0, len(toks)
    tabla, declarados = [], set()
    while i < n:
        tk = toks[i]
        if tk["tipo"] == "TYPE":
            t = tk["lex"]; i += 1
            while i < n:
                if i < n and toks[i]["tipo"] == "IDENT":
                    name = toks[i]["lex"]
                    if name not in declarados:
                        tabla.append((name, t)); declarados.add(name)
                    i += 1
                    if i < n and toks[i]["lex"] == "=":
                        i += 1
                        depth = 0
                        while i < n and not (depth == 0 and toks[i]["lex"] in (",", ";")):
                            if toks[i]["lex"] == "(": depth += 1
                            elif toks[i]["lex"] == ")": depth -= 1
                            i += 1
                    if i < n and toks[i]["lex"] == ",": i += 1; continue
                    if i < n and toks[i]["lex"] == ";": i += 1; break
                    break
                else:
                    while i < n and toks[i]["lex"] != ";": i += 1
                    if i < n: i += 1
                    break
        else:
            i += 1
    return tabla


# ===== 2ª pasada: Errores semánticos (básicos y no bloqueantes) =====
def _tipo_literal(tok):
    if tok['tipo'] == 'INT':   return TOK_INT
    if tok['tipo'] == 'FLOAT': return TOK_FLOAT
    if tok['tipo'] == 'STR':   return TOK_STR
    return ''


def recolectar_errores_semanticos(toks, tabla):
    scope = {name: t for name, t in tabla}
    errores = []
    i, n = 0, len(toks)

    def atom(j):
        if j >= n: return ('', ''), j
        tk = toks[j]
        if tk['lex'] == '(':
            t1, k, _ = expr(j+1)
            while k < n and toks[k]['lex'] != ')': k += 1
            return ('', t1), (k+1 if k < n else k)
        if tk['tipo'] == 'IDENT':
            return (tk['lex'], scope.get(tk['lex'], '')), j+1
        return (tk['lex'], _tipo_literal(tk)), j+1

    def expr(j):
        (lex1, t1), k = atom(j)
        parts = [(lex1, t1)]
        while k < n and toks[k]['lex'] in OPA:
            op = toks[k]['lex']; k += 1
            (lex2, t2), k = atom(k)
            if t1 in (TOK_INT, TOK_FLOAT) and t2 in (TOK_INT, TOK_FLOAT):
                t1 = TOK_FLOAT if (op == '/' or TOK_FLOAT in (t1, t2)) else TOK_INT
            elif t1 == TOK_STR and t2 == TOK_STR and op == '+':
                t1 = TOK_STR
            else:
                errores.append({'token':'ErrTipoOPA','linea':toks[k-1]['linea'],'col':toks[k-1]['col'],
                                'lex':lex2 or lex1 or op,'desc':'Operación incompatible de tipos'})
                t1 = t1 or t2 or TOK_INT
            parts.append((lex2, t2))
        return t1, k, parts

    while i < n:
        tk = toks[i]
        if tk['tipo'] == 'IDENT' and i+1 < n and toks[i+1]['lex'] == '=':
            lhs = tk['lex']; lhs_t = scope.get(lhs, '')
            if not lhs_t:
                errores.append({'token':'ErrUndef','linea':tk['linea'],'col':tk['col'],
                                'lex':lhs,'desc':'Variable indefinida'})
            j = i+2
            rhs_t, j, parts = expr(j)
            indef = set()
            for lx, ty in parts:
                if lx and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lx) and scope.get(lx, '') == '':
                    if lx not in indef:
                        indef.add(lx)
                        errores.append({'token':'ErrUndef','linea':tk['linea'],'col':tk['col'],
                                        'lex':lx,'desc':'Variable indefinida'})
            if lhs_t:
                allowed = {TOK_INT} if lhs_t == TOK_INT else ({TOK_INT, TOK_FLOAT} if lhs_t == TOK_FLOAT else {TOK_STR})
                if rhs_t not in allowed:
                    errores.append({'token':f'ErrAsig_{lhs_t}','linea':tk['linea'],'col':tk['col'],
                                    'lex':lhs,'desc':f'Asignación incompatible (resultado {rhs_t} ≠ {lhs_t})'})
            while j < n and toks[j]['lex'] != ';': j += 1
            i = j + 1 if j < n else j
            continue
        i += 1
    return errores


def analizar_dos_pasadas(texto: str):
    tabla = analizar_primera_pasada(texto)
    errores = recolectar_errores_semanticos(lex(texto), tabla)
    return tabla, errores


# ===== Utilidades parsing (para la fase de triplos) =====
def _strip_parens(tokens):
    if len(tokens) >= 2 and tokens[0]['lex'] == '(' and tokens[-1]['lex'] == ')':
        depth = 0; ok = True
        for k, tk in enumerate(tokens):
            if tk['lex'] == '(': depth += 1
            elif tk['lex'] == ')':
                depth -= 1
                if depth == 0 and k != len(tokens)-1:
                    ok = False; break
        if ok: return tokens[1:-1]
    return tokens


def _split_top(tokens, seps):
    out, depth, last = [], 0, 0
    for i, tk in enumerate(tokens):
        if tk['lex'] == '(': depth += 1
        elif tk['lex'] == ')': depth -= 1
        elif depth == 0 and tk['lex'] in seps:
            out.append(tokens[last:i]); last = i + 1
    out.append(tokens[last:])
    return out


def _slice_until(tokens, i, stop_lexes):
    j, depth = i, 0
    while j < len(tokens):
        if tokens[j]['lex'] == '(': depth += 1
        elif tokens[j]['lex'] == ')': depth -= 1
        if depth == 0 and tokens[j]['lex'] in stop_lexes: break
        j += 1
    return tokens[i:j], j


# ===== Emisor de TRIPLOS =====
class TriploEmitter:
    def __init__(self):
        self.rows = []  # [{'N','O','DO','DF'}]
        self.N = 0
        self.temp = 0

    def _nextN(self):
        self.N += 1
        return self.N

    def emit(self, O, DO="", DF=""):
        n = self._nextN()
        self.rows.append({"N": n, "O": O, "DO": DO, "DF": DF})
        return n

    def newT(self):
        self.temp += 1
        return f"T{self.temp}"

    # ---- Expresiones aritméticas (con unario -) ----
    def _gen_factor(self, toks, i):
        if i < len(toks) and toks[i]['lex'] == '-':
            i += 1
            tname, i = self._gen_factor(toks, i)
            t0 = self.newT(); self.emit('=', t0, '0'); self.emit('-', t0, tname)
            return t0, i
        if i < len(toks) and toks[i]['lex'] == '(':
            tname, j = self._gen_expr(toks, i + 1)
            k, depth = j, 1
            while k < len(toks) and depth > 0:
                if toks[k]['lex'] == '(': depth += 1
                elif toks[k]['lex'] == ')': depth -= 1
                k += 1
            return tname, k
        tk = toks[i]; t = self.newT(); self.emit('=', t, tk['lex']); return t, i + 1

    def _gen_term(self, toks, i):
        leftT, k = self._gen_factor(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('*', '/', '%'):
            op = toks[k]['lex']; k += 1
            rightT, k = self._gen_factor(toks, k)
            self.emit(op, leftT, rightT)
        return leftT, k

    def _gen_expr(self, toks, i):
        leftT, k = self._gen_term(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('+', '-'):
            op = toks[k]['lex']; k += 1
            rightT, k = self._gen_term(toks, k)
            self.emit(op, leftT, rightT)
        return leftT, k

    def gen_assignment(self, lhs_lex, rhs_tokens):
        self.temp = 0
        if not rhs_tokens:
            self.emit('=', lhs_lex, '0'); return
        tname, _ = self._gen_expr(rhs_tokens, 0)
        self.emit('=', lhs_lex, tname)

    # ---- Relacional: emite solo la fila [REL], retorna su N ----
    def gen_rel_row(self, rel_tokens):
        self.temp = 0
        depth, pos, relop = 0, -1, None
        for j, tk in enumerate(rel_tokens):
            if tk['lex'] == '(':
                depth += 1
            elif tk['lex'] == ')':
                depth -= 1
            elif depth == 0 and tk['lex'] in REL_OPS:
                pos = j; relop = tk['lex']; break
        if relop is None:
            t, _ = self._gen_expr(rel_tokens, 0)
            return self.emit('!=', t, '0')
        left = rel_tokens[:pos]; right = rel_tokens[pos+1:]
        tL, _ = self._gen_expr(left, 0)
        tR, _ = self._gen_expr(right, 0)
        return self.emit(relop, tL, tR)

    # ---- Condición de do-while (|| y && con cortocircuito) ----
    def gen_condition_do_while(self, cond_tokens, N_body):
        """
        Por cada relacional emite EXACTAMENTE 3 filas consecutivas:
          [REL], TRUE, FALSE  (TRUE/FALSE en O; destino en DO)
        Devuelve una lista de **índices** de filas que deben apuntar a END del do actual.
        """
        cond_tokens = _strip_parens(cond_tokens)
        or_terms = _split_top(cond_tokens, {'||'})

        pending_false_to_next_term = []  # índices: FALSE -> primer [REL] del sig. término
        pending_to_end = []              # índices: (TRUE/FALSE) -> END del do

        for t_idx, term in enumerate(or_terms):
            term = _strip_parens(term)
            and_factors = _split_top(term, {'&&'})

            pending_true_to_next_factor = []  # índices: TRUE -> [REL] del sig. factor
            first_relN_of_term = None
            local_false_of_term = []

            for f_idx, factor in enumerate(and_factors):
                factor = _strip_parens(factor)

                relN = self.gen_rel_row(factor)

                # Primer [REL] del término: resuelve pendientes FALSE→siguiente término
                if first_relN_of_term is None:
                    first_relN_of_term = relN
                    for ridx in pending_false_to_next_term:
                        self.rows[ridx]["DO"] = str(first_relN_of_term)
                    pending_false_to_next_term = []

                # TRUE pendientes (AND) van al [REL] actual
                for ridx in pending_true_to_next_factor:
                    self.rows[ridx]["DO"] = str(relN)
                pending_true_to_next_factor = []

                last_factor = (f_idx == len(and_factors) - 1)
                last_term   = (t_idx == len(or_terms) - 1)

                if len(and_factors) == 1:
                    # término con un solo factor
                    self.emit("TRUE", str(N_body), "")
                    self.emit("FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM", "")
                    idx_false = len(self.rows) - 1
                    (pending_to_end if last_term else local_false_of_term).append(idx_false)
                else:
                    if not last_factor:
                        self.emit("TRUE", "PENDING_NEXT_FACTOR", "")
                        idx_true = len(self.rows) - 1
                        pending_true_to_next_factor.append(idx_true)
                        self.emit("FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM", "")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)
                    else:
                        self.emit("TRUE", str(N_body), "")
                        self.emit("FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM", "")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)

            pending_false_to_next_term.extend(local_false_of_term)

        # FALSE→siguiente término que quedó sin siguiente término ⇒ END
        for ridx in pending_false_to_next_term:
            self.rows[ridx]["DO"] = "PENDING_END"

        # Seguridad: TRUE→NEXT_FACTOR sin resolver ⇒ BODY
        for r in self.rows:
            if r["O"] == "TRUE" and r["DO"] == "PENDING_NEXT_FACTOR":
                r["DO"] = str(N_body)

        return pending_to_end


# ===== Generación de triplos desde el código =====
def generar_triplos(texto: str):
    toks = lex(texto)
    em = TriploEmitter()
    i, n = 0, len(toks)

    def parse_assignment(idx):
        lhs = toks[idx]["lex"]; j = idx + 2
        rhs, k = _slice_until(toks, j, {';', '}'})
        em.gen_assignment(lhs, rhs)
        idx = k
        if idx < n and toks[idx]["lex"] == ';': idx += 1
        return idx

    def parse_do(idx):
        # do { ... } while (cond);
        assert toks[idx]["tipo"] == "KW" and toks[idx]["lex"].lower() == "do"
        idx += 1
        if idx >= n or toks[idx]["lex"] != "{":
            return idx
        idx += 1

        body_start_N = em.N + 1  # primera instrucción dentro del bloque

        # cuerpo del do (asignaciones y do anidados)
        while idx < n and toks[idx]["lex"] != "}":
            if idx+1 < n and toks[idx]["tipo"] == "IDENT" and toks[idx+1]["lex"] == "=":
                idx = parse_assignment(idx)
            elif toks[idx]["tipo"] == "KW" and toks[idx]["lex"].lower() == "do":
                idx = parse_do(idx)
            else:
                idx += 1

        if idx < n and toks[idx]["lex"] == "}": idx += 1

        # while (cond);
        if idx < n and toks[idx]["tipo"] == "KW" and toks[idx]["lex"].lower() == "while":
            idx += 1
            if idx < n and toks[idx]["lex"] == "(":
                j, depth = idx + 1, 1
                while j < n and depth > 0:
                    if toks[j]["lex"] == "(": depth += 1
                    elif toks[j]["lex"] == ")": depth -= 1
                    j += 1
                cond_tokens = toks[idx+1:j-1] if j-1 > idx+1 else []
                pending_end_rows = em.gen_condition_do_while(cond_tokens, body_start_N)
                idx = j
                if idx < n and toks[idx]["lex"] == ";": idx += 1

                # END local = primera N después de este do
                local_end = em.N + 1
                for ridx in pending_end_rows:
                    em.rows[ridx]["DO"] = str(local_end)
        return idx

    while i < n:
        tk = toks[i]
        if tk["tipo"] == "KW" and tk["lex"].lower() == "do":
            i = parse_do(i); continue
        if i+1 < n and tk["tipo"] == "IDENT" and toks[i+1]["lex"] == "=":
            i = parse_assignment(i); continue
        i += 1

    return [(r["N"], r["O"], r["DO"], r["DF"]) for r in em.rows]


def analizar_y_triplos(texto: str):
    tabla, errores = analizar_dos_pasadas(texto)
    try:
        triplos = generar_triplos(texto)
    except Exception:
        triplos = []
    return tabla, errores, triplos
