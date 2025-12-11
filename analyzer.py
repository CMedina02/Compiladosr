# analyzer.py — Compilador educativo + optimización local

# analyzer.py — Compilador educativo + optimización local

import re
from rules import (
    TOKEN_PATTERN, idx_a_line_col,
    RE_TIPO, RE_IDENT, RE_FLOAT, RE_INT, RE_STR, RE_CHAR, RE_SYSPRINT,
    TOK_INT, TOK_STR, TOK_FLOAT
)

# =========================================================================
#                          CLASES AST
# =========================================================================

class Node:
    def __init__(self, token=None, children=None):
        self.token = token
        self.children = children if children is not None else []

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class BinOp(Node):
    def __init__(self, op_token, left, right):
        super().__init__(op_token, [left, right])
        self.op = op_token['lex']
        self.left = left
        self.right = right

    def __repr__(self):
        return f"<BinOp: {self.op}>"


class Literal(Node):
    def __init__(self, token):
        super().__init__(token)
        self.value = token['lex']

    def __repr__(self):
        return f"<Literal: {self.value}>"


class Ident(Node):
    def __init__(self, token):
        super().__init__(token)
        self.name = token['lex']

    def __repr__(self):
        return f"<Ident: {self.name}>"


# =========================================================================
#                              LEXER
# =========================================================================

KW_SET = {'if', 'else', 'while', 'for', 'do', 'break', 'continue', 'func', 'return'}

def lex(texto):
    """
    Devuelve lista de dicts: {'tipo','lex','linea','col'}.
    tipo ∈ { TIPO, IDENT, INT, FLOAT, STR, CHAR, KW, OP, DESCONOCIDO }
    NO agrega token EOF; el parser lo simula en peek().
    """
    toks = []
    i = 0
    while i < len(texto):
        m = TOKEN_PATTERN.match(texto, i)
        if not m:
            linea, col = idx_a_line_col(texto, i)
            toks.append({'tipo': 'DESCONOCIDO', 'lex': texto[i], 'linea': linea, 'col': col})
            i += 1
            continue

        start = m.start(1)
        linea, col = idx_a_line_col(texto, start)
        lexema = m.group(1)
        i = m.end()

        if re.fullmatch(RE_TIPO, lexema):
            tipo = 'TIPO'
        elif re.fullmatch(RE_FLOAT, lexema):
            tipo = 'FLOAT'
        elif re.fullmatch(RE_INT, lexema):
            tipo = 'INT'
        elif re.fullmatch(RE_STR, lexema):
            tipo = 'STR'
        elif re.fullmatch(RE_CHAR, lexema):
            tipo = 'CHAR'
        elif re.fullmatch(RE_SYSPRINT, lexema):
            tipo = 'KW'
            lexema = 'System.out.println'
        elif re.fullmatch(RE_IDENT, lexema):
            lower = lexema.lower()
            if lower in KW_SET:
                tipo = 'KW'
                lexema = lower
            else:
                tipo = 'IDENT'
        else:
            tipo = 'OP'

        toks.append({'tipo': tipo, 'lex': lexema, 'linea': linea, 'col': col})

    return toks

def optimizar_codigo_fuente(texto: str) -> str:
    """
    Optimización local del código fuente ANTES del análisis léxico.

    Hace:
    - CSE completo (expresiones idénticas sin cambios en sus variables).
    - CSE parcial de subexpresiones:
        * X = (A * B) + (C / D);
          Z = ((A * B) + (C / D)) * P;
          --> Z = X * P;

        * K = (X + 1) * A;
          M = (X + 1) * A + B;
          --> M = K + B;

    - Simplificaciones algebraicas:
        /1, *1, 1*, +0, 0+, -0, y caso especial var + (var * 2).
    """

    lineas = texto.splitlines()

    # Normalizamos el patrón de identificador para incrustarlo en otros regex
    ident_core = RE_IDENT
    if ident_core.startswith('^'):
        ident_core = ident_core[1:]
    if ident_core.endswith('$'):
        ident_core = ident_core[:-1]

    # Asignaciones del tipo:   ID = expr;
    re_asig = re.compile(r'^(\s*)(' + ident_core + r')\s*=\s*(.+?);?\s*$')
    re_ident = re.compile(ident_core)

    # ---------------------------------------------
    # Simplificación algebraica local
    # ---------------------------------------------
    def simplificar_expr(expr: str) -> str:
        if '"' in expr or "'" in expr:
            return expr

        e = expr
        old = None
        while old != e:
            old = e
            e = re.sub(r'/\s*1\b', '', e)         # /1
            e = re.sub(r'\*\s*1\b', '', e)        # *1
            e = re.sub(r'\b1\s*\*\s*', '', e)     # 1*
            e = re.sub(r'\s*\+\s*0\b', '', e)     # +0
            e = re.sub(r'\b0\s*\+\s*', '', e)     # 0+
            e = re.sub(r'\s*-\s*0\b', '', e)      # -0

            # C = C + B * 2;  ->  C = C + B;
            pattern_sum_var_times2 = (
                r'(\b' + ident_core + r'\s*\+\s*)'
                r'(' + ident_core + r')\s*\*\s*2\b'
            )
            e = re.sub(pattern_sum_var_times2, r'\1\2', e)

        return e

    def sin_espacios(s: str) -> str:
        return re.sub(r'\s+', '', s)

    def pretty_from_ns(ns: str) -> str:
        out = []
        buf = ""
        for ch in ns:
            if ch in "+-*/%":
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(f" {ch} ")
            elif ch in "()":
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(ch)
            else:
                buf += ch
        if buf:
            out.append(buf)
        s = "".join(out)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    versiones = {}          # nombre -> versión
    expr_cache = {}         # (expr_norm, snapshot) -> lhs
    subexprs = []           # candidatos de subexpresión

    resultado = []

    def snapshot_versiones(usados):
        return tuple(sorted((v, versiones.get(v, 0)) for v in usados))

    def normalizar_expr(expr: str) -> str:
        expr = expr.strip()

        while expr.startswith('(') and expr.endswith(')'):
            depth = 0
            wrapped = True
            for i, ch in enumerate(expr[:-1]):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                if depth == 0 and i > 0:
                    wrapped = False
                    break
            if wrapped:
                expr = expr[1:-1].strip()
            else:
                break

        if '"' in expr or "'" in expr or '/' in expr:
            return re.sub(r'\s+', '', expr)

        expr_clean = re.sub(r'\s+', '', expr)
        if not (expr_clean.startswith('+') or expr_clean.startswith('-')):
            expr_clean = '+' + expr_clean

        terms = re.findall(r'[+-][^+-]+', expr_clean)
        if not terms or len("".join(terms)) != len(expr_clean):
            return re.sub(r'\s+', '', expr)

        norm_terms = []
        for term in terms:
            sign = term[0]
            content = term[1:]
            if '*' in content and '(' not in content and ')' not in content:
                factors = content.split('*')
                factors.sort()
                content = '*'.join(factors)
            norm_terms.append(sign + content)

        norm_terms.sort()
        return ''.join(norm_terms)

    # ---------------------------------------------
    # Reemplazo de subexpresiones con variables previas
    # ---------------------------------------------
    def reemplazar_subexpresiones(expr: str) -> str:
        expr_ns = sin_espacios(expr)
        if not expr_ns:
            return expr

        # Probamos primero las expresiones más largas
        candidatos = sorted(subexprs, key=lambda c: len(c["expr_ns"]), reverse=True)

        for cand in candidatos:
            snap = cand["snapshot"]
            ok = True
            for v, ver in snap:
                if versiones.get(v, 0) != ver:
                    ok = False
                    break
            if not ok:
                continue

            sub_ns = cand["expr_ns"]
            if not sub_ns:
                continue
            if sub_ns not in expr_ns:
                continue

            var = cand["lhs"]
            new_ns = None

            # Prefijo directo: (X+1)*A+B  donde (X+1)*A ya existe
            if expr_ns.startswith(sub_ns) and len(expr_ns) > len(sub_ns):
                sig = expr_ns[len(sub_ns)]
                if sig in "+-*/%":
                    new_ns = var + expr_ns[len(sub_ns):]

            # Prefijo con paréntesis: ((A*B)+(C/D))*P
            if new_ns is None:
                pref = "(" + sub_ns + ")"
                if expr_ns.startswith(pref) and len(expr_ns) > len(pref):
                    sig = expr_ns[len(pref)]
                    if sig in "+-*/%":
                        new_ns = var + expr_ns[len(pref):]

            # Sufijo directo
            if new_ns is None and len(expr_ns) > len(sub_ns):
                if expr_ns.endswith(sub_ns):
                    pos = len(expr_ns) - len(sub_ns) - 1
                    if pos >= 0 and expr_ns[pos] in "+-*/%":
                        new_ns = expr_ns[:pos + 1] + var

            # Sufijo con paréntesis
            if new_ns is None:
                suf = "(" + sub_ns + ")"
                if expr_ns.endswith(suf) and len(expr_ns) > len(suf):
                    pos = len(expr_ns) - len(suf) - 1
                    if pos >= 0 and expr_ns[pos] in "+-*/%":
                        new_ns = expr_ns[:pos + 1] + var

            if new_ns is not None:
                return pretty_from_ns(new_ns)

        return expr

    # ---------------------------------------------
    # Recorrido línea por línea
    # ---------------------------------------------
    for linea in lineas:
        m = re_asig.match(linea)
        if not m:
            resultado.append(linea)
            continue

        indent, lhs, rhs = m.group(1), m.group(2), m.group(3).strip()

        rhs_simpl = simplificar_expr(rhs)

        # Actualizamos versión de la variable de la izquierda
        versiones[lhs] = versiones.get(lhs, 0) + 1

        usados = set(re_ident.findall(rhs_simpl))

        # Asignación constante (sin variables): no se optimiza
        if not usados:
            nueva_linea = f"{indent}{lhs} = {rhs_simpl};"
            resultado.append(nueva_linea)
            continue

        # Evitar tocar contadores tipo R = R + 1
        if lhs in usados:
            nueva_linea = f"{indent}{lhs} = {rhs_simpl};"
            resultado.append(nueva_linea)
            continue

        # Intentar primero reemplazo de subexpresiones
        rhs_simpl = reemplazar_subexpresiones(rhs_simpl)
        usados = set(re_ident.findall(rhs_simpl))

        snap = snapshot_versiones(usados)
        expr_norm = normalizar_expr(rhs_simpl)
        key = (expr_norm, snap)

        # CSE completo (expresión idéntica previa)
        if key in expr_cache:
            var_prev = expr_cache[key]
            nueva_linea = f"{indent}{lhs} = {var_prev};"
            resultado.append(nueva_linea)
            continue

        # Nueva expresión: la registramos para futuros CSE y subexpresiones
        expr_cache[key] = lhs

        if any(op in rhs_simpl for op in "+-*/%"):
            subexprs.append({
                "lhs": lhs,
                "expr_text": rhs_simpl,
                "expr_ns": sin_espacios(rhs_simpl),
                "snapshot": snap
            })

        nueva_linea = f"{indent}{lhs} = {rhs_simpl};"
        resultado.append(nueva_linea)

    return "\n".join(resultado)


# =========================================================================
#                             PARSER
# =========================================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errores_sintacticos = []

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return {'lex': 'EOF', 'tipo': 'EOF', 'linea': -1, 'col': -1}

    def consume(self, expected_lex=None, expected_tipo=None):
        current = self.peek()
        match = False

        if expected_lex and current['lex'] == expected_lex:
            match = True
        elif expected_tipo and current['tipo'] == expected_tipo:
            match = True
        elif not expected_lex and not expected_tipo:
            match = True

        if match:
            self.pos += 1
            return current
        else:
            self.report_error(
                'ErrSintaxis',
                current['linea'],
                current['lex'],
                f"Se esperaba '{expected_lex or expected_tipo}' pero se encontró '{current['lex']}'"
            )
            if self.pos < len(self.tokens):
                self.pos += 1
            return None

    def report_error(self, token_type, line, lexeme, desc):
        self.errores_sintacticos.append({
            'token': token_type,
            'linea': line,
            'lex': lexeme,
            'desc': desc
        })

    def parse_factor(self):
        token = self.peek()
        if token['tipo'] in ('INT', 'FLOAT', 'STR', 'CHAR'):
            self.consume()
            return Literal(token)
        elif token['tipo'] == 'IDENT':
            self.consume()
            return Ident(token)
        elif token['lex'] == '(':
            self.consume('(')
            node = self.parse_expresion()
            self.consume(')')
            return node
        else:
            self.report_error(
                'ErrSintaxis', token['linea'], token['lex'],
                "Se esperaba un literal, identificador o '('"
            )
            return None

    def parse_term(self):
        node = self.parse_factor()
        while self.peek()['lex'] in ('*', '/', '%'):
            op_token = self.consume()
            right = self.parse_factor()
            if node and right:
                node = BinOp(op_token, node, right)
            else:
                break
        return node

    def parse_expresion(self):
        node = self.parse_term()
        while self.peek()['lex'] in ('+', '-'):
            op_token = self.consume()
            right = self.parse_term()
            if node and right:
                node = BinOp(op_token, node, right)
            else:
                break
        return node

    def parse_asignacion(self, identifier_token):
        self.consume('=')
        _rhs = self.parse_expresion()
        self.consume(';')
        return True

    def parse_sentencia(self):
        token = self.peek()

        # Asignación: id = expr;
        if token['tipo'] == 'IDENT' and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1]['lex'] == '=':
            ident_tok = self.consume()
            return self.parse_asignacion(ident_tok)

        # System.out.println(...)
        elif token['lex'] == 'System.out.println':
            self.consume('System.out.println')
            self.consume('(')
            self.parse_expresion()
            self.consume(')')
            self.consume(';')
            return True

        # Declaraciones: TIPO ...
        elif token['tipo'] == 'TIPO':
            while self.peek()['lex'] not in (';', 'EOF'):
                self.consume()
            if self.peek()['lex'] == ';':
                self.consume(';')
            return True

        # Palabras clave de control
        elif token['tipo'] == 'KW':
            while self.peek()['lex'] not in (';', '{', 'EOF'):
                self.consume()
            if self.peek()['lex'] == '{':
                self.consume('{')
                self.parse_bloque()
                self.consume('}')
            elif self.peek()['lex'] == ';':
                self.consume(';')
            return True

        # Cualquier otra cosa: consumir hasta ';'
        if token['lex'] != 'EOF':
            self.consume()
            while self.peek()['lex'] not in (';', 'EOF'):
                self.consume()
            if self.peek()['lex'] == ';':
                self.consume(';')
            return True

        return None

    def parse_bloque(self):
        while self.peek()['lex'] not in ('}', 'EOF'):
            self.parse_sentencia()

    def parse(self):
        while self.peek()['tipo'] != 'EOF':
            self.parse_sentencia()
        return self.errores_sintacticos


# =========================================================================
#                       HELPERS DE TIPADO
# =========================================================================

def _tipo_literal_token(tok):
    if tok['tipo'] == 'INT':
        return TOK_INT
    if tok['tipo'] == 'FLOAT':
        return TOK_FLOAT
    if tok['tipo'] in ('STR', 'CHAR'):
        return TOK_STR
    return ''


def _en_algun_ambito(nombre, scopes):
    for sc in scopes:
        if nombre in sc and sc[nombre]:
            return True
    return False


# =========================================================================
#     A) TABLA DE SÍMBOLOS (P1)
# =========================================================================

def construir_tabla_simbolos_y_err_p1(texto):
    toks = lex(texto)
    tabla = {}
    errores = []
    scopes = [{}]

    def put(lexema, typ):
        if lexema not in tabla:
            tabla[lexema] = typ

    i, n = 0, len(toks)
    while i < n:
        t = toks[i]

        if t['lex'] == '{':
            scopes.append({})
            put('{', '')
            i += 1
            continue
        if t['lex'] == '}':
            if len(scopes) > 1:
                scopes.pop()
            put('}', '')
            i += 1
            continue

        if t['tipo'] == 'TIPO':
            decl_tok = t['lex'].upper()
            put(decl_tok, '')
            i += 1

            # Caso raro <TIPO> = id ... ;
            if i < n and toks[i]['lex'] == '=':
                put('=', '')
                i += 1
                while i < n and toks[i]['lex'] != ';':
                    if toks[i]['tipo'] == 'IDENT':
                        ident = toks[i]
                        if _en_algun_ambito(ident['lex'], scopes):
                            errores.append({
                                'token': 'ErrDeclDup',
                                'linea': ident['linea'],
                                'lex': ident['lex'],
                                'desc': 'declaracion duplicada'
                            })
                        put(ident['lex'], tabla.get(ident['lex'], ''))
                    else:
                        lit_tk = _tipo_literal_token(toks[i])
                        put(toks[i]['lex'], lit_tk if lit_tk else '')
                    i += 1
                if i < n and toks[i]['lex'] == ';':
                    put(';', '')
                    i += 1
                continue

            # Lista de id's
            while i < n and toks[i]['tipo'] == 'IDENT':
                ident = toks[i]
                i += 1
                cur = scopes[-1]
                if ident['lex'] in cur:
                    errores.append({
                        'token': 'ErrDeclDup',
                        'linea': ident['linea'],
                        'lex': ident['lex'],
                        'desc': 'declaracion duplicada'
                    })
                else:
                    cur[ident['lex']] = decl_tok
                    put(ident['lex'], decl_tok)

                # Asignación de inicialización
                if i < n and toks[i]['lex'] == '=':
                    put('=', '')
                    i += 1
                    while i < n and toks[i]['lex'] not in (',', ';'):
                        lit_tk = _tipo_literal_token(toks[i])
                        put(toks[i]['lex'], lit_tk if lit_tk else '')
                        i += 1

                if i < n and toks[i]['lex'] == ',':
                    put(',', '')
                    i += 1
                    continue
                break

            while i < n and toks[i]['lex'] != ';':
                lit_tk = _tipo_literal_token(toks[i])
                put(toks[i]['lex'], lit_tk if lit_tk else '')
                i += 1
            if i < n and toks[i]['lex'] == ';':
                put(';', '')
                i += 1
            continue

        # Fuera de declaraciones
        lit_tk = _tipo_literal_token(t)
        if lit_tk:
            put(t['lex'], lit_tk)
        else:
            put(t['lex'], '')
        i += 1

    tabla_list = [(lexema, tipo) for lexema, tipo in tabla.items()]
    return toks, tabla_list, scopes, errores


# =========================================================================
#            B) ERRORES SEMÁNTICOS (P2)
# =========================================================================

def recolectar_errores_semanticos(toks, tabla_simbolos):
    global_scope = {}
    for lexema, tipo in tabla_simbolos:
        if re.fullmatch(RE_IDENT, lexema) and tipo in (TOK_INT, TOK_STR, TOK_FLOAT):
            global_scope[lexema] = tipo

    def lookup_stack(idname, stacks):
        for sc in reversed(stacks):
            if idname in sc and sc[idname]:
                return sc[idname]
        return ''

    def tipo_token(tok, stacks):
        if tok['tipo'] in ('INT', 'FLOAT', 'STR', 'CHAR'):
            return _tipo_literal_token(tok)
        if tok['tipo'] == 'IDENT':
            return lookup_stack(tok['lex'], stacks)
        return ''

    errores = []
    scope_stack = [dict(global_scope)]
    loop_depth = 0
    pending_do = 0
    OPA = set(['+', '-', '*', '/'])
    REL = set(['==', '!=', '<=', '>=', '<', '>'])

    def atom(j, emit_opa_errors):
        if j >= len(toks):
            return (('', '', 0), j)
        tk = toks[j]
        if tk['lex'] == '(':
            t1, k, _ops, _opsinfo = expr(j + 1, emit_opa_errors)
            while k < len(toks) and toks[k]['lex'] != ')':
                k += 1
            return (('', t1, tk['linea']), k + 1 if k < len(toks) else k)
        return ((tk['lex'], tipo_token(tk, scope_stack), tk['linea']), j + 1)

    def expr(i, emit_opa_errors=True):
        (lex1, t1, ln1), j = atom(i, emit_opa_errors)
        ops = []
        opsinfo = []
        if lex1 or t1:
            ops.append((lex1, t1, ln1))
        while j < len(toks) and toks[j]['lex'] in OPA:
            op_tok = toks[j]
            j += 1
            (lex2, t2, ln2), j = atom(j, emit_opa_errors)

            prev_t1 = t1
            if t1 in (TOK_INT, TOK_FLOAT) and t2 in (TOK_INT, TOK_FLOAT):
                t1 = TOK_FLOAT if (op_tok['lex'] == '/' or TOK_FLOAT in (t1, t2)) else TOK_INT
            elif t1 == TOK_STR and t2 == TOK_STR and op_tok['lex'] == '+':
                t1 = TOK_STR
            else:
                if emit_opa_errors:
                    culprit = lex2 or lex1 or op_tok['lex']
                    errores.append({
                        'token': 'ErrTipoOPA',
                        'linea': op_tok['linea'],
                        'lex': culprit,
                        'desc': 'Operación incompatible de tipos'
                    })
                t1 = t1 or t2 or TOK_INT
            opsinfo.append((op_tok['lex'], prev_t1, t2, op_tok['linea']))
            if lex2 or t2:
                ops.append((lex2, t2, ln2))
        return t1, j, ops, opsinfo

    i, n = 0, len(toks)
    while i < n:
        tk = toks[i]

        if tk['lex'] == '{':
            scope_stack.append({})
            i += 1
            continue
        if tk['lex'] == '}':
            if len(scope_stack) > 1:
                scope_stack.pop()
            i += 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'do':
            pending_do += 1
            loop_depth += 1
            i += 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'while':
            j = i
            while j < n and toks[j]['lex'] != '(':
                j += 1
            k = j + 1
            has_rel = False
            while k < n and toks[k]['lex'] != ')':
                if toks[k]['lex'] in REL:
                    has_rel = True
                k += 1
            if not has_rel:
                errores.append({
                    'token': 'ErrCond',
                    'linea': tk['linea'],
                    'lex': tk['lex'],
                    'desc': 'Condición de bucle sin operador relacional'
                })

            # while de un do-while
            if pending_do > 0 and i > 0 and toks[i - 1]['lex'] == ';':
                i = k + 2 if k < n and k + 1 < n and toks[k + 1]['lex'] == ';' else k + 1
                pending_do -= 1
                loop_depth = max(0, loop_depth - 1)
                continue

            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'for':
            j = i
            while j < n and toks[j]['lex'] != '(':
                j += 1
            k = j + 1
            has_rel = False
            while k < n and toks[k]['lex'] != ')':
                if toks[k]['lex'] in REL:
                    has_rel = True
                k += 1
            if not has_rel:
                errores.append({
                    'token': 'ErrCond',
                    'linea': tk['linea'],
                    'lex': 'for',
                    'desc': 'Condición de bucle sin operador relacional'
                })
            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] in ('break', 'continue'):
            if loop_depth == 0:
                errores.append({
                    'token': 'ErrLoopCtl',
                    'linea': tk['linea'],
                    'lex': tk['lex'],
                    'desc': 'Sentencia de control de bucle fuera de contexto'
                })
            i += 1
            continue

        if tk['tipo'] == 'IDENT' and i + 1 < n and toks[i + 1]['lex'] == '=':
            lhs = tk
            lhs_t = lookup_stack(lhs['lex'], scope_stack)
            if not lhs_t:
                errores.append({
                    'token': 'ErrUndef',
                    'linea': lhs['linea'],
                    'lex': lhs['lex'],
                    'desc': 'Variable indefinida'
                })

            j = i + 2
            rhs_t, j, rhs_ops, rhs_opsinfo = expr(j, emit_opa_errors=False)

            indefs = set()
            for lexeme, ty, lno in rhs_ops:
                if lexeme and re.fullmatch(RE_IDENT, lexeme) and not ty:
                    if lexeme not in indefs:
                        errores.append({
                            'token': 'ErrUndef',
                            'linea': lno or lhs['linea'],
                            'lex': lexeme,
                            'desc': 'Variable indefinida'
                        })
                        indefs.add(lexeme)

            if lhs_t:
                if lhs_t == TOK_INT:
                    allowed = {TOK_INT}
                elif lhs_t == TOK_FLOAT:
                    allowed = {TOK_INT, TOK_FLOAT}
                else:
                    allowed = {TOK_STR}

                vistos = set()
                hubo_operando_invalido = False
                for lexeme, ty, lno in rhs_ops:
                    if ty and ty not in allowed:
                        key = (lexeme, lno)
                        if key not in vistos:
                            errores.append({
                                'token': f'ErrAsig_{lhs_t}',
                                'linea': lno,
                                'lex': lexeme,
                                'desc': f'Asignación incompatible, se esperaba {lhs_t}'
                            })
                            vistos.add(key)
                            hubo_operando_invalido = True

                if not hubo_operando_invalido and not indefs and rhs_t not in allowed:
                    culprit_lex = rhs_ops[-1][0] if rhs_ops else lhs['lex']
                    culprit_line = rhs_ops[-1][2] if rhs_ops else lhs['linea']
                    errores.append({
                        'token': f'ErrAsig_{lhs_t}',
                        'linea': culprit_line,
                        'lex': culprit_lex,
                        'desc': f'Resultado {rhs_t} ≠ {lhs_t}'
                    })

            while j < n and toks[j]['lex'] != ';':
                j += 1
            i = j + 1 if j < n else j
            continue

        if tk['tipo'] == 'IDENT':
            if not lookup_stack(tk['lex'], scope_stack):
                errores.append({
                    'token': 'ErrUndef',
                    'linea': tk['linea'],
                    'lex': tk['lex'],
                    'desc': 'Variable indefinida'
                })

        i += 1

    return errores


# =========================================================================
#      C) TABLA DE TOKENS OPCIONAL
# =========================================================================

def tabla_tokens(texto: str):
    toks = lex(texto)
    out = []
    for i, t in enumerate(toks, 1):
        out.append((i, t["tipo"], t["lex"], t["linea"], t["col"]))
    return out


# =========================================================================
#      D) TRIPLOS (do-while, &&, ||, TRUE/FALSE en D.O)
# =========================================================================

REL_OPS = {"==", "!=", "<", "<=", ">", ">="}

def _strip_parens(tokens):
    if len(tokens) >= 2 and tokens[0]['lex'] == '(' and tokens[-1]['lex'] == ')':
        depth = 0
        ok = True
        for k, tk in enumerate(tokens):
            if tk['lex'] == '(':
                depth += 1
            elif tk['lex'] == ')':
                depth -= 1
                if depth == 0 and k != len(tokens) - 1:
                    ok = False
                    break
        if ok:
            return tokens[1:-1]
    return tokens


def _split_top(tokens, seps):
    out, depth, last = [], 0, 0
    for i, tk in enumerate(tokens):
        if tk['lex'] == '(':
            depth += 1
        elif tk['lex'] == ')':
            depth -= 1
        elif depth == 0 and tk['lex'] in seps:
            out.append(tokens[last:i])
            last = i + 1
    out.append(tokens[last:])
    return out


def _slice_until(tokens, i, stop_lexes):
    j, depth = i, 0
    while j < len(tokens):
        if tokens[j]['lex'] == '(':
            depth += 1
        elif tokens[j]['lex'] == ')':
            depth -= 1
        if depth == 0 and tokens[j]['lex'] in stop_lexes:
            break
        j += 1
    return tokens[i:j], j


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

    # ---------- Expresiones ----------
    def _gen_factor(self, toks, i):
        if i < len(toks) and toks[i]['lex'] == '-':
            i += 1
            tname, i = self._gen_factor(toks, i)
            t0 = self.newT()
            self.emit('=', t0, '0')
            self.emit('-', t0, tname)
            return t0, i
        if i < len(toks) and toks[i]['lex'] == '(':
            i0 = i
            depth = 0
            while i < len(toks):
                if toks[i]['lex'] == '(':
                    depth += 1
                elif toks[i]['lex'] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            # Generar expr interna sin paréntesis
            inner_start = i0 + 1
            inner_end = i
            tname, _ = self._gen_expr(toks, inner_start)
            return tname, i + 1
        tk = toks[i]
        t = self.newT()
        self.emit('=', t, tk['lex'])
        return t, i + 1

    def _gen_term(self, toks, i):
        leftT, k = self._gen_factor(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('*', '/', '%'):
            op = toks[k]['lex']
            k += 1
            rightT, k = self._gen_factor(toks, k)
            self.emit(op, leftT, rightT)
        return leftT, k

    def _gen_expr(self, toks, i):
        leftT, k = self._gen_term(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('+', '-'):
            op = toks[k]['lex']
            k += 1
            rightT, k = self._gen_term(toks, k)
            self.emit(op, leftT, rightT)
        return leftT, k

    def gen_assignment(self, lhs_lex, rhs_tokens):
        self.temp = 0
        if not rhs_tokens:
            self.emit('=', lhs_lex, '0')
            return
        tname, _ = self._gen_expr(rhs_tokens, 0)
        self.emit('=', lhs_lex, tname)

    # ---------- Relacional: devuelve (firstN, relN) ----------
    def gen_rel_row(self, rel_tokens):
        """
        Genera el cálculo de temporales del factor y la fila relacional.
        Devuelve (firstN, relN):
          - firstN: N de la PRIMERA instrucción emitida al comenzar este factor/término
          - relN:   N de la fila del relacional (<, >, ==, etc.)
        """
        self.temp = 0
        firstN = self.N + 1

        depth, pos, relop = 0, -1, None
        for j, tk in enumerate(rel_tokens):
            if tk['lex'] == '(':
                depth += 1
            elif tk['lex'] == ')':
                depth -= 1
            elif depth == 0 and tk['lex'] in REL_OPS:
                pos = j
                relop = tk['lex']
                break

        if relop is None:
            t, _ = self._gen_expr(rel_tokens, 0)
            relN = self.emit('!=', t, '0')
            return firstN, relN

        left = rel_tokens[:pos]
        right = rel_tokens[pos + 1:]
        tL, _ = self._gen_expr(left, 0)
        tR, _ = self._gen_expr(right, 0)
        relN = self.emit(relop, tL, tR)
        return firstN, relN

    def gen_condition_do_while(self, cond_tokens, N_body):
        """
        Por cada relacional emite EXACTAMENTE 3 filas:
          [REL]      -> O:<op>, DO:<LHS>, DF:<RHS>
          "" TRUE    -> DO:TRUE,  DF:<destino>
          "" FALSE   -> DO:FALSE, DF:<destino>

        Cortocircuito:
        - FALSE de OR  -> primer "=" del siguiente término.
        - TRUE de AND  -> primer "=" del siguiente factor.
        - FALSE del último término -> END del do actual (se completa luego).
        """
        cond_tokens = _strip_parens(cond_tokens)
        or_terms = _split_top(cond_tokens, {'||'})

        pending_false_to_next_term = []
        pending_to_end = []

        for t_idx, term in enumerate(or_terms):
            term = _strip_parens(term)
            and_factors = _split_top(term, {'&&'})

            pending_true_to_next_factor = []
            first_firstN_of_term = None
            local_false_of_term = []

            for f_idx, factor in enumerate(and_factors):
                factor = _strip_parens(factor)

                firstN, relN = self.gen_rel_row(factor)

                if first_firstN_of_term is None:
                    first_firstN_of_term = firstN
                    for row_idx in pending_false_to_next_term:
                        self.rows[row_idx]["DF"] = str(first_firstN_of_term)
                    pending_false_to_next_term = []

                for row_idx in pending_true_to_next_factor:
                    self.rows[row_idx]["DF"] = str(firstN)
                pending_true_to_next_factor = []

                last_factor = (f_idx == len(and_factors) - 1)
                last_term = (t_idx == len(or_terms) - 1)

                if len(and_factors) == 1:
                    # Un solo factor en el término
                    self.emit("", "TRUE", str(N_body))
                    self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                    idx_false = len(self.rows) - 1
                    (pending_to_end if last_term else local_false_of_term).append(idx_false)
                else:
                    if not last_factor:
                        # AND intermedio
                        self.emit("", "TRUE", "PENDING_NEXT_FACTOR")
                        idx_true = len(self.rows) - 1
                        pending_true_to_next_factor.append(idx_true)

                        self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)
                    else:
                        # Último factor del término
                        self.emit("", "TRUE", str(N_body))
                        self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)

            pending_false_to_next_term.extend(local_false_of_term)

        # Los false que iban al siguiente término del último término => END
        for idx in pending_false_to_next_term:
            self.rows[idx]["DF"] = "PENDING_END"

        # TRUE pendientes a NEXT_FACTOR se resuelven al body (caso límite)
        for r in self.rows:
            if r["DO"] == "TRUE" and r["DF"] == "PENDING_NEXT_FACTOR":
                r["DF"] = str(N_body)

        return pending_to_end


def generar_triplos(texto: str):
    toks = lex(texto)
    n = len(toks)

    em = TriploEmitter()
    i = 0

    def parse_assignment(idx):
        lhs = toks[idx]["lex"]
        j = idx + 2
        rhs, k = _slice_until(toks, j, {';', '}'})
        em.gen_assignment(lhs, rhs)
        idx2 = k
        if idx2 < n and toks[idx2]["lex"] == ';':
            idx2 += 1
        return idx2

    def parse_do(idx):
        assert toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "do"
        idx += 1
        if idx >= n or toks[idx]["lex"] != "{":
            return idx
        idx += 1

        body_start_N = em.N + 1

        while idx < n and toks[idx]["lex"] != "}":
            if idx + 1 < n and toks[idx]["tipo"] == "IDENT" and toks[idx + 1]["lex"] == "=":
                idx = parse_assignment(idx)
            elif toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "do":
                idx = parse_do(idx)
            else:
                idx += 1

        if idx < n and toks[idx]["lex"] == "}":
            idx += 1

        if idx < n and toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "while":
            idx += 1
            if idx < n and toks[idx]["lex"] == "(":
                j, depth = idx + 1, 1
                while j < n and depth > 0:
                    if toks[j]["lex"] == "(":
                        depth += 1
                    elif toks[j]["lex"] == ")":
                        depth -= 1
                    j += 1
                cond_tokens = toks[idx + 1:j - 1] if j - 1 > idx + 1 else []
                pending_end_rows = em.gen_condition_do_while(cond_tokens, body_start_N)
                idx = j
                if idx < n and toks[idx]["lex"] == ";":
                    idx += 1

                local_end = em.N + 1
                for ridx in pending_end_rows:
                    em.rows[ridx]["DF"] = str(local_end)
        return idx

    while i < n:
        tk = toks[i]
        if tk["tipo"] == "KW" and tk["lex"] == "do":
            i = parse_do(i)
            continue
        if i + 1 < n and tk["tipo"] == "IDENT" and toks[i + 1]["lex"] == "=":
            i = parse_assignment(i)
            continue
        i += 1

    return [(r["N"], r["O"], r["DO"], r["DF"]) for r in em.rows]


# =========================================================================
#            E) API PARA LA GUI
# =========================================================================

def analizar_dos_pasadas(texto):
    toks = lex(texto)

    parser = Parser(toks)
    err_p_sintaxis = parser.parse()

    _toks_aux, tabla_simbolos, _scopes, err_p1 = construir_tabla_simbolos_y_err_p1(texto)
    err_p2 = recolectar_errores_semanticos(toks, tabla_simbolos)

    errores = []
    errores.extend(err_p_sintaxis)
    errores.extend(err_p1)
    errores.extend(err_p2)

    for e in errores:
        e.setdefault('token', 'ErrSem')
        e.setdefault('linea', 0)
        e.setdefault('lex', '')
        e.setdefault('desc', 'Error')

    errores.sort(key=lambda e: (e.get('linea', 0), e.get('col', 0) if 'col' in e else 0))
    return tabla_simbolos, errores


def analizar_y_triplos(texto):
    tabla, errores = analizar_dos_pasadas(texto)
    try:
        triplos = generar_triplos(texto)
    except Exception:
        triplos = []
    return tabla, errores, triplos


def analizar_y_generar_ensamblador(texto: str):
    """
    Envuelve todo:
    - Usa analizar_y_triplos para obtener tabla, errores y triplos.
    - Traduce esos triplos a ASM 8086 usando codegen8086.
    """
    tabla, errores, triplos = analizar_y_triplos(texto)
    asm = generar_codigo_ensamblador(triplos)
    return tabla, errores, triplos, asm
