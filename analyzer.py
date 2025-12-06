# analyzer.py — Unión estable + FIX de saltos:
#   FALSE de OR salta al PRIMER "=" del siguiente término,
#   TRUE intermedio de AND salta al PRIMER "=" del siguiente factor.
#
# API:
#   analizar_dos_pasadas(texto) -> (tabla_simbolos, errores)
#   analizar_y_triplos(texto)   -> (tabla_simbolos, errores, triplos)
#   tabla_tokens(texto)         -> [(#, tipo, lexema, linea, col)]

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
        self.op    = op_token['lex']
        self.left  = left
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
KW_SET = {'if','else','while','for','do','break','continue','func','return'}

def lex(texto):
    """
    Devuelve lista de dicts: {'tipo','lex','linea','col'}.
    tipo ∈ { TIPO, IDENT, INT, FLOAT, STR, CHAR, KW, OP, DESCONOCIDO }
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
            tipo = 'KW'; lexema = 'System.out.println'
        elif re.fullmatch(RE_IDENT, lexema):
            lower = lexema.lower()
            if lower in KW_SET:
                tipo = 'KW'; lexema = lower
            else:
                tipo = 'IDENT'
        else:
            tipo = 'OP'

        toks.append({'tipo': tipo, 'lex': lexema, 'linea': linea, 'col': col})

    # EOF sintético para el parser (no se muestra en GUI)
    toks.append({'tipo': 'EOF', 'lex': 'EOF', 'linea': -1, 'col': -1})
    return toks


# =========================================================================
#                             PARSER
# =========================================================================

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errores_sintacticos = []

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else {'lex':'EOF','tipo':'EOF','linea':-1,'col':-1}

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
            self.report_error('ErrSintaxis', current['linea'], current['lex'],
                              f"Se esperaba '{expected_lex or expected_tipo}' pero se encontró '{current['lex']}'")
            if self.pos < len(self.tokens):
                self.pos += 1
            return None

    def report_error(self, token_type, line, lexeme, desc):
        self.errores_sintacticos.append({'token': token_type, 'linea': line, 'lex': lexeme, 'desc': desc})

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
            self.report_error('ErrSintaxis', token['linea'], token['lex'],
                              "Se esperaba un literal, identificador o '('")
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
        if token['tipo'] == 'IDENT' and self.tokens[self.pos + 1]['lex'] == '=':
            ident_tok = self.consume()
            return self.parse_asignacion(ident_tok)

        elif token['lex'] == 'System.out.println':
            self.consume('System.out.println')
            self.consume('('); self.parse_expresion(); self.consume(')'); self.consume(';')
            return True

        elif token['tipo'] == 'TIPO':
            while self.peek()['lex'] != ';' and self.peek()['lex'] != 'EOF':
                self.consume()
            if self.peek()['lex'] == ';': self.consume(';')
            return True

        elif token['tipo'] == 'KW':
            while self.peek()['lex'] not in (';', '{', 'EOF'):
                self.consume()
            if self.peek()['lex'] == '{':
                self.consume('{'); self.parse_bloque(); self.consume('}')
            elif self.peek()['lex'] == ';':
                self.consume(';')
            return True

        if token['lex'] != 'EOF':
            self.consume()
            while self.peek()['lex'] not in (';', 'EOF'):
                self.consume()
            if self.peek()['lex'] == ';': self.consume(';')
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
    if tok['tipo'] == 'INT':   return TOK_INT
    if tok['tipo'] == 'FLOAT': return TOK_FLOAT
    if tok['tipo'] in ('STR','CHAR'): return TOK_STR
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
    scopes = [ {} ]  # pila de ámbitos

    def put(lexema, typ):
        if lexema not in tabla:
            tabla[lexema] = typ

    i, n = 0, len(toks)
    while i < n:
        t = toks[i]

        if t['lex'] == '{':
            scopes.append({})
            put('{', ''); i += 1; continue
        if t['lex'] == '}':
            if len(scopes) > 1: scopes.pop()
            put('}', ''); i += 1; continue

        if t['tipo'] == 'TIPO':
            decl_tok = t['lex'].upper()  # E$|C$|F$
            put(decl_tok, '')
            i += 1

            if i < n and toks[i]['lex'] == '=':
                put('=', ''); i += 1
                while i < n and toks[i]['lex'] != ';':
                    if toks[i]['tipo'] == 'IDENT':
                        ident = toks[i]
                        if _en_algun_ambito(ident['lex'], scopes):
                            errores.append({'token':'ErrDeclDup','linea':ident['linea'],'lex':ident['lex'],'desc':'declaracion duplicada'})
                        put(ident['lex'], tabla.get(ident['lex'], ''))
                    else:
                        lit_tk = _tipo_literal_token(toks[i])
                        put(toks[i]['lex'], lit_tk if lit_tk else '')
                    i += 1
                if i < n and toks[i]['lex'] == ';':
                    put(';',''); i += 1
                continue

            while i < n and toks[i]['tipo'] == 'IDENT':
                ident = toks[i]; i += 1
                cur = scopes[-1]
                if ident['lex'] in cur:
                    errores.append({'token':'ErrDeclDup','linea':ident['linea'],'lex':ident['lex'],'desc':'declaracion duplicada'})
                else:
                    cur[ident['lex']] = decl_tok
                    put(ident['lex'], decl_tok)

                if i < n and toks[i]['lex'] == '=':
                    put('=',''); i += 1
                    while i < n and toks[i]['lex'] not in (',',';'):
                        lit_tk = _tipo_literal_token(toks[i])
                        put(toks[i]['lex'], lit_tk if lit_tk else '')
                        i += 1

                if i < n and toks[i]['lex'] == ',':
                    put(',',''); i += 1
                    continue
                break

            while i < n and toks[i]['lex'] != ';':
                lit_tk = _tipo_literal_token(toks[i])
                put(toks[i]['lex'], lit_tk if lit_tk else '')
                i += 1
            if i < n and toks[i]['lex'] == ';':
                put(';',''); i += 1
            continue

        lit_tk = _tipo_literal_token(t)
        if lit_tk:
            put(t['lex'], lit_tk)
        else:
            put(t['lex'], '')
        i += 1

    tabla_list = [(lexema, tipo) for lexema, tipo in tabla.items()]
    return toks, tabla_list, scopes, errores  # errores P1


# =========================================================================
#            B) ERRORES (P2)
# =========================================================================

def recolectar_errores_semanticos(toks, tabla_simbolos):
    global_scope = {}
    for lexema, tipo in tabla_simbolos:
        if re.fullmatch(RE_IDENT, lexema) and tipo in (TOK_INT, TOK_STR, TOK_FLOAT):
            global_scope[lexema] = tipo

    def lookup(idname, stacks):
        for sc in reversed(stacks):
            if idname in sc and sc[idname]:
                return sc[idname]
        return ''

    def tipo_token(tok, stacks):
        if tok['tipo'] in ('INT','FLOAT','STR','CHAR'):
            return _tipo_literal_token(tok)
        if tok['tipo'] == 'IDENT':
            return lookup(tok['lex'], stacks)
        return ''

    errores = []
    scope_stack = [dict(global_scope)]
    loop_depth = 0
    pending_do = 0
    OPA = set(['+','-','*','/'])
    REL = set(['==','!=','<=','>=','<','>'])

    def atom(j, emit_opa_errors):
        if j >= len(toks): return (('', '', 0), j)
        tk = toks[j]
        if tk['lex'] == '(':
            t1, k, _ops, _opsinfo = expr(j+1, emit_opa_errors)
            while k < len(toks) and toks[k]['lex'] != ')': k += 1
            return (('', t1, tk['linea']), k+1 if k < len(toks) else k)
        return ((tk['lex'], tipo_token(tk, scope_stack), tk['linea']), j+1)

    def expr(i, emit_opa_errors=True):
        (lex1, t1, ln1), j = atom(i, emit_opa_errors)
        ops = []
        opsinfo = []
        if lex1 or t1: ops.append((lex1, t1, ln1))
        while j < len(toks) and toks[j]['lex'] in OPA:
            op_tok = toks[j]; j += 1
            (lex2, t2, ln2), j = atom(j, emit_opa_errors)

            prev_t1 = t1
            if t1 in (TOK_INT, TOK_FLOAT) and t2 in (TOK_INT, TOK_FLOAT):
                t1 = TOK_FLOAT if (op_tok['lex'] == '/' or TOK_FLOAT in (t1, t2)) else TOK_INT
            elif t1 == TOK_STR and t2 == TOK_STR and op_tok['lex'] == '+':
                t1 = TOK_STR
            else:
                if emit_opa_errors:
                    culprit = lex2 or lex1 or op_tok['lex']
                    errores.append({'token':'ErrTipoOPA','linea':op_tok['linea'],'lex':culprit,'desc':'Operación incompatible de tipos'})
                t1 = t1 or t2 or TOK_INT
            opsinfo.append((op_tok['lex'], prev_t1, t2, op_tok['linea']))
            if lex2 or t2: ops.append((lex2, t2, ln2))
        return t1, j, ops, opsinfo

    i, n = 0, len(toks)
    while i < n:
        tk = toks[i]

        if tk['lex'] == '{':
            scope_stack.append({}); i += 1; continue
        if tk['lex'] == '}':
            if len(scope_stack) > 1: scope_stack.pop(); i += 1; continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'do':
            pending_do += 1; loop_depth += 1; i += 1; continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'while':
            j = i
            while j < n and toks[j]['lex'] != '(': j += 1
            k = j + 1
            has_rel = False
            while k < n and toks[k]['lex'] != ')':
                if toks[k]['lex'] in REL: has_rel = True
                k += 1
            if not has_rel:
                errores.append({'token':'ErrCond','linea':tk['linea'],'lex':tk['lex'],'desc':'Condición de bucle sin operador relacional'})

            if pending_do > 0 and i > 0 and toks[i-1]['lex'] == ';':
                i = k + 2 if k < n and toks[k+1]['lex'] == ';' else k+1
                pending_do -= 1
                loop_depth = max(0, loop_depth - 1)
                continue

            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'for':
            j = i
            while j < n and toks[j]['lex'] != '(': j += 1
            k = j + 1
            has_rel = False
            while k < n and toks[k]['lex'] != ')':
                if toks[k]['lex'] in REL: has_rel = True
                k += 1
            if not has_rel:
                errores.append({'token':'ErrCond','linea':tk['linea'],'lex':'for','desc':'Condición de bucle sin operador relacional'})
            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] in ('break','continue'):
            if loop_depth == 0:
                errores.append({'token':'ErrLoopCtl','linea':tk['linea'],'lex':tk['lex'],'desc':'Sentencia de control de bucle fuera de contexto'})
            i += 1
            continue

        if tk['tipo'] == 'IDENT' and i+1 < n and toks[i+1]['lex'] == '=':
            lhs = tk
            def lookup_stack(name):
                for sc in reversed(scope_stack):
                    if name in sc and sc[name]: return sc[name]
                return ''
            lhs_t = lookup_stack(lhs['lex'])
            if not lhs_t:
                errores.append({'token':'ErrUndef','linea':lhs['linea'],'lex':lhs['lex'],'desc':'Variable indefinida'})

            j = i + 2
            rhs_t, j, rhs_ops, rhs_opsinfo = expr(j, emit_opa_errors=False)

            indefs = set()
            for lexeme, ty, lno in rhs_ops:
                if lexeme and re.fullmatch(RE_IDENT, lexeme) and not ty:
                    if lexeme not in indefs:
                        errores.append({'token':'ErrUndef','linea':lno or lhs['linea'],'lex':lexeme,'desc':'Variable indefinida'})
                        indefs.add(lexeme)

            if lhs_t:
                if lhs_t == TOK_INT: allowed = {TOK_INT}
                elif lhs_t == TOK_FLOAT: allowed = {TOK_INT, TOK_FLOAT}
                else: allowed = {TOK_STR}

                vistos = set()
                hubo_operando_invalido = False
                for lexeme, ty, lno in rhs_ops:
                    if ty and ty not in allowed:
                        key = (lexeme, lno)
                        if key not in vistos:
                            errores.append({'token':f'ErrAsig_{lhs_t}','linea':lno,'lex':lexeme,'desc':f'Asignación incompatible, se esperaba {lhs_t}'})
                            vistos.add(key); hubo_operando_invalido = True
                if not hubo_operando_invalido and not indefs and rhs_t not in allowed:
                    culprit_lex  = rhs_ops[-1][0] if rhs_ops else lhs['lex']
                    culprit_line = rhs_ops[-1][2] if rhs_ops else lhs['linea']
                    errores.append({'token':f'ErrAsig_{lhs_t}','linea':culprit_line,'lex':culprit_lex,'desc':f'Resultado {rhs_t} ≠ {lhs_t}'})

            while j < n and toks[j]['lex'] != ';': j += 1
            i = j + 1 if j < n else j
            continue

        if tk['tipo'] == 'IDENT':
            def lookup_stack(name):
                for sc in reversed(scope_stack):
                    if name in sc and sc[name]: return sc[name]
                return ''
            if not lookup_stack(tk['lex']):
                errores.append({'token':'ErrUndef','linea':tk['linea'],'lex':tk['lex'],'desc':'Variable indefinida'})

        i += 1

    return errores


# =========================================================================
#            C) TABLA DE TOKENS (para GUI pestaña “Símbolos”)
# =========================================================================

def tabla_tokens(texto: str):
    toks = lex(texto)
    if toks and toks[-1]['tipo'] == 'EOF':  # ocultar EOF sintético
        toks = toks[:-1]
    out = []
    for i, t in enumerate(toks, 1):
        out.append((i, t["tipo"], t["lex"], t["linea"], t["col"]))
    return out


# =========================================================================
#            D) TRIPLOS (con FIX de saltos entre términos/factores)
# =========================================================================

REL_OPS = {"==","!=", "<","<=",">",">="}
OPA     = {"+","-","*","/","%"}

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
        tk = toks[i]
        t = self.newT(); self.emit('=', t, tk['lex'])
        return t, i + 1

    def _gen_term(self, toks, i):
        leftT, k = self._gen_factor(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('*','/','%'):
            op = toks[k]['lex']; k += 1
            rightT, k = self._gen_factor(toks, k)
            self.emit(op, leftT, rightT)
        return leftT, k

    def _gen_expr(self, toks, i):
        leftT, k = self._gen_term(toks, i)
        while k < len(toks) and toks[k]['lex'] in ('+','-'):
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

    # ---------- FIX: Relacional devuelve (firstN, relN) ----------
    def gen_rel_row(self, rel_tokens):
        """
        Genera el cálculo de temporales del factor y la fila relacional.
        Devuelve (firstN, relN):
          - firstN: N de la PRIMERA instrucción emitida al comenzar este factor/término
          - relN:   N de la fila del relacional (<, >, ==, etc.)
        """
        self.temp = 0
        firstN = self.N + 1  # primera N que se generará a partir de aquí

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
            relN = self.emit('!=', t, '0')
            return firstN, relN

        left  = rel_tokens[:pos]
        right = rel_tokens[pos+1:]
        tL, _ = self._gen_expr(left, 0)
        tR, _ = self._gen_expr(right, 0)
        relN = self.emit(relop, tL, tR)
        return firstN, relN

    def gen_condition_do_while(self, cond_tokens, N_body):
        """
        Por cada relacional emite EXACTAMENTE 3 filas:
          [REL]  -> O:<op>, DO:<LHS>, DF:<RHS>
          "" TRUE  -> DO:TRUE,  DF:<destino>
          "" FALSE -> DO:FALSE, DF:<destino>
        FALSE de OR  -> primer "=" del siguiente término (no al relacional).
        TRUE de AND  -> primer "=" del siguiente factor.
        """
        cond_tokens = _strip_parens(cond_tokens)
        or_terms = _split_top(cond_tokens, {'||'})

        pending_false_to_next_term = []  # filas FALSE que irán al firstN del siguiente término
        pending_to_end = []              # filas cuyo DF debe parcharse al END local

        for t_idx, term in enumerate(or_terms):
            term = _strip_parens(term)
            and_factors = _split_top(term, {'&&'})

            pending_true_to_next_factor = []  # filas TRUE que irán al firstN del siguiente factor
            first_firstN_of_term = None
            local_false_of_term = []

            for f_idx, factor in enumerate(and_factors):
                factor = _strip_parens(factor)

                # [REL] de este factor
                firstN, relN = self.gen_rel_row(factor)

                # Primer factor del término: resolver FALSE pendientes del término anterior
                if first_firstN_of_term is None:
                    first_firstN_of_term = firstN
                    for row_idx in pending_false_to_next_term:
                        self.rows[row_idx]["DF"] = str(first_firstN_of_term)
                    pending_false_to_next_term = []

                # Resolver TRUE pendientes (AND) hacia el INICIO (firstN) de este factor
                for row_idx in pending_true_to_next_factor:
                    self.rows[row_idx]["DF"] = str(firstN)
                pending_true_to_next_factor = []

                last_factor = (f_idx == len(and_factors) - 1)
                last_term   = (t_idx == len(or_terms) - 1)

                if len(and_factors) == 1:
                    self.emit("", "TRUE",  str(N_body))  # BODY
                    self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                    idx_false = len(self.rows) - 1
                    (pending_to_end if last_term else local_false_of_term).append(idx_false)
                else:
                    if not last_factor:
                        # AND: TRUE -> firstN del siguiente factor
                        self.emit("", "TRUE",  "PENDING_NEXT_FACTOR")
                        idx_true = len(self.rows) - 1
                        pending_true_to_next_factor.append(idx_true)
                        # AND: FALSE -> siguiente término o END
                        self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)
                    else:
                        # último factor del término
                        self.emit("", "TRUE",  str(N_body))  # BODY
                        self.emit("", "FALSE", "PENDING_END" if last_term else "PENDING_NEXT_TERM")
                        idx_false = len(self.rows) - 1
                        (pending_to_end if last_term else local_false_of_term).append(idx_false)

            # al cerrar término, sus FALSE locales → firstN del siguiente término
            pending_false_to_next_term.extend(local_false_of_term)

        # FALSE sin siguiente término ⇒ END local
        for idx in pending_false_to_next_term:
            self.rows[idx]["DF"] = "PENDING_END"

        # TRUE→NEXT_FACTOR sin resolver ⇒ BODY del do actual
        for r in self.rows:
            if r["DO"] == "TRUE" and r["DF"] == "PENDING_NEXT_FACTOR":
                r["DF"] = str(N_body)

        return pending_to_end


def generar_triplos(texto: str):
    toks = lex(texto)
    if toks and toks[-1]['tipo'] == 'EOF':
        toks = toks[:-1]

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
        assert toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "do"
        idx += 1
        if idx >= n or toks[idx]["lex"] != "{":
            return idx
        idx += 1

        body_start_N = em.N + 1  # primera N del cuerpo

        while idx < n and toks[idx]["lex"] != "}":
            if idx+1 < n and toks[idx]["tipo"] == "IDENT" and toks[idx+1]["lex"] == "=":
                idx = parse_assignment(idx)
            elif toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "do":
                idx = parse_do(idx)
            else:
                idx += 1

        if idx < n and toks[idx]["lex"] == "}": idx += 1

        if idx < n and toks[idx]["tipo"] == "KW" and toks[idx]["lex"] == "while":
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

                # END local del do actual
                local_end = em.N + 1
                for ridx in pending_end_rows:
                    em.rows[ridx]["DF"] = str(local_end)
        return idx

    while i < n:
        tk = toks[i]
        if tk["tipo"] == "KW" and tk["lex"] == "do":
            i = parse_do(i); continue
        if i+1 < n and tk["tipo"] == "IDENT" and toks[i+1]["lex"] == "=":
            i = parse_assignment(i); continue
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
# -------------------------------------------------------------------------
#   OPTIMIZACIÓN LOCAL DE CÓDIGO FUENTE (ANTES DE TODO LO DEMÁS)
#   Regla: instrucciones que se repiten sin modificación de sus valores
# -------------------------------------------------------------------------

def optimizar_codigo_fuente(texto: str) -> str:
    """
    Optimización local muy simple basada en:
      'Instrucciones que se repiten sin haber tenido modificación
       alguna en uno de sus valores'.

    Idea:
      Si encontramos dos asignaciones del tipo:

          x = expr;
          ...
          y = expr;

      y ninguna variable de 'expr' ha sido modificada entre ambas
      asignaciones, reescribimos la segunda como:

          y = x;

      De esta forma no volvemos a calcular la expresión completa,
      solo copiamos el resultado.

    NOTA:
      - Trabajamos línea por línea.
      - No tocamos declaraciones E$/F$/C$ ni líneas sin forma 'id = expr;'.
      - No modificamos la estructura de control (do/while, llaves, etc.).
    """
    lines = texto.splitlines()
    last_assign = {}   # nombre -> índice de línea (1-based)
    seen_exprs = {}    # expr_key -> (line_idx, lhs, vars)
    optimized = []

    for idx, line in enumerate(lines, start=1):
        original = line
        stripped = line.strip()

        # Saltar líneas vacías o posibles comentarios
        if not stripped or stripped.startswith(("//", "/*", "*")):
            optimized.append(original)
            continue

        # No optimizar declaraciones de tipo (E$, F$, C$ ...)
        if stripped.startswith(("E$", "F$", "C$")):
            optimized.append(original)
            continue

        # Patrón simple de asignación:   <ws> id = ... ; <ws>
        m = re.match(r'^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);(\s*)$', line)
        if not m:
            # No es una asignación "normal": dejamos la línea igual
            optimized.append(original)
            continue

        leading_ws, lhs, rhs_str, trailing_ws = m.groups()

        # Tokenizar solo el RHS para obtener una clave de expresión y sus variables
        rhs_tokens = [t for t in lex(rhs_str + ';') if t['lex'] != ';' and t['tipo'] != 'EOF']
        expr_key = " ".join(t['lex'] for t in rhs_tokens)
        vars_in_expr = {t['lex'] for t in rhs_tokens if t['tipo'] == 'IDENT'}

        replacement = None

        # ¿Ya vimos esta expresión antes?
        if expr_key in seen_exprs and vars_in_expr:
            first_line, first_lhs, expr_vars = seen_exprs[expr_key]
            # Verificar que ninguna variable usada en la expresión haya cambiado
            ok = True
            for v in expr_vars:
                if last_assign.get(v, first_line) > first_line:
                    ok = False
                    break
            if ok and first_lhs != lhs:
                # Podemos reutilizar el resultado anterior
                replacement = f"{leading_ws}{lhs} = {first_lhs};{trailing_ws}"

        # Si no se pudo reutilizar, registramos esta expresión como nueva
        if replacement is None:
            seen_exprs[expr_key] = (idx, lhs, vars_in_expr)
            replacement = f"{leading_ws}{lhs} = {rhs_str};{trailing_ws}"

        optimized.append(replacement)
        # Registrar la escritura del LHS
        last_assign[lhs] = idx

    return "\n".join(optimized)
