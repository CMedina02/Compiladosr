# analyzer.py
import re
from rules import (
    TOKEN_PATTERN, idx_a_line_col,
    RE_TIPO, RE_IDENT, RE_FLOAT, RE_INT, RE_STR, RE_CHAR, RE_SYSPRINT,
    TOK_INT, TOK_STR, TOK_FLOAT
)


#                       LEXER (tokens con posiciones)
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
            tipo = 'KW'
            lexema = 'System.out.println'
        elif re.fullmatch(RE_IDENT, lexema):
            lower = lexema.lower()
            if lower in ('if','else','while','for','do','break','continue','func','return'):
                tipo = 'KW'; lexema = lower
            else:
                tipo = 'IDENT'
        else:
            tipo = 'OP'
        toks.append({'tipo': tipo, 'lex': lexema, 'linea': linea, 'col': col})
    return toks


#                HELPERS DE TIPADO (E$ / C$ / F$)
def _tipo_literal_token(tok):
    """Mapea literales léxicos a nuestros tokens E$/C$/F$."""
    if tok['tipo'] == 'INT':   return TOK_INT
    if tok['tipo'] == 'FLOAT': return TOK_FLOAT
    if tok['tipo'] in ('STR','CHAR'): return TOK_STR
    return ''

def _en_algun_ambito(nombre, scopes):
    """True si el identificador ya fue declarado en algún ámbito de la pila."""
    for sc in scopes:
        if nombre in sc and sc[nombre]:
            return True
    return False



#   A) LLENADO DE LA TABLA DE SÍMBOLOS (Primera pasada)
def construir_tabla_simbolos_y_err_p1(texto):
    """
    - Construye la TABLA DE SÍMBOLOS con TODOS los lexemas (sin repetidos).
    - Tipifica IDs declarados y literales (E$/C$/F$).
    - Detecta 1ª pasada: ErrDeclDup en redeclaración o en '<TIPO> = id, ... ;'
    """
    toks = lex(texto)
    tabla = {}
    errores = []
    scopes = [ {} ]  # pila de ámbitos; cada dict: {ident: 'E$'|'C$'|'F$'}

    def put(lexema, typ):
        if lexema not in tabla:
            tabla[lexema] = typ

    i, n = 0, len(toks)
    while i < n:
        t = toks[i]

        if t['lex'] == '{':
            scopes.append({})
            put('{', '')
            i += 1; continue
        if t['lex'] == '}':
            if len(scopes) > 1: scopes.pop()
            put('}', '')
            i += 1; continue

        if t['tipo'] == 'TIPO':
            decl_tok = t['lex'].upper()  # E$|C$|F$
            put(decl_tok, '')
            i += 1

            #  PATRÓN: <TIPO> = id (, id)* ;  (re-tipeo inválido) 
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

            while i < n and toks[i]['tipo'] == 'IDENT':
                ident = toks[i]; i += 1
                cur = scopes[-1]
                if ident['lex'] in cur:
                    errores.append({
                        'token': 'ErrDeclDup',
                        'linea': ident['linea'], 'lex': ident['lex'],
                        'desc': 'declaracion duplicada'
                    })
                else:
                    cur[ident['lex']] = decl_tok
                    put(ident['lex'], decl_tok)

                # ¿coma o asignación de inicialización?
                if i < n and toks[i]['lex'] == '=':
                    put('=', ''); i += 1
                    # consumir la expresión de inicialización hasta coma/;
                    while i < n and toks[i]['lex'] not in (',', ';'):
                        lit_tk = _tipo_literal_token(toks[i])
                        put(toks[i]['lex'], lit_tk if lit_tk else '')
                        i += 1

                if i < n and toks[i]['lex'] == ',':
                    put(',', ''); i += 1
                    continue
                break

            # consumir hasta ';'
            while i < n and toks[i]['lex'] != ';':
                lit_tk = _tipo_literal_token(toks[i])
                put(toks[i]['lex'], lit_tk if lit_tk else '')
                i += 1
            if i < n and toks[i]['lex'] == ';':
                put(';', ''); i += 1
            continue

        # fuera de declaraciones: agregar a la tabla
        lit_tk = _tipo_literal_token(t)
        if lit_tk:
            put(t['lex'], lit_tk)
        else:
            put(t['lex'], '')

        i += 1

    tabla_list = [(lexema, tipo) for lexema, tipo in tabla.items()]
    return toks, tabla_list, scopes, errores  # errores de PRIMERA pasada



#   B) LLENADO DE LA TABLA DE ERRORES (Segunda pasada)
def recolectar_errores_semanticos(texto, toks, tabla_simbolos):
    """
    Genera la TABLA DE ERRORES (segunda pasada):
      - ErrUndef (uso de ident no declarado)
      - ErrAsig_<LHS> (incompatibilidad en asignación; múltiples culpables)
      - ErrTipoOPA (operación incompatible fuera de asignación)
      - ErrCond (condiciones sin relacional)
      - ErrLoopCtl (break/continue fuera de bucle)
    """
    # scope global desde tabla de símbolos (IDs con tipo)
    global_scope = {}
    for lexema, tipo in tabla_simbolos:
        if re.fullmatch(RE_IDENT, lexema) and tipo in (TOK_INT, TOK_STR, TOK_FLOAT):
            global_scope[lexema] = tipo

    def lookup(idname, stacks):
        for sc in reversed(stacks):
            if idname in sc and sc[idname]:
                return sc[idname]
        return ''  # no declarado

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

    #  evaluador simple de expresiones (izq→der) 
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

            prev_t1 = t1  # tipo a la izquierda antes de aplicar el operador

            if t1 in (TOK_INT, TOK_FLOAT) and t2 in (TOK_INT, TOK_FLOAT):
                t1 = TOK_FLOAT if (op_tok['lex'] == '/' or TOK_FLOAT in (t1, t2)) else TOK_INT
            elif t1 == TOK_STR and t2 == TOK_STR and op_tok['lex'] in ('+','-'):
                t1 = TOK_STR
            else:
                if emit_opa_errors:
                    culprit = lex2 or lex1 or op_tok['lex']
                    errores.append({
                        'token': 'ErrTipoOPA',
                        'linea': op_tok['linea'],
                        'lex': culprit,
                        'desc': 'Incompatibilidad de tipos'
                    })
                t1 = t1 or t2 or TOK_INT

            # registrar información del operador usado
            opsinfo.append((op_tok['lex'], prev_t1, t2, op_tok['linea']))

            if lex2 or t2:
                ops.append((lex2, t2, ln2))
        return t1, j, ops, opsinfo

    #  Recorrido principal 
    i, n = 0, len(toks)
    while i < n:
        tk = toks[i]

        # scopes
        if tk['lex'] == '{':
            scope_stack.append({})
            i += 1; continue
        if tk['lex'] == '}':
            if len(scope_stack) > 1: scope_stack.pop()
            i += 1; continue

        # DO: abre bucle; condición se valida en while de cierre 
        if tk['tipo'] == 'KW' and tk['lex'] == 'do':
            pending_do += 1
            loop_depth += 1
            i += 1
            continue

        # WHILE 
        if tk['tipo'] == 'KW' and tk['lex'] == 'while':
            # ¿cierra un do-while?
            prev = i - 1
            while prev >= 0 and toks[prev]['lex'] in (';',):
                prev -= 1
            if pending_do > 0 and prev >= 0 and toks[prev]['lex'] == '}':
                # validar condición del while de cierre
                j = i
                while j < n and toks[j]['lex'] != '(': j += 1
                k = j + 1
                has_rel = False
                while k < n and toks[k]['lex'] != ')':
                    if toks[k]['lex'] in REL: has_rel = True
                    k += 1
                if not has_rel:
                    errores.append({
                        'token': 'ErrCond',
                        'linea': tk['linea'], 'lex': 'while',
                        'desc': 'Incompatibilidad de tipos'
                    })
                # consumir ')' y ';'
                i = k + 1 if k < n else k
                if i < n and toks[i]['lex'] == ';':
                    i += 1
                pending_do -= 1
                loop_depth = max(0, loop_depth - 1)
                continue
            else:
                # while normal
                j = i
                while j < n and toks[j]['lex'] != '(': j += 1
                k = j + 1
                has_rel = False
                while k < n and toks[k]['lex'] != ')':
                    if toks[k]['lex'] in REL: has_rel = True
                    k += 1
                if not has_rel:
                    errores.append({
                        'token': 'ErrCond',
                        'linea': tk['linea'], 'lex': 'while',
                        'desc': 'Incompatibilidad de tipos'
                    })
                loop_depth += 1
                i = k + 1 if k < n else i + 1
                continue

        # FOR 
        if tk['tipo'] == 'KW' and tk['lex'] == 'for':
            j = i
            while j < n and toks[j]['lex'] != '(': j += 1
            k = j + 1
            has_rel = False
            while k < n and toks[k]['lex'] != ')':
                if toks[k]['lex'] in REL: has_rel = True
                k += 1
            if not has_rel:
                errores.append({
                    'token': 'ErrCond',
                    'linea': tk['linea'], 'lex': 'for',
                    'desc': 'Incompatibilidad de tipos'
                })
            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        # break/continue fuera de bucle
        if tk['tipo'] == 'KW' and tk['lex'] in ('break', 'continue'):
            if loop_depth == 0:
                errores.append({
                    'token': 'ErrLoopCtl',
                    'linea': tk['linea'], 'lex': tk['lex'],
                    'desc': 'Incompatibilidad de tipos'
                })
            i += 1
            continue

        # Asignación: IDENT '=' expr
        if tk['tipo'] == 'IDENT' and i+1 < n and toks[i+1]['lex'] == '=':
            lhs = tk
            lhs_t = lookup(lhs['lex'], scope_stack)
            if not lhs_t:
                errores.append({
                    'token': 'ErrUndef',
                    'linea': lhs['linea'], 'lex': lhs['lex'],
                    'desc': 'Variable indefinida'
                })

            j = i + 2
            # NO emitir ErrTipoOPA dentro de la RHS de asignación
            rhs_t, j, rhs_ops, rhs_opsinfo = expr(j, emit_opa_errors=False)

            # Indefinidas en RHS (TODAS las que aparezcan, sin cortar el análisis)
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
            # IMPORTANTE: ya NO hacemos 'continue' aquí.
            # Seguimos, para también reportar incompatibilidades de tipo en la misma línea.

            # Incompatibilidad según LHS (múltiples culpables)
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
                    # Solo cuenta como incompatibilidad si tiene tipo conocido y NO está permitido
                    if ty and ty not in allowed:
                        key = (lexeme, lno)
                        if key not in vistos:
                            errores.append({
                                'token': f'ErrAsig_{lhs_t}',
                                'linea': lno,
                                'lex': lexeme,
                                'desc': f'Incompatibilidad de tipos, {lhs_t}'
                            })
                            vistos.add(key)
                            hubo_operando_invalido = True

                # Si no hubo operando inválido PERO el tipo global choca y NO hay indefinidas,
                # culpamos al operador responsable (p. ej., '/').
                if not hubo_operando_invalido and not indefs and rhs_t not in allowed:
                    culprit_lex = None
                    culprit_line = lhs['linea']
                    for op_lex, tleft, tright, opln in rhs_opsinfo:
                        if op_lex == '/' and tleft in (TOK_INT, TOK_FLOAT) and tright in (TOK_INT, TOK_FLOAT):
                            culprit_lex = '/'
                            culprit_line = opln
                            break
                    if not culprit_lex:
                        if rhs_ops:
                            culprit_lex  = rhs_ops[-1][0]
                            culprit_line = rhs_ops[-1][2]
                        else:
                            culprit_lex  = lhs['lex']
                            culprit_line = lhs['linea']
                    errores.append({
                        'token': f'ErrAsig_{lhs_t}',
                        'linea': culprit_line,
                        'lex': culprit_lex,
                        'desc': f'Incompatibilidad de tipos, {lhs_t}'
                    })

            # saltar hasta ';'
            while j < n and toks[j]['lex'] != ';': j += 1
            i = j + 1 if j < n else j
            continue

        # identificador usado suelto
        if tk['tipo'] == 'IDENT':
            if not lookup(tk['lex'], scope_stack):
                errores.append({
                    'token': 'ErrUndef',
                    'linea': tk['linea'], 'lex': tk['lex'],
                    'desc': 'Variable indefinida'
                })

        i += 1

    return errores  # errores de SEGUNDA pasada


#            API PARA LA GUI (misma firma de siempre)
def analizar_dos_pasadas(texto):
    # TABLA DE SÍMBOLOS + errores de 1ª pasada
    toks, tabla_simbolos, _scopes, err_p1 = construir_tabla_simbolos_y_err_p1(texto)

    # TABLA DE ERRORES (2ª pasada semántica)
    err_p2 = recolectar_errores_semanticos(texto, toks, tabla_simbolos)

    # Unir y ordenar por línea (y columna si se hubiera guardado)
    errores = []
    errores.extend(err_p1)
    errores.extend(err_p2)

    for e in errores:
        e.setdefault('token', 'ErrSem')
        e.setdefault('linea', 0)
        e.setdefault('lex', '')
        e.setdefault('desc', 'Incompatibilidad de tipos')

    errores.sort(key=lambda e: (e.get('linea', 0), e.get('col', 0)))

    return tabla_simbolos, errores
