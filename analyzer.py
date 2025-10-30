# analyzer.py (ACTUALIZADO CON PARSER)
import re
from rules import (
    TOKEN_PATTERN, idx_a_line_col,
    RE_TIPO, RE_IDENT, RE_FLOAT, RE_INT, RE_STR, RE_CHAR, RE_SYSPRINT,
    TOK_INT, TOK_STR, TOK_FLOAT
)


# =========================================================================
#                          CLASES AST (Abstract Syntax Tree)
# =========================================================================

class Node:
    """Clase base para todos los nodos del AST."""
    def __init__(self, token=None, children=None):
        self.token = token       # Token léxico asociado
        self.children = children if children is not None else []

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

class BinOp(Node):
    """Operación binaria (ej: a + b)."""
    def __init__(self, op_token, left, right):
        super().__init__(op_token, [left, right])
        self.op = op_token['lex']
        self.left = left
        self.right = right

    def __repr__(self):
        return f"<BinOp: {self.op}>"

class Literal(Node):
    """Literales (ej: 5, "hola", 3.14)."""
    def __init__(self, token):
        super().__init__(token)
        self.value = token['lex']

    def __repr__(self):
        return f"<Literal: {self.value}>"

class Ident(Node):
    """Identificador (ej: x, miVariable)."""
    def __init__(self, token):
        super().__init__(token)
        self.name = token['lex']

    def __repr__(self):
        return f"<Ident: {self.name}>"

# =========================================================================
#                              LEXER
# =========================================================================

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

        # Usar re.fullmatch para clasificar el lexema capturado
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
            if lower in ('if','else','while','for','do','break','continue','func','return'):
                tipo = 'KW'; lexema = lower
            else:
                tipo = 'IDENT'
        else:
            tipo = 'OP'
        toks.append({'tipo': tipo, 'lex': lexema, 'linea': linea, 'col': col})
    return toks


# =========================================================================
#                             PARSER
# =========================================================================

class Parser:
    """Implementa el Parser Descendente Recursivo y construye el AST."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.errores_sintacticos = []

    def peek(self):
        """Devuelve el token actual sin avanzar."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else {'lex': 'EOF', 'tipo': 'EOF', 'linea': -1, 'col': -1}

    def consume(self, expected_lex=None, expected_tipo=None):
        """Avanza y devuelve el token, o reporta error si no coincide."""
        current = self.peek()
        
        match = False
        if expected_lex and current['lex'] == expected_lex:
            match = True
        elif expected_tipo and current['tipo'] == expected_tipo:
            match = True
        elif not expected_lex and not expected_tipo:
            match = True # Si no se espera nada específico, solo consume

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
            # En un parser real, se intentaría recuperar el error; aquí solo saltamos el token
            if self.pos < len(self.tokens):
                self.pos += 1
            return None # Fallo en el parseo

    def report_error(self, token_type, line, lexeme, desc):
        """Agrega un error sintáctico a la lista."""
        self.errores_sintacticos.append({
            'token': token_type,
            'linea': line,
            'lex': lexeme,
            'desc': desc
        })

    # Regla: factor (Literales, Identificadores, (Expresiones))
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
            self.report_error('ErrSintaxis', token['linea'], token['lex'], "Se esperaba un literal, identificador o '('")
            return None

    # Regla: term (Multiplicación y División)
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

    # Regla: expresion (Suma y Resta)
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
    
    # Regla: Sentencia de Asignación (id = expresion ;)
    def parse_asignacion(self, identifier_token):
        self.consume('=')
        rhs = self.parse_expresion()
        self.consume(';')
        # AST de asignación: aquí podríamos crear un nodo Asignacion(Ident(identifier_token), rhs)
        return rhs # Por ahora solo devolvemos la expresión RHS para el análisis semántico

    # Regla: Sentencia Genérica (para validar que todo termine con ';')
    def parse_sentencia(self):
        token = self.peek()
        # Manejo de asignación (el más complejo)
        if token['tipo'] == 'IDENT' and self.tokens[self.pos + 1]['lex'] == '=':
            ident_tok = self.consume()
            return self.parse_asignacion(ident_tok)
        
        # Manejo de System.out.println
        elif token['lex'] == 'System.out.println':
            self.consume('System.out.println')
            self.consume('(')
            self.parse_expresion()
            self.consume(')')
            self.consume(';')
            return True

        # Manejo de declaraciones (que ya fueron validadas en la p1, solo validamos ';' ahora)
        elif token['tipo'] == 'TIPO':
            while self.peek()['lex'] != ';':
                self.consume()
            self.consume(';')
            return True
        
        # Sentencias de control (if, while, etc.)
        elif token['tipo'] == 'KW':
            # Simplificado: Consumir tokens hasta encontrar un ';' o un '{' de bloque
            while self.peek()['lex'] not in (';', '{', 'EOF'):
                self.consume()
            # Si es un bucle/if, debe tener bloque o ;
            if self.peek()['lex'] == '{':
                self.consume('{')
                self.parse_bloque() # Consumir el bloque
                self.consume('}')
            elif self.peek()['lex'] == ';':
                self.consume(';')
            return True

        # Solo consumir tokens hasta ';' o EOF si no es un tipo conocido (puede ser error sintáctico)
        if token['lex'] != 'EOF':
            self.consume()
            # Asumiendo que el error está en el token actual, saltamos el resto de la línea
            while self.peek()['lex'] != ';' and self.peek()['lex'] != 'EOF':
                self.consume()
            if self.peek()['lex'] == ';':
                self.consume(';')
            return True

        return None # EOF


    # Regla: Bloque de código ({ sentencia* })
    def parse_bloque(self):
        # En un parser real, se llamaría a parse_sentencia() en un bucle
        # Aquí solo aseguramos que las llaves estén balanceadas y consumimos sentencias
        # Para evitar complejidad excesiva en este nivel.
        while self.peek()['lex'] != '}' and self.peek()['lex'] != 'EOF':
            self.parse_sentencia()
        
        
    def parse(self):
        """Punto de entrada: Programa = (sentencia)* EOF"""
        # En un programa real, llamaríamos a parse_sentencia() hasta EOF
        while self.peek()['tipo'] != 'EOF':
            self.parse_sentencia()
            # Esto asegurará que los errores sintácticos se reporten
            
        return self.errores_sintacticos # Devolvemos los errores sintácticos para la API

# =========================================================================
#                       HELPERS DE TIPADO
# =========================================================================
# ... (Mantener originales)

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


# =========================================================================
#             A) LLENADO DE LA TABLA DE SÍMBOLOS (Primera pasada)
# =========================================================================

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

            #  PATRÓN: <TIPO> = id (, id)* ;  (re-tipeo inválido/declaración duplicada) 
            #  Esta sintaxis DEBERÍA ser marcada como ErrSintaxis, pero la mantenemos aquí por el P1
            if i < n and toks[i]['lex'] == '=':
                put('=', '')
                i += 1
                # Consumir hasta ';'
                while i < n and toks[i]['lex'] != ';':
                    if toks[i]['tipo'] == 'IDENT':
                        ident = toks[i]
                        # El ErrDeclDup es lo importante aquí
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

# =========================================================================
#            B) LLENADO DE LA TABLA DE ERRORES (Segunda pasada)
# =========================================================================
# Esta función es la que más necesitará ajustes futuros, pero por ahora se mantiene similar.

def recolectar_errores_semanticos(toks, tabla_simbolos):
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

    # El analizador semántico debe usar la lógica del parser para evaluar la expresión
    # Sin el AST, mantenemos la evaluación simple (izq->der) SOLO para tipado, 
    # pero el Parser ya verificó la estructura.

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
        # Iteración simplificada (sin considerar precedencia, que ya hizo el parser)
        while j < len(toks) and toks[j]['lex'] in OPA:
            op_tok = toks[j]; j += 1
            (lex2, t2, ln2), j = atom(j, emit_opa_errors)

            prev_t1 = t1  # tipo a la izquierda antes de aplicar el operador

            if t1 in (TOK_INT, TOK_FLOAT) and t2 in (TOK_INT, TOK_FLOAT):
                # Regla de promoción: si hay '/' o F$, el resultado es F$
                t1 = TOK_FLOAT if (op_tok['lex'] == '/' or TOK_FLOAT in (t1, t2)) else TOK_INT
            elif t1 == TOK_STR and t2 == TOK_STR and op_tok['lex'] in ('+'): # Solo concatenación
                t1 = TOK_STR
            else:
                if op_tok['lex'] != '-': # Permitimos INT/FLOAT - STR para ErrAsig_C$, si es una asignacion
                    if emit_opa_errors:
                        culprit = lex2 or lex1 or op_tok['lex']
                        errores.append({
                            'token': 'ErrTipoOPA',
                            'linea': op_tok['linea'],
                            'lex': culprit,
                            'desc': 'Operación incompatible de tipos'
                        })
                t1 = t1 or t2 or TOK_INT # Asumir INT para continuar si hay error

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

        # DO, WHILE, FOR: Manejo de loop_depth y ErrCond
        if tk['tipo'] == 'KW' and tk['lex'] == 'do':
            pending_do += 1
            loop_depth += 1
            i += 1; continue
        if tk['tipo'] == 'KW' and tk['lex'] == 'while':
            # Revisar condición (busca relacional dentro de paréntesis)
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
                    'linea': tk['linea'], 'lex': tk['lex'],
                    'desc': 'Condición de bucle sin operador relacional'
                })

            if pending_do > 0 and i > 0 and toks[i-1]['lex'] == ';': # while(i<1); de un do-while
                # Consumir el resto del while
                i = k + 2 if k < n and toks[k+1]['lex'] == ';' else k+1
                pending_do -= 1
                loop_depth = max(0, loop_depth - 1)
                continue
            
            loop_depth += 1
            i = k + 1 if k < n else i + 1
            continue

        if tk['tipo'] == 'KW' and tk['lex'] == 'for':
            # ... (Lógica similar a while para ErrCond) ...
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
                    'desc': 'Condición de bucle sin operador relacional'
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
                    'desc': 'Sentencia de control de bucle fuera de contexto'
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

            # Indefinidas en RHS
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
            
            # Incompatibilidad según LHS (ErrAsig)
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
                            errores.append({
                                'token': f'ErrAsig_{lhs_t}',
                                'linea': lno,
                                'lex': lexeme,
                                'desc': f'Asignación incompatible, se esperaba {lhs_t}'
                            })
                            vistos.add(key)
                            hubo_operando_invalido = True
                            
                # Caso especial: Incompatibilidad de tipo de resultado final (ej: E$ = 5.0)
                if not hubo_operando_invalido and not indefs and rhs_t not in allowed:
                    # En este punto el parser ya debería haber verificado la sintaxis
                    culprit_lex = rhs_ops[-1][0] if rhs_ops else lhs['lex']
                    culprit_line = rhs_ops[-1][2] if rhs_ops else lhs['linea']
                    errores.append({
                        'token': f'ErrAsig_{lhs_t}',
                        'linea': culprit_line,
                        'lex': culprit_lex,
                        'desc': f'Asignación incompatible, el resultado es {rhs_t}, se esperaba {lhs_t}'
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

# =========================================================================
#            API PARA LA GUI (misma firma de siempre)
# =========================================================================

def analizar_dos_pasadas(texto):
    # 1. Análisis Léxico (se mantiene)
    toks = lex(texto)

    # 2. Análisis Sintáctico (NUEVA FASE)
    parser = Parser(toks)
    err_p_sintaxis = parser.parse()

    # 3. Llenado de la Tabla de Símbolos y Errores P1 (se mantiene)
    _toks_aux, tabla_simbolos, _scopes, err_p1 = construir_tabla_simbolos_y_err_p1(texto)
    
    # 4. Análisis Semántico y Errores P2 (se mantiene)
    err_p2 = recolectar_errores_semanticos(toks, tabla_simbolos)

    # Unir todos los errores
    errores = []
    errores.extend(err_p_sintaxis)
    errores.extend(err_p1)
    errores.extend(err_p2)

    for e in errores:
        e.setdefault('token', 'ErrSem')
        e.setdefault('linea', 0)
        e.setdefault('lex', '')
        e.setdefault('desc', 'Error general')

    # Eliminar duplicados si el parser y el p1/p2 reportan el mismo error en la misma línea
    # Esto es avanzado, por ahora solo ordenamos.

    errores.sort(key=lambda e: (e.get('linea', 0), e.get('col', 0)))

    return tabla_simbolos, errores