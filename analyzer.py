import re

# ===== Tipos =====
TOK_INT   = 'E$'
TOK_FLOAT = 'F$'
TOK_STR   = 'C$'
TOK_BOOL  = 'B$'  # lógico derivado (relacionales y lógicos)

# ===== Lexer =====
TOKEN_SPEC = [
    ("WS",      r"[ \t]+"),
    ("NEWLINE", r"\r?\n"),
    ("COMMENT", r"//[^\n]*"),
    ("STR",     r"\"([^\"\\]|\\.)*\""),
    ("FLOAT",   r"\d+\.\d+"),
    ("INT",     r"\d+"),
    ("OP",      r"==|!=|<=|>=|\|\||&&|[+\-*/%<>=]"),
    ("PUNC",    r"[(){},;]"),
    ("TYPE",    r"(E\$|F\$|C\$)"),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
]
MASTER_RE = re.compile("|".join(f"(?P<{n}>{p})" for n,p in TOKEN_SPEC))
KW = {"do","while","for","break","continue","System","out","println"}

def lex(texto:str):
    tokens, line, col, pos = [], 1, 1, 0
    for m in MASTER_RE.finditer(texto):
        typ, lexeme, start = m.lastgroup, m.group(), m.start()
        while pos < start:
            if texto[pos] == "\n": line += 1; col = 1
            else: col += 1
            pos += 1
        if typ in ("WS","COMMENT"):  # ignora
            for ch in lexeme:
                if ch == "\n": line += 1; col = 1
                else: col += 1
            pos = m.end(); continue
        if typ == "NEWLINE":
            line += 1; col = 1; pos = m.end(); continue
        tok = {"tipo": typ, "lex": lexeme, "linea": line, "col": col}
        if typ == "IDENT" and lexeme in KW: tok["tipo"] = "KW"
        tokens.append(tok)
        for ch in lexeme:
            if ch == "\n": line += 1; col = 1
            else: col += 1
        pos = m.end()
    return tokens

# ===== 1ª pasada: Tabla de símbolos =====
def analizar_primera_pasada(texto:str):
    toks = lex(texto)
    i, n = 0, len(toks)
    tabla, declarados = [], set()
    while i < n:
        tk = toks[i]
        if tk["tipo"] == "TYPE":
            t = tk["lex"]; i += 1
            while i < n:
                if toks[i]["tipo"] == "IDENT":
                    name = toks[i]["lex"]
                    if name not in declarados:
                        tabla.append((name, t)); declarados.add(name)
                    i += 1
                    if i < n and toks[i]["lex"] == "=":
                        i += 1
                        depth = 0
                        while i < n and not (depth == 0 and toks[i]["lex"] in (",",";")):
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

# ===== 2ª pasada: Errores =====
REL = {'==','!=','<','<=','>','>='}
LOG = {'&&','||'}
OPA = {'+','-','*','/','%'}

def _tipo_literal(tok):
    if tok['tipo']=='INT': return TOK_INT
    if tok['tipo']=='FLOAT': return TOK_FLOAT
    if tok['tipo'] in ('STR','CHAR'): return TOK_STR
    return ''

def _has_rel_or_log(tokens):
    depth=0; rel=False; log=False
    for tk in tokens:
        if tk['lex']=='(':
            depth+=1
        elif tk['lex']==')':
            depth-=1
        elif depth==0:
            if tk['lex'] in REL: rel=True
            if tk['lex'] in LOG: log=True
    return rel or (log and rel)

def recolectar_errores_semanticos(toks, tabla):
    global_scope = {lex:t for (lex,t) in tabla}
    def lookup(name): return global_scope.get(name,'')

    errores = []
    i, n = 0, len(toks)

    def atom(j):
        if j>=n: return ('','',0,0), j
        tk = toks[j]
        if tk['lex']=='(':
            t1, k, ops = expr(j+1)
            while k<n and toks[k]['lex']!=')': k+=1
            return ('', t1, tk['linea'], tk.get('col',0)), (k+1 if k<n else k)
        ty = _tipo_literal(tk) if tk['tipo']!='IDENT' else lookup(tk['lex'])
        return (tk['lex'], ty, tk['linea'], tk.get('col',0)), j+1

    def expr(j):
        (lex1, t1, ln1, col1), k = atom(j)
        ops=[(lex1,t1,ln1,col1)]
        while k<n and toks[k]['lex'] in OPA:
            op = toks[k]; k+=1
            (lex2,t2,ln2,col2), k = atom(k)
            if t1 in (TOK_INT,TOK_FLOAT) and t2 in (TOK_INT,TOK_FLOAT):
                t1 = TOK_FLOAT if (op['lex']=='/' or TOK_FLOAT in (t1,t2)) else TOK_INT
            elif t1==TOK_STR and t2==TOK_STR and op['lex']=='+':
                t1 = TOK_STR
            else:
                errores.append({'token':'ErrTipoOPA','linea':op['linea'],'col':op.get('col',0),
                                'lex':lex2 or lex1 or op['lex'],'desc':'Operación incompatible de tipos'})
                t1 = t1 or t2 or TOK_INT
            ops.append((lex2,t2,ln2,col2))
        return t1, k, ops

    def validar_cond(i_start):
        j=i_start
        while j<n and toks[j]['lex']!='(': j+=1
        if j>=n: return i_start, False
        k=j+1; depth=1
        while k<n and depth>0:
            if toks[k]['lex']=='(': depth+=1
            elif toks[k]['lex']==')': depth-=1
            k+=1
        cond = toks[j+1:k-1] if k-1>j+1 else []
        return k, _has_rel_or_log(cond)

    while i<n:
        tk=toks[i]
        if tk['tipo']=='IDENT' and i+1<n and toks[i+1]['lex']=='=':
            lhs=tk; lhs_t=lookup(lhs['lex'])
            if not lhs_t:
                errores.append({'token':'ErrUndef','linea':lhs['linea'],'col':lhs.get('col',0),
                                'lex':lhs['lex'],'desc':'Variable indefinida'})
            j=i+2
            rhs_t, j, ops = expr(j)
            indef=set()
            for lx,ty,ln,co in ops:
                if lx and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lx) and not lookup(lx):
                    if lx not in indef:
                        errores.append({'token':'ErrUndef','linea':ln,'col':co,'lex':lx,'desc':'Variable indefinida'})
                        indef.add(lx)
            if lhs_t:
                allowed = {TOK_INT} if lhs_t==TOK_INT else ({TOK_INT,TOK_FLOAT} if lhs_t==TOK_FLOAT else {TOK_STR})
                bad=False
                for lx,ty,ln,co in ops:
                    if ty and ty not in allowed:
                        errores.append({'token':f'ErrAsig_{lhs_t}','linea':ln,'col':co,'lex':lx,
                                        'desc':f'Asignación incompatible, se esperaba {lhs_t}'})
                        bad=True
                if not bad and not indef and rhs_t not in allowed:
                    culprit = ops[-1][0] if ops else lhs['lex']
                    errores.append({'token':f'ErrAsig_{lhs_t}','linea':lhs['linea'],
                                    'col':lhs.get('col',0),'lex':culprit,
                                    'desc':f'Asignación incompatible, resultado {rhs_t} ≠ {lhs_t}'})
            while j<n and toks[j]['lex']!=';': j+=1
            i = j+1 if j<n else j
            continue

        if tk['tipo']=='KW' and tk['lex']=='while':
            k_after, ok = validar_cond(i)
            if not ok:
                errores.append({'token':'ErrCond','linea':tk['linea'],'col':tk.get('col',0),
                                'lex':'while','desc':'Condición inválida: requiere operador relacional'})
            i = k_after + (1 if k_after<n and toks[k_after]['lex']==';' else 0)
            continue

        i+=1
    return errores

def analizar_dos_pasadas(texto:str):
    tabla = analizar_primera_pasada(texto)
    errores = recolectar_errores_semanticos(lex(texto), tabla)
    return tabla, errores

# ===== utilidades parsing expr/cond para triplos =====
REL_OPS  = {"==","!=", "<","<=",">",">="}

def _strip_parens(tokens):
    if len(tokens)>=2 and tokens[0]["lex"]=="(" and tokens[-1]["lex"]==")":
        depth=0; ok=True
        for k, tk in enumerate(tokens):
            if tk["lex"]=="(": depth+=1
            elif tk["lex"]==")":
                depth-=1
                if depth==0 and k!=len(tokens)-1: ok=False; break
        if ok: return tokens[1:-1]
    return tokens

def _split_top(tokens, seps):
    out=[]; depth=0; last=0
    for i, tk in enumerate(tokens):
        if tk['lex']=='(':
            depth+=1
        elif tk['lex']==')':
            depth-=1
        elif depth==0 and tk['lex'] in seps:
            out.append(tokens[last:i]); last=i+1
    out.append(tokens[last:]); return out

def _slice_until(tokens, i, stops):
    j=i; depth=0
    while j<len(tokens):
        if tokens[j]["lex"]=="(": depth+=1
        elif tokens[j]["lex"]==")": depth-=1
        if depth==0 and tokens[j]["lex"] in stops: break
        j+=1
    return tokens[i:j], j

# ===== Emisor de triplos =====
class TriploEmitter:
    def __init__(self):
        self.rows=[]; self.N=0; self.temp_counter=0
    def _nextN(self): self.N+=1; return self.N
    def emit(self, O, DO="", DF=""):
        n=self._nextN(); self.rows.append({"N":n,"O":O,"DO":DO,"DF":DF}); return n
    def newT(self): self.temp_counter+=1; return f"T{self.temp_counter}"
    def resetTemps(self): self.temp_counter=0

    # expr con unario -
    def _gen_factor(self, toks, i):
        if i<len(toks) and toks[i]["lex"]=="-":
            i+=1; t, i = self._gen_factor(toks, i)
            t0=self.newT(); self.emit("=", t0, "0"); self.emit("-", t0, t); return t0, i
        if i<len(toks) and toks[i]["lex"]=="(":
            t, j = self._gen_expr(toks, i+1)
            k,depth=j,1
            while k<len(toks) and depth>0:
                if toks[k]["lex"]=="(": depth+=1
                elif toks[k]["lex"]==")": depth-=1
                k+=1
            return t, k
        tk=toks[i]; t=self.newT(); self.emit("=", t, tk["lex"]); return t, i+1

    def _gen_term(self, toks, i):
        left, k = self._gen_factor(toks, i)
        while k<len(toks) and toks[k]["lex"] in ("*","/","%"):
            op=toks[k]["lex"]; k+=1
            right, k = self._gen_factor(toks, k)
            self.emit(op, left, right)
        return left, k

    def _gen_expr(self, toks, i):
        left, k = self._gen_term(toks, i)
        while k<len(toks) and toks[k]["lex"] in ("+","-"):
            op=toks[k]["lex"]; k+=1
            right, k = self._gen_term(toks, k)
            self.emit(op, left, right)
        return left, k

    def gen_assignment(self, lhs, rhs_tokens):
        self.resetTemps()
        if not rhs_tokens: self.emit("=", lhs, "0"); return
        t,_ = self._gen_expr(rhs_tokens, 0)
        self.emit("=", lhs, t)

    def gen_rel_row(self, rel_tokens):
        self.resetTemps()
        depth=0; pos=-1; relop=None
        for j, tk in enumerate(rel_tokens):
            if tk["lex"]=="(": depth+=1
            elif tk["lex"]==")": depth-=1
            elif depth==0 and tk["lex"] in REL_OPS: relop=tk["lex"]; pos=j; break
        if relop is None:
            t,_ = self._gen_expr(rel_tokens, 0)
            return self.emit("!=", t, "0")
        L = rel_tokens[:pos]; R = rel_tokens[pos+1:]
        tL,_ = self._gen_expr(L, 0); tR,_ = self._gen_expr(R, 0)
        return self.emit(relop, tL, tR)

    def gen_condition_do_while(self, cond_tokens, N_body):
        """
        OR: FALSE -> 1er rel del siguiente término; último FALSE -> END
        AND: TRUE -> siguiente factor (o BODY si último); FALSE -> (si hay OR) siguiente término; si no -> END
        Siempre se emite: [REL] + TRUE/ FALSE justo debajo.
        Devuelve índices de filas que deben parchearse con END.
        """
        cond_tokens=_strip_parens(cond_tokens)
        or_terms=_split_top(cond_tokens, {'||'})

        pending_false_to_next_term=[]
        pending_to_end=[]

        for t_idx, term in enumerate(or_terms):
            term=_strip_parens(term)
            factors=_split_top(term, {'&&'})
            pending_true_to_next_factor=[]
            first_relN=None
            local_false_of_term=[]

            for f_idx, fac in enumerate(factors):
                fac=_strip_parens(fac)
                relN=self.gen_rel_row(fac)
                if first_relN is None:
                    first_relN=relN
                    # parchear FALSE de términos previos hacia el inicio de este término
                    for r in pending_false_to_next_term:
                        self.rows[r]["DO"]=str(first_relN)
                    pending_false_to_next_term=[]

                # parchear TRUE de factor previo hacia este relN
                for r in pending_true_to_next_factor:
                    self.rows[r]["DO"]=str(relN)
                pending_true_to_next_factor=[]

                last_factor=(f_idx==len(factors)-1)
                last_term=(t_idx==len(or_terms)-1)

                if len(factors)==1:
                    self.emit("TRUE", str(N_body), "")
                    if last_term:
                        r=self.emit("FALSE","PENDING_END",""); pending_to_end.append(r)
                    else:
                        r=self.emit("FALSE","PENDING_NEXT_TERM",""); local_false_of_term.append(r)
                else:
                    if not last_factor:
                        rT=self.emit("TRUE","PENDING_NEXT_FACTOR",""); pending_true_to_next_factor.append(rT)
                        if last_term:
                            rF=self.emit("FALSE","PENDING_END",""); pending_to_end.append(rF)
                        else:
                            rF=self.emit("FALSE","PENDING_NEXT_TERM",""); local_false_of_term.append(rF)
                    else:
                        self.emit("TRUE", str(N_body), "")
                        if last_term:
                            rF=self.emit("FALSE","PENDING_END",""); pending_to_end.append(rF)
                        else:
                            rF=self.emit("FALSE","PENDING_NEXT_TERM",""); local_false_of_term.append(rF)

            pending_false_to_next_term.extend(local_false_of_term)

        # lo que quedó apuntando a siguiente término (y ya no hay) -> END
        for r in pending_false_to_next_term:
            self.rows[r]["DO"]="PENDING_END"

        end_needed=[i for i,row in enumerate(self.rows) if row["O"] in ("TRUE","FALSE") and row["DO"]=="PENDING_END"]
        return end_needed

def generar_triplos(texto:str):
    toks=lex(texto)
    em=TriploEmitter()
    i,n=0,len(toks)

    while i<n:
        tk=toks[i]
        # do { ... } while (cond);
        if tk["tipo"]=="KW" and tk["lex"]=="do":
            i+=1
            if i<n and toks[i]["lex"]=="{":
                i+=1
                body_start = em.N + 1

                while i<n and toks[i]["lex"]!="}":
                    if i+1<n and toks[i]["tipo"]=="IDENT" and toks[i+1]["lex"]=="=":
                        lhs=toks[i]["lex"]; i+=2
                        rhs, j = _slice_until(toks, i, {";","}"})
                        em.gen_assignment(lhs, rhs)
                        i=j
                        if i<n and toks[i]["lex"]==";": i+=1
                    else:
                        i+=1

                if i<n and toks[i]["lex"]=="}": i+=1

                if i<n and toks[i]["tipo"]=="KW" and toks[i]["lex"]=="while":
                    i+=1
                    if i<n and toks[i]["lex"]=="(":
                        j=i+1; depth=1
                        while j<n and depth>0:
                            if toks[j]["lex"]=="(": depth+=1
                            elif toks[j]["lex"]==")": depth-=1
                            j+=1
                        cond=toks[i+1:j-1] if j-1>i+1 else []

                        pend_end = em.gen_condition_do_while(cond, body_start)

                        i=j
                        if i<n and toks[i]["lex"]==";": i+=1

                        local_end = em.N + 1
                        for idx in pend_end:
                            em.rows[idx]["DO"]=str(local_end)
                continue

        # asignaciones sueltas
        if i+1<n and tk["tipo"]=="IDENT" and toks[i+1]["lex"]=="=":
            lhs=tk["lex"]; i+=2
            rhs, j = _slice_until(toks, i, {";","}"})
            em.gen_assignment(lhs, rhs)
            i=j
            if i<n and toks[i]["lex"]==";": i+=1
            continue

        i+=1

    return [(r["N"], r["O"], r["DO"], r["DF"]) for r in em.rows]

# ===== API para la app =====
def analizar_y_triplos(texto:str):
    tabla, errores = analizar_dos_pasadas(texto)
    try: triplos = generar_triplos(texto)
    except Exception: triplos = []
    return tabla, errores, triplos
