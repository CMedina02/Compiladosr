import re  # asegúrate de tener este import al inicio de analyzer.py

def optimizar_codigo_fuente(texto: str) -> str:
    """
    Optimización local basada en:
      'Instrucciones que se repiten sin haber tenido modificación alguna
       en uno de sus valores'.

    Regla implementada:

    - Si detectamos que una MISMA expresión 'expr' ya se calculó antes
      con las mismas versiones de sus variables (es decir, ninguna de las
      variables usadas ha sido modificada):

        1) Primera vez:
              A = expr;
           Se guarda que 'expr' fue calculada en A con cierto estado de variables.

        2) Si luego encontramos:
              A = expr;
           con el mismo estado de variables:
              -> Se ELIMINA la segunda (es redundante).

        3) Si luego encontramos:
              B = expr;
           con el mismo estado de variables:
              -> Se transforma en:
                 B = A;

    Esto se ejecuta ANTES de generar tokens, tabla de símbolos, errores y triplos.
    """

    lineas = texto.splitlines()

    # Versión de cada variable (incrementa cada vez que esa variable se asigna)
    versiones = {}  # nombre -> entero

    # Mapa de expresiones ya calculadas:
    # clave = (expr_normalizada, snapshot_versiones) -> primer_lhs
    expr_cache = {}

    # Regex para detectar asignaciones simples:  id = expr ;
    re_asig = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.+?);?\s*$')
    re_ident = re.compile(r'\b[A-Za-z_]\w*\b')

    def normalizar_expr(expr: str) -> str:
        # Quitamos espacios para comparar expresiones de forma robusta
        return re.sub(r'\s+', '', expr)

    resultado = []

    for linea in lineas:
        original = linea
        stripped = linea.strip()

        m = re_asig.match(stripped)
        if not m:
            # No es una asignación simple -> la dejamos igual
            resultado.append(original)
            continue

        lhs = m.group(1)           # variable a la izquierda
        rhs = m.group(2).strip()   # expresión a la derecha

        # Normalizamos la expresión para que "B + C" y "B+C" se consideren iguales
        rhs_norm = normalizar_expr(rhs)

        # Variables usadas en la expresión de la derecha
        usados = set(re_ident.findall(rhs))

        # Snapshot de versiones de esas variables en este punto
        snapshot = tuple(sorted((var, versiones.get(var, 0)) for var in usados))

        key = (rhs_norm, snapshot)

        if key in expr_cache:
            # Ya se calculó esta expresión antes con las mismas versiones
            primer_lhs = expr_cache[key]

            if primer_lhs == lhs:
                # Caso 1: A = expr; ... A = expr;  => segunda es redundante
                # => simplemente no agregamos la línea (la eliminamos)
                continue
            else:
                # Caso 2: A = expr; ... B = expr;  => optimizamos a B = A;
                nueva_linea = f"{lhs} = {primer_lhs};"
                resultado.append(nueva_linea)
                # Actualizamos versión de B (lhs)
                versiones[lhs] = versiones.get(lhs, 0) + 1
                continue

        # Es la PRIMERA vez que vemos esta expresión con este snapshot de versiones
        expr_cache[key] = lhs
        resultado.append(original)

        # Después de ejecutar la instrucción, la versión de lhs cambia
        versiones[lhs] = versiones.get(lhs, 0) + 1

    return "\n".join(resultado)
