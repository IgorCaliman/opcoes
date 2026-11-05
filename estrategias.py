"""Ideias de operações (direcional e travas) a partir de HiLo, IFR e médias móveis.

Tudo aqui é regra simples e explícita, para você ajustar os limites abaixo.
Não usa preço de opção: os strikes são só referências calculadas sobre o ativo.
"""
import numpy as np
import pandas as pd

# ---- Parâmetros (ajuste à vontade) -----------------------------------------
HORIZONTE = 20          # candles (~1 mês) usados no movimento esperado (σ)
IFR_ALTO = 70           # acima disso: sobrecompra
IFR_BAIXO = 30          # abaixo disso: sobrevenda
SCORE_DIRECIONAL = 4    # força mínima (em módulo) para ideia direcional
SCORE_TRAVA = 3         # força mínima (em módulo) para trava de débito
SCORE_FORTE = 5         # força mínima (em módulo) para trava de crédito
DIST_ESTICADA = 0.08    # preço a mais de 8% da MM20 = entrada esticada

# ---- Nomes das estruturas ---------------------------------------------------
ALTA_DEB = "Trava de alta (call spread)"
ALTA_CRED = "Trava de alta com put (put spread)"
BAIXA_DEB = "Trava de baixa (put spread)"
BAIXA_CRED = "Trava de baixa com call (call spread)"
NEUTRA = "Neutra / lateral (iron condor)"
ESTRUTURAS = [ALTA_DEB, ALTA_CRED, BAIXA_DEB, BAIXA_CRED, NEUTRA]

DESCRICAO = {
    ALTA_DEB: "Compra de call + venda de call de strike maior (débito). Ganho e perda limitados.",
    ALTA_CRED: "Venda de put + compra de put de strike menor (crédito). Ganha se o ativo subir ou ficar parado.",
    BAIXA_DEB: "Compra de put + venda de put de strike menor (débito). Ganho e perda limitados.",
    BAIXA_CRED: "Venda de call + compra de call de strike maior (crédito). Ganha se o ativo cair ou ficar parado.",
    NEUTRA: "Venda de put e de call + compra de put e call mais afastadas (crédito). Ganha se o ativo ficar no meio.",
}

MMS = (("MM9", "d9"), ("MM20", "d20"), ("MM200", "d200"))


# ---- Utilidades -------------------------------------------------------------
def brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(v: float) -> str:
    return f"{v * 100:+.1f}%".replace(".", ",")


def _sgn(x) -> int:
    if pd.isna(x):
        return 0
    return 1 if x > 0 else -1 if x < 0 else 0


def _zero(x) -> float:
    return 0.0 if pd.isna(x) else float(x)


# ---- Leitura dos indicadores ------------------------------------------------
def analisar(df: pd.DataFrame) -> dict:
    """Resume o último candle (df vem de indicadores.calcular)."""
    u = df.iloc[-1]
    preco = float(u["Close"])
    t = df["HiLo_tendencia"].to_numpy()
    tend = int(t[-1])

    dias = 1  # candles seguidos na tendência atual do HiLo
    for x in t[-2::-1]:
        if x == t[-1]:
            dias += 1
        else:
            break

    rec = t[-(HORIZONTE + 1):]
    viradas = int(np.sum((np.diff(rec) != 0) & (rec[1:] != 0) & (rec[:-1] != 0)))

    mm9, mm20, mm200 = (float(u[k]) for k in ("MM9", "MM20", "MM200"))

    def dist(ref):
        return preco / ref - 1 if pd.notna(ref) else np.nan

    d9, d20, d200 = dist(mm9), dist(mm20), dist(mm200)

    alinh = 0
    if not np.isnan([mm9, mm20, mm200]).any():
        if mm9 > mm20 > mm200:
            alinh = 1
        elif mm9 < mm20 < mm200:
            alinh = -1

    # Força: HiLo pesa 2; MM9, MM20, MM200 e alinhamento pesam 1 cada (−6 a +6)
    score = 2 * tend + _sgn(d9) + _sgn(d20) + _sgn(d200) + alinh

    ret = np.log(df["Close"]).diff().tail(HORIZONTE)
    sigma = preco * float(ret.std()) * np.sqrt(HORIZONTE)  # movimento esperado de ~1 mês

    return {
        "preco": preco, "tend": tend, "dias": dias, "viradas": viradas,
        "ifr": float(u["IFR"]), "hilo": float(u["HiLo"]) if pd.notna(u["HiLo"]) else np.nan,
        "d9": d9, "d20": d20, "d200": d200, "alinh": alinh,
        "score": score, "sigma": sigma,
    }


# ---- Frases -----------------------------------------------------------------
def _lista(nomes: list[str]) -> str:
    return nomes[0] if len(nomes) == 1 else ", ".join(nomes[:-1]) + " e " + nomes[-1]


def _posicao_mms(a: dict) -> str:
    acima = [n for n, k in MMS if pd.notna(a[k]) and a[k] > 0]
    abaixo = [n for n, k in MMS if pd.notna(a[k]) and a[k] < 0]
    if len(acima) == 3:
        return "fechamento acima das três médias" + (" (alinhadas para alta)" if a["alinh"] == 1 else "")
    if len(abaixo) == 3:
        return "fechamento abaixo das três médias" + (" (alinhadas para baixa)" if a["alinh"] == -1 else "")
    partes = []
    if acima:
        partes.append(f"acima de {_lista(acima)}")
    if abaixo:
        partes.append(f"abaixo de {_lista(abaixo)}")
    return "fechamento " + " e ".join(partes) if partes else "médias indisponíveis"


def _tempo(a: dict) -> str:
    d = a["dias"]
    if d <= 3:
        return f"há {d} candle{'s' if d > 1 else ''} (sinal recente)"
    if d >= 20:
        return f"há {d} candles (tendência madura, risco de exaustão)"
    return f"há {d} candles"


def _base_alta(a: dict) -> str:
    return f"HiLo em Call {_tempo(a)}; {_posicao_mms(a)}; IFR {a['ifr']:.0f}."


def _base_baixa(a: dict) -> str:
    return f"HiLo em Put {_tempo(a)}; {_posicao_mms(a)}; IFR {a['ifr']:.0f}."


# ---- Direcional -------------------------------------------------------------
def direcional(a: dict) -> tuple[str | None, str, str]:
    """Devolve (tipo, rótulo, explicação). tipo: 'call', 'put', 'obs' ou None."""
    s, ifr = a["score"], a["ifr"]

    if s >= SCORE_DIRECIONAL:
        if ifr <= IFR_ALTO:
            txt = _base_alta(a) + " Tendência de alta com força e sem sobrecompra: contexto favorável à compra de call"
            if _zero(a["d20"]) > DIST_ESTICADA:
                txt += (f", mas o preço já está {pct(a['d20'])} da MM20 (entrada esticada): avalie esperar um recuo "
                        "à média ou usar trava")
            return "call", "Compra de Call", txt + f". Invalidação de referência: fechamento abaixo do HiLo ({brl(a['hilo'])})."
        txt = (_base_alta(a) + f" IFR acima de {IFR_ALTO}: sobrecompra de curto prazo, a call seca fica cara e exposta "
               "a realização de lucros. Melhor aguardar um recuo ou olhar a aba de travas.")
        return "obs", "Alta esticada", txt

    if s <= -SCORE_DIRECIONAL:
        if ifr >= IFR_BAIXO:
            txt = _base_baixa(a) + " Tendência de baixa com força e sem sobrevenda: contexto favorável à compra de put"
            if _zero(a["d20"]) < -DIST_ESTICADA:
                txt += (f", mas o preço já está {pct(a['d20'])} da MM20 (entrada esticada): avalie esperar um repique "
                        "à média ou usar trava")
            return "put", "Compra de Put", txt + f". Invalidação de referência: fechamento acima do HiLo ({brl(a['hilo'])})."
        txt = (_base_baixa(a) + f" IFR abaixo de {IFR_BAIXO}: sobrevenda de curto prazo, risco de repique contra a put. "
               "Melhor aguardar um repique ou olhar a aba de travas.")
        return "obs", "Baixa esticada", txt

    return None, "", ""


# ---- Travas -----------------------------------------------------------------
def travas(a: dict) -> list[dict]:
    """Lista de ideias de trava para o ativo: [{estrutura, strikes, texto}]."""
    p, sig, hilo, ifr, s = a["preco"], a["sigma"], a["hilo"], a["ifr"], a["score"]
    d20 = _zero(a["d20"])
    if np.isnan(sig) or sig <= 0:
        return []
    out = []

    if a["tend"] == 1 and d20 > 0 and s >= SCORE_TRAVA:
        base = _base_alta(a)
        if ifr > 65:
            motivo = (f"IFR já elevado: a trava reduz o custo do prêmio e a exposição a uma realização, "
                      "em troca de limitar o ganho ao strike vendido.")
        else:
            motivo = "A trava barateia a posição frente à call seca e já define o alvo no strike vendido."
        out.append({
            "estrutura": ALTA_DEB,
            "strikes": f"Compra call ≈ {brl(p)} · Venda call ≈ {brl(p + sig)}",
            "texto": f"{base} {motivo} Strike vendido ≈ +1σ (movimento esperado de ~1 mês). "
                     f"Invalidação: fechamento abaixo do HiLo ({brl(hilo)}).",
        })
        if s >= SCORE_FORTE and 45 <= ifr <= IFR_ALTO and d20 <= DIST_ESTICADA:
            venda = min(hilo, p - 0.5 * sig)
            out.append({
                "estrutura": ALTA_CRED,
                "strikes": f"Venda put ≈ {brl(venda)} · Compra put ≈ {brl(venda - 0.5 * sig)}",
                "texto": f"{base} Alta forte{' e com médias alinhadas' if a['alinh'] == 1 else ''}: vender put abaixo do HiLo/suporte recebe prêmio e ganha se o "
                         "ativo subir, andar de lado ou cair pouco; a put comprada limita a perda. "
                         f"Risco: perder o HiLo e romper o strike vendido ({brl(venda)}).",
            })

    elif a["tend"] == -1 and d20 < 0 and s <= -SCORE_TRAVA:
        base = _base_baixa(a)
        if ifr < 35:
            motivo = (f"IFR já baixo: a trava reduz o custo do prêmio e a exposição a um repique, "
                      "em troca de limitar o ganho ao strike vendido.")
        else:
            motivo = "A trava barateia a posição frente à put seca e já define o alvo no strike vendido."
        out.append({
            "estrutura": BAIXA_DEB,
            "strikes": f"Compra put ≈ {brl(p)} · Venda put ≈ {brl(p - sig)}",
            "texto": f"{base} {motivo} Strike vendido ≈ −1σ (movimento esperado de ~1 mês). "
                     f"Invalidação: fechamento acima do HiLo ({brl(hilo)}).",
        })
        if s <= -SCORE_FORTE and IFR_BAIXO <= ifr <= 55 and d20 >= -DIST_ESTICADA:
            venda = max(hilo, p + 0.5 * sig)
            out.append({
                "estrutura": BAIXA_CRED,
                "strikes": f"Venda call ≈ {brl(venda)} · Compra call ≈ {brl(venda + 0.5 * sig)}",
                "texto": f"{base} Baixa forte{' e com médias alinhadas' if a['alinh'] == -1 else ''}: vender call acima do HiLo/resistência recebe prêmio e ganha "
                         "se o ativo cair, andar de lado ou subir pouco; a call comprada limita a perda. "
                         f"Risco: perder o HiLo e romper o strike vendido ({brl(venda)}).",
            })

    elif a["viradas"] >= 2 and 40 <= ifr <= 60 and abs(d20) <= 0.03:
        out.append({
            "estrutura": NEUTRA,
            "strikes": (f"Put {brl(p - 1.5 * sig)}/{brl(p - sig)} · Call {brl(p + sig)}/{brl(p + 1.5 * sig)} "
                        "(compra/venda)"),
            "texto": f"HiLo virou {a['viradas']}x nos últimos {HORIZONTE} candles, preço a {pct(d20)} da MM20 e IFR "
                     f"neutro ({ifr:.0f}): sem tendência definida. O iron condor ganha se o ativo ficar entre os "
                     f"strikes vendidos (≈ ±1σ: {brl(p - sig)} a {brl(p + sig)}); o risco é limitado pelas pontas "
                     "compradas, mas há perda se houver rompimento forte para qualquer lado.",
        })

    return out
