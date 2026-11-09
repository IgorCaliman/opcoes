"""Alvos e gestão de saída para as ideias direcionais (aba "Acompanhamento - direcional").

Para cada ativo, na direção da ideia (call = +1, put = -1), calcula:
- Alvo: banda de Bollinger do lado da operação (superior na call, inferior na put).
- Espaço até o alvo e risco até um stop curto (Chandelier: máxima de 22 candles − 3×ATR na call).
- Relação alvo/stop: quanto de movimento a favor resta para cada 1% de risco.
- %B: posição do preço dentro das bandas (0 = banda inferior, 1 = banda superior).
- Percentil de esticamento: onde a distância atual à MM20 cai no histórico do próprio ativo.
- Histórico condicional: retorno a favor nos 5 candles seguintes, em situações parecidas no passado
  (mesma faixa de %B e preço do mesmo lado da MM20).
- Sinais de saída: fechamento voltando para dentro da banda, IFR saindo da sobrecompra/sobrevenda
  e divergência de IFR.

Todos os parâmetros ficam aqui no topo para ajuste.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BB_PERIODOS = 20
BB_DESVIOS = 2.0
ATR_PERIODOS = 14
CHANDELIER_JANELA = 22
CHANDELIER_MULT = 3.0
IFR_PERIODOS = 9
IFR_ALTO, IFR_BAIXO = 70, 30
HORIZONTE_FWD = 5          # candles à frente no histórico condicional
MIN_AMOSTRAS = 15          # abaixo disso o histórico condicional não é mostrado
PCT_ESTICADO = 90          # percentil a partir do qual o preço é considerado esticado
RR_MINIMA = 1.0            # relação alvo/stop abaixo disso é desfavorável
JANELA_SINAL = 3           # sinais de saída valem se ocorreram nos últimos N candles


# ---- Indicadores --------------------------------------------------------------
def _ifr(c: pd.Series, n: int = IFR_PERIODOS) -> pd.Series:
    d = c.diff()
    ganho = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perda = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ganho / perda.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def _atr(df: pd.DataFrame, n: int = ATR_PERIODOS) -> pd.Series:
    c_ant = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c_ant).abs(), (df["Low"] - c_ant).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _faixa_pctb(x: float, direcao: int) -> tuple[float, float]:
    """Faixa de %B usada para achar situações parecidas no histórico."""
    if direcao == 1:
        if x >= 1:
            return 1.0, np.inf
        if x >= 0.8:
            return 0.8, 1.0
        return -np.inf, 0.8
    if x <= 0:
        return -np.inf, 0.0
    if x <= 0.2:
        return 0.0, 0.2
    return 0.2, np.inf


# ---- Cálculo principal --------------------------------------------------------
def calcular(df: pd.DataFrame, direcao: int) -> dict:
    """df com colunas High, Low, Close (diário). direcao: 1 para call, -1 para put."""
    df = df.dropna(subset=["High", "Low", "Close"])
    vazio = {"ok": False}
    if len(df) < max(BB_PERIODOS, CHANDELIER_JANELA) + 30:
        return vazio

    c, h, l = df["Close"], df["High"], df["Low"]
    mm20 = c.rolling(BB_PERIODOS).mean()
    dp = c.rolling(BB_PERIODOS).std(ddof=0)
    sup, inf_ = mm20 + BB_DESVIOS * dp, mm20 - BB_DESVIOS * dp
    pctb = (c - inf_) / (sup - inf_)
    atr = _atr(df)
    ifr = _ifr(c)

    p = float(c.iloc[-1])
    if direcao == 1:
        alvo = float(sup.iloc[-1])
        stop = float(h.rolling(CHANDELIER_JANELA).max().iloc[-1] - CHANDELIER_MULT * atr.iloc[-1])
        espaco = alvo / p - 1
        risco = 1 - stop / p
    else:
        alvo = float(inf_.iloc[-1])
        stop = float(l.rolling(CHANDELIER_JANELA).min().iloc[-1] + CHANDELIER_MULT * atr.iloc[-1])
        espaco = 1 - alvo / p
        risco = stop / p - 1
    rr = espaco / risco if (risco > 0 and espaco > 0) else (0.0 if risco > 0 else np.nan)

    # Percentil de esticamento: distância à MM20 no histórico do próprio ativo, no sentido da operação
    dist = (c / mm20 - 1).dropna() * direcao
    percentil = float((dist <= dist.iloc[-1]).mean() * 100)

    # Histórico condicional
    fwd = (c.shift(-HORIZONTE_FWD) / c - 1) * direcao
    lo, hi = _faixa_pctb(float(pctb.iloc[-1]), direcao)
    cond = (pctb >= lo) & (pctb < hi) & (np.sign(c - mm20) == direcao)
    amostra = fwd[cond].dropna()
    n = len(amostra)
    med = float(amostra.median()) if n >= MIN_AMOSTRAS else np.nan
    acerto = float((amostra > 0).mean() * 100) if n >= MIN_AMOSTRAS else np.nan

    # Sinais de saída
    sinais = []
    j = JANELA_SINAL
    if direcao == 1:
        fora = pctb > 1
        if fora.iloc[-j - 1:-1].any() and not fora.iloc[-1]:
            sinais.append("fechou de volta para dentro da banda superior")
        cruzou = (ifr.shift() > IFR_ALTO) & (ifr <= IFR_ALTO)
        if cruzou.iloc[-j:].any():
            sinais.append(f"IFR saiu da sobrecompra ({IFR_ALTO})")
        topo_recente = c.iloc[-20:].idxmax() in c.index[-j:]
        ifr_max = ifr.iloc[-20:]
        if topo_recente and ifr_max.idxmax() not in ifr.index[-j:] and ifr.iloc[-1] < ifr_max.max() - 5:
            sinais.append("divergência: preço na máxima de 20 candles com IFR abaixo do pico anterior")
    else:
        fora = pctb < 0
        if fora.iloc[-j - 1:-1].any() and not fora.iloc[-1]:
            sinais.append("fechou de volta para dentro da banda inferior")
        cruzou = (ifr.shift() < IFR_BAIXO) & (ifr >= IFR_BAIXO)
        if cruzou.iloc[-j:].any():
            sinais.append(f"IFR saiu da sobrevenda ({IFR_BAIXO})")
        fundo_recente = c.iloc[-20:].idxmin() in c.index[-j:]
        ifr_min = ifr.iloc[-20:]
        if fundo_recente and ifr_min.idxmin() not in ifr.index[-j:] and ifr.iloc[-1] > ifr_min.min() + 5:
            sinais.append("divergência: preço na mínima de 20 candles com IFR acima do fundo anterior")

    return {
        "ok": True, "preco": p, "alvo": alvo, "stop": stop, "espaco": espaco, "risco": risco, "rr": rr,
        "pctb": float(pctb.iloc[-1]), "percentil": percentil, "med_fwd": med, "acerto_fwd": acerto,
        "n_fwd": n, "sinais": sinais, "direcao": direcao,
    }


# ---- Leitura em texto ---------------------------------------------------------
def _pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:+.1f}%".replace(".", ",")


def _brl(x: float) -> str:
    return "—" if pd.isna(x) else ("R$ " + f"{x:,.2f}").replace(",", "X").replace(".", ",").replace("X", ".")


def status(r: dict) -> str:
    """Rótulo curto para a coluna 'Leitura'."""
    if not r.get("ok"):
        return "—"
    if r["risco"] <= 0:
        return "Stop curto rompido"
    if r["sinais"]:
        return "Sinal de saída"
    fora = r["pctb"] > 1 if r["direcao"] == 1 else r["pctb"] < 0
    if fora:
        return "Acima do alvo" if r["direcao"] == 1 else "Abaixo do alvo"
    if r["rr"] < RR_MINIMA or r["percentil"] >= PCT_ESTICADO:
        return "Esticado"
    return "Com espaço"


def texto(r: dict) -> str:
    """Explicação em texto (sem o nome do ativo, que a função explicacoes() do app já coloca)."""
    if not r.get("ok"):
        return "Histórico insuficiente para calcular alvo e stop."
    lado = "superior" if r["direcao"] == 1 else "inferior"
    stop_txt = (f"stop curto em {_brl(r['stop'])}, já rompido" if r["risco"] <= 0
                else f"stop curto em {_brl(r['stop'])}, risco de {_pct(-r['risco'])}")
    partes = [f"Alvo (banda {lado}) em {_brl(r['alvo'])}, {_pct(r['espaco'])} do preço; {stop_txt}"]
    if r["risco"] <= 0:
        partes.append("o preço já está além do stop curto: a perna recente perdeu força, mesmo com o HiLo ainda a favor")
    elif r["espaco"] <= 0:
        partes.append(f"o preço já passou da banda {lado}. Em tendência forte ele pode seguir 'andando na banda', "
                      f"então o gatilho mais útil é o fechamento voltar para dentro dela")
    else:
        rr = f"{r['rr']:.1f}".replace(".", ",")
        partes.append(f"relação alvo/stop de {rr}x" + (" (desfavorável: pouco espaço frente ao risco)"
                                                       if r["rr"] < RR_MINIMA else ""))
    partes.append(f"distância à MM20 no percentil {r['percentil']:.0f} do histórico do ativo"
                  + (" (esticado)" if r["percentil"] >= PCT_ESTICADO else ""))
    if not pd.isna(r["med_fwd"]):
        partes.append(f"em {r['n_fwd']} situações parecidas, o retorno a favor em {HORIZONTE_FWD} candles teve mediana "
                      f"{_pct(r['med_fwd'])} e foi positivo em {r['acerto_fwd']:.0f}% das vezes")
    else:
        partes.append(f"poucas situações parecidas no histórico ({r['n_fwd']}) para estimar o comportamento seguinte")
    if r["sinais"]:
        partes.append("sinal de saída: " + "; ".join(r["sinais"]))
    return ". ".join(p[0].upper() + p[1:] for p in partes) + "."
