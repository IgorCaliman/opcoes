"""Indicadores técnicos e de volatilidade. Fonte única: alvos.py e estrategias.py
usam as funções daqui, para que IFR, ATR e bandas sejam iguais em todas as abas."""
from __future__ import annotations

import numpy as np
import pandas as pd

HILO_PERIODOS = 13
IFR_PERIODOS = 9        # igual ao IFR (RSI) [9] da plataforma
BB_PERIODOS = 20
BB_DESVIOS = 2.0
ATR_PERIODOS = 14
EWMA_LAMBDA = 0.94      # RiskMetrics: dá ~94% de peso ao histórico, reage rápido sem ser ruidoso
HV_JANELAS = (20, 60, 252)
DIAS_ANO = 252


# ---- Tendência ---------------------------------------------------------------------
def hilo(df: pd.DataFrame, periodos: int = HILO_PERIODOS) -> pd.DataFrame:
    """Gann HiLo Activator.

    Média das máximas e das mínimas de `periodos` candles. Fechamento acima da
    média das máximas anterior -> tendência de alta (linha = média das mínimas).
    Fechamento abaixo da média das mínimas anterior -> tendência de baixa
    (linha = média das máximas). Entre as duas, mantém a tendência anterior.
    """
    media_max = df["High"].rolling(periodos).mean()
    media_min = df["Low"].rolling(periodos).mean()
    rmax = media_max.shift(1).to_numpy()
    rmin = media_min.shift(1).to_numpy()
    close = df["Close"].to_numpy()
    tendencia = np.zeros(len(df))  # +1 alta, -1 baixa, 0 indefinido

    for i in range(len(df)):
        anterior = tendencia[i - 1] if i > 0 else 0
        if np.isnan(rmax[i]) or np.isnan(rmin[i]):
            tendencia[i] = 0
        elif close[i] > rmax[i]:
            tendencia[i] = 1
        elif close[i] < rmin[i]:
            tendencia[i] = -1
        else:
            tendencia[i] = anterior

    linha = np.where(tendencia == 1, media_min, np.where(tendencia == -1, media_max, np.nan))
    return pd.DataFrame({"HiLo": linha, "HiLo_tendencia": tendencia}, index=df.index)


# ---- Osciladores e faixas ----------------------------------------------------------
def ifr(close: pd.Series, periodos: int = IFR_PERIODOS) -> pd.Series:
    """IFR (RSI) com suavização de Wilder. Sem perdas no período, vale 100."""
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / periodos, adjust=False, min_periods=periodos).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / periodos, adjust=False, min_periods=periodos).mean()
    rs = ganho / perda.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(~(perda == 0) | ganho.isna(), 100.0)


def atr(df: pd.DataFrame, periodos: int = ATR_PERIODOS) -> pd.Series:
    """Average True Range (Wilder)."""
    c_ant = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c_ant).abs(), (df["Low"] - c_ant).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / periodos, adjust=False).mean()


def bollinger(close: pd.Series, periodos: int = BB_PERIODOS, desvios: float = BB_DESVIOS) -> pd.DataFrame:
    """Média, banda superior, banda inferior e %B (0 = banda inferior, 1 = superior)."""
    media = close.rolling(periodos).mean()
    dp = close.rolling(periodos).std(ddof=0)
    sup, inf_ = media + desvios * dp, media - desvios * dp
    return pd.DataFrame({"BB_media": media, "BB_sup": sup, "BB_inf": inf_,
                         "PctB": (close - inf_) / (sup - inf_)}, index=close.index)


# ---- Volatilidade -------------------------------------------------------------------
def retornos_log(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def vol_ewma(close: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """Volatilidade diária EWMA (RiskMetrics). Menos ruidosa que a janela curta fixa."""
    r2 = retornos_log(close) ** 2
    return np.sqrt(r2.ewm(alpha=1 - lam, adjust=False, min_periods=20).mean())


def vol_historica(close: pd.Series, janela: int) -> pd.Series:
    """Volatilidade histórica anualizada (desvio-padrão dos retornos log na janela)."""
    return retornos_log(close).rolling(janela).std() * np.sqrt(DIAS_ANO)


def movimento_esperado(preco: float, vol_diaria: float, dias: int) -> float:
    """1σ em reais para um horizonte de `dias` pregões."""
    return preco * vol_diaria * np.sqrt(dias)


# ---- Tudo junto ---------------------------------------------------------------------
def calcular(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.join(hilo(out, HILO_PERIODOS))
    out["IFR"] = ifr(out["Close"], IFR_PERIODOS)
    for n in (9, 20, 200):
        out[f"MM{n}"] = out["Close"].rolling(n).mean()
    out["ATR"] = atr(out)
    out = out.join(bollinger(out["Close"]))
    out["Vol_EWMA_d"] = vol_ewma(out["Close"])
    out["Vol_EWMA"] = out["Vol_EWMA_d"] * np.sqrt(DIAS_ANO)
    for n in HV_JANELAS:
        out[f"HV{n}"] = vol_historica(out["Close"], n)
    return out


def sinal(df: pd.DataFrame) -> str:
    """Call se o último fechamento está acima do HiLo, Put se abaixo."""
    t = df["HiLo_tendencia"].iloc[-1]
    if t == 1:
        return "Call"
    if t == -1:
        return "Put"
    return "—"


def distancias_mm(ult: pd.Series) -> dict:
    """Distância (%) do último fechamento às MM9, MM20, MM200 e à média das três.

    `ult` é a última linha do DataFrame de `calcular`. Se alguma média não
    existir (histórico curto), a média das três e a distância a ela ficam NaN.
    """
    preco = ult["Close"]
    media = float(np.mean([ult["MM9"], ult["MM20"], ult["MM200"]]))

    def dist(ref):
        return preco / ref - 1 if pd.notna(ref) else np.nan

    return {
        "dist_mm9": dist(ult["MM9"]),
        "dist_mm20": dist(ult["MM20"]),
        "dist_mm200": dist(ult["MM200"]),
        "media_mm": media,
        "dist_media": dist(media),
    }
