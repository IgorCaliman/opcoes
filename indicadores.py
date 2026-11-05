import numpy as np
import pandas as pd

HILO_PERIODOS = 13
IFR_PERIODOS = 9  # igual ao IFR (RSI) [9] da plataforma


def hilo(df: pd.DataFrame, periodos: int = HILO_PERIODOS) -> pd.DataFrame:
    """Gann HiLo Activator.

    Média das máximas e das mínimas de `periodos` candles. Fechamento acima da
    média das máximas anterior -> tendência de alta (linha = média das mínimas).
    Fechamento abaixo da média das mínimas anterior -> tendência de baixa
    (linha = média das máximas). Entre as duas, mantém a tendência anterior.
    """
    media_max = df["High"].rolling(periodos).mean()
    media_min = df["Low"].rolling(periodos).mean()
    ref_max = media_max.shift(1)
    ref_min = media_min.shift(1)

    close = df["Close"].to_numpy()
    rmax, rmin = ref_max.to_numpy(), ref_min.to_numpy()
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
    return pd.DataFrame(
        {"HiLo": linha, "HiLo_tendencia": tendencia}, index=df.index
    )


def ifr(close: pd.Series, periodos: int = IFR_PERIODOS) -> pd.Series:
    """IFR (RSI) com suavização de Wilder."""
    delta = close.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.ewm(alpha=1 / periodos, adjust=False, min_periods=periodos).mean()
    media_perda = perda.ewm(alpha=1 / periodos, adjust=False, min_periods=periodos).mean()
    rs = media_ganho / media_perda
    return 100 - 100 / (1 + rs)


def calcular(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.join(hilo(out, HILO_PERIODOS))
    out["IFR"] = ifr(out["Close"], IFR_PERIODOS)
    for n in (9, 20, 200):
        out[f"MM{n}"] = out["Close"].rolling(n).mean()
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