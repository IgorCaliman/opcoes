"""Cotações diárias com cache em disco.

Cada ativo fica em dados/<TICKER>.csv. Na primeira vez baixa o histórico
completo; depois baixa só os últimos dias e junta ao que já está salvo.
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PASTA = Path(__file__).parent / "dados"
COLUNAS = ["Open", "High", "Low", "Close", "Volume"]
ANOS_HISTORICO = 3  # histórico inicial (a MM200 precisa de folga)
MARGEM_DIAS = 5     # rebaixa alguns dias para trás: corrige o último candle,
                    # que pode ter sido salvo incompleto (pregão em andamento)


def _arquivo(ticker: str) -> Path:
    return PASTA / f"{ticker}.csv"


def _ler(ticker: str) -> pd.DataFrame | None:
    arq = _arquivo(ticker)
    if not arq.exists():
        return None
    df = pd.read_csv(arq, index_col="Date", parse_dates=True)
    return df if not df.empty else None


def _baixar(ticker: str, inicio: pd.Timestamp | None = None) -> pd.DataFrame:
    args = dict(interval="1d", auto_adjust=False, progress=False)
    if inicio is None:
        args["period"] = f"{ANOS_HISTORICO}y"
    else:
        args["start"] = inicio.strftime("%Y-%m-%d")
    df = yf.download(f"{ticker}.SA", **args)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame(columns=COLUNAS)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    return df[COLUNAS]


def obter(ticker: str) -> pd.DataFrame:
    """Devolve o histórico do ativo, atualizando só o que falta."""
    local = _ler(ticker)
    try:
        inicio = None if local is None else local.index[-1] - timedelta(days=MARGEM_DIAS)
        novo = _baixar(ticker, inicio)
    except Exception:
        if local is None:
            raise
        return local  # sem internet: usa o que já está salvo

    if novo.empty:
        if local is None:
            raise ValueError(f"Sem dados para {ticker}")
        return local

    completo = novo if local is None else pd.concat([local, novo])
    completo = completo[~completo.index.duplicated(keep="last")].sort_index()
    PASTA.mkdir(parents=True, exist_ok=True)
    completo.to_csv(_arquivo(ticker))
    return completo


def limpar_cache() -> None:
    """Apaga os arquivos salvos; o próximo acesso refaz o histórico completo."""
    for arq in PASTA.glob("*.csv"):
        arq.unlink()
