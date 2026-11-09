"""Cotações diárias com cache em disco, download em lote e ajuste por proventos.

Cada ativo fica em dados/<TICKER>.csv com OHLC bruto (o Yahoo já ajusta por
desdobramento/grupamento), volume, dividendos e desdobramentos.

- Download em lote: vários tickers por chamada ao Yahoo (bem mais rápido que um a um).
- Incremental: depois da primeira carga, baixa só os últimos dias e junta ao arquivo.
- Desdobramento/grupamento novo é detectado sozinho e refaz o histórico daquele ativo.
- Ajuste por dividendos/JCP é feito na leitura (carregar_ajustado), a partir da coluna
  de dividendos: um provento novo reajusta o histórico inteiro sem baixar tudo de novo.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PASTA = Path(__file__).parent / "dados"
OHLC = ["Open", "High", "Low", "Close"]
COLUNAS = OHLC + ["Volume", "Dividends", "Stock Splits"]
ANOS_HISTORICO = 3   # histórico inicial (a MM200 precisa de folga)
MARGEM_DIAS = 5      # rebaixa alguns dias para trás: corrige o último candle,
                     # que pode ter sido salvo incompleto (pregão em andamento)
TAMANHO_LOTE = 40    # tickers por chamada ao Yahoo
TOL_DIVERGENCIA = 0.02  # fechamento salvo x novo diferindo mais que isso = histórico mudou


# ---- Disco ---------------------------------------------------------------------
def _arquivo(ticker: str) -> Path:
    return PASTA / f"{ticker}.csv"


def ler(ticker: str) -> pd.DataFrame | None:
    """Histórico bruto salvo em disco, ou None se não existir / estiver no formato antigo."""
    arq = _arquivo(ticker)
    if not arq.exists():
        return None
    try:
        df = pd.read_csv(arq, index_col="Date", parse_dates=True)
    except Exception:
        return None
    if df.empty or not set(COLUNAS).issubset(df.columns):
        return None  # formato antigo (sem dividendos): força recarga completa
    return df[COLUNAS]


def _salvar(ticker: str, df: pd.DataFrame) -> None:
    PASTA.mkdir(parents=True, exist_ok=True)
    df.to_csv(_arquivo(ticker))


def limpar_cache() -> None:
    """Apaga os arquivos salvos; o próximo acesso refaz o histórico completo."""
    for arq in PASTA.glob("*.csv"):
        arq.unlink()


# ---- Download --------------------------------------------------------------------
def _normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLUNAS:
        if c not in df.columns:
            df[c] = 0.0 if c in ("Dividends", "Stock Splits") else np.nan
    df = df[COLUNAS].dropna(subset=["Close"])
    if df.empty:
        return pd.DataFrame(columns=COLUNAS)
    df[["Dividends", "Stock Splits"]] = df[["Dividends", "Stock Splits"]].fillna(0.0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def _extrair(bruto: pd.DataFrame, simbolo: str) -> pd.DataFrame | None:
    """Separa um ticker do resultado do yf.download (com ou sem MultiIndex)."""
    if bruto is None or bruto.empty:
        return None
    if not isinstance(bruto.columns, pd.MultiIndex):
        return bruto
    for nivel in range(bruto.columns.nlevels):
        if simbolo in bruto.columns.get_level_values(nivel):
            return bruto.xs(simbolo, axis=1, level=nivel)
    return None


def _baixar_lote(tickers: list[str], inicio: pd.Timestamp | None) -> dict[str, pd.DataFrame]:
    """Baixa um grupo de tickers de uma vez. Devolve {ticker: DataFrame normalizado}."""
    saida: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), TAMANHO_LOTE):
        grupo = tickers[i:i + TAMANHO_LOTE]
        simbolos = [f"{t}.SA" for t in grupo]
        args = dict(interval="1d", auto_adjust=False, actions=True, progress=False,
                    group_by="ticker", threads=True)
        if inicio is None:
            args["period"] = f"{ANOS_HISTORICO}y"
        else:
            args["start"] = inicio.strftime("%Y-%m-%d")
        try:
            bruto = yf.download(simbolos if len(simbolos) > 1 else simbolos[0], **args)
        except Exception:
            continue
        for t, s in zip(grupo, simbolos):
            df = _extrair(bruto, s)
            if df is None:
                continue
            df = _normalizar(df)
            if not df.empty:
                saida[t] = df
    return saida


def _precisa_refazer(local: pd.DataFrame, novo: pd.DataFrame) -> bool:
    """True se o Yahoo reajustou o histórico (desdobramento/grupamento novo)."""
    novos_splits = set(novo.index[novo["Stock Splits"] != 0])
    conhecidos = set(local.index[local["Stock Splits"] != 0])
    if novos_splits - conhecidos:
        return True
    comum = local.index.intersection(novo.index)
    if len(comum) == 0:
        return False
    dif = (novo.loc[comum, "Close"] / local.loc[comum, "Close"] - 1).abs()
    return bool((dif > TOL_DIVERGENCIA).any())


def atualizar(tickers: list[str]) -> list[str]:
    """Atualiza o cache em disco de todos os tickers, em lote.

    Devolve a lista de tickers que ficaram sem dados (nem no disco, nem no Yahoo).
    Sem internet, mantém o que já está salvo.
    """
    tickers = list(dict.fromkeys(tickers))
    locais = {t: ler(t) for t in tickers}
    completos = [t for t, df in locais.items() if df is None]

    # Incrementais: agrupa por data de início para baixar em lote
    por_inicio: dict[pd.Timestamp, list[str]] = {}
    for t, df in locais.items():
        if df is not None:
            inicio = df.index[-1] - timedelta(days=MARGEM_DIAS)
            por_inicio.setdefault(inicio, []).append(t)

    for inicio, grupo in por_inicio.items():
        novos = _baixar_lote(grupo, inicio)
        for t in grupo:
            novo, local = novos.get(t), locais[t]
            if novo is None:
                continue  # sem resposta: fica com o que tem
            if _precisa_refazer(local, novo):
                completos.append(t)
                continue
            junto = pd.concat([local, novo])
            junto = junto[~junto.index.duplicated(keep="last")].sort_index()
            _salvar(t, junto)

    if completos:
        novos = _baixar_lote(completos, None)
        for t, df in novos.items():
            _salvar(t, df)

    return [t for t in tickers if ler(t) is None]


def obter(ticker: str) -> pd.DataFrame:
    """Histórico bruto de um ativo, atualizando o que falta."""
    atualizar([ticker])
    df = ler(ticker)
    if df is None:
        raise ValueError(f"Sem dados para {ticker}")
    return df


# ---- Ajuste por proventos --------------------------------------------------------
def fator_ajuste(df: pd.DataFrame) -> pd.Series:
    """Fator multiplicativo de ajuste por dividendos/JCP (mesmo método do Yahoo).

    Em cada data ex com provento D, todos os preços ANTERIORES são multiplicados por
    (1 − D / fechamento do pregão anterior). O último pregão sempre tem fator 1.
    """
    close = df["Close"]
    div = df["Dividends"].fillna(0.0)
    razao = div / close.shift(1)
    # Ignora proventos absurdos (erro de dado): mais de 50% do preço
    evento = (1 - razao).where((div > 0) & (razao > 0) & (razao < 0.5), 1.0)
    acumulado = evento[::-1].cumprod()[::-1]   # produto das datas ex >= t
    return acumulado.shift(-1).fillna(1.0)     # só as datas ex > t


def ajustar(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC ajustado por proventos (base para indicadores). Mantém o fechamento bruto."""
    out = df.copy()
    f = fator_ajuste(df)
    for c in OHLC:
        out[c] = df[c] * f
    out["Close_bruto"] = df["Close"]
    return out


def carregar_ajustado(ticker: str) -> pd.DataFrame:
    """Lê do disco (ou baixa, se não houver) e devolve o histórico ajustado."""
    df = ler(ticker)
    if df is None:
        df = obter(ticker)
    return ajustar(df)
