"""Cadeia de opções da B3 a partir do arquivo oficial COTAHIST diário.

Fonte: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_DddmmAAAA.ZIP
(gratuito, publicado pela B3 no fim do dia). Para cada opção traz último, mínima,
máxima, melhor oferta de compra (bid) e de venda (ask) no fechamento, número de
negócios, volume, strike e vencimento. São dados de FIM DE DIA, não tempo real.

O arquivo de cada pregão é baixado uma vez e guardado em dados/cotahist/.
"""
from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import black_scholes as bs
import calendario_b3 as cal

PASTA = Path(__file__).parent / "dados" / "cotahist"
URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{:%d%m%Y}.ZIP"
MERCADOS = {"010": "Vista", "070": "Call", "080": "Put"}
DIAS_PROCURA = 7  # quantos pregões para trás procurar o arquivo mais recente

# Posições do layout oficial (base 1, inclusivas) -> fatias base 0
_CAMPOS = {
    "data": (3, 10), "codbdi": (11, 12), "codneg": (13, 24), "tpmerc": (25, 27), "especi": (40, 49),
    "abertura": (57, 69), "maxima": (70, 82), "minima": (83, 95), "medio": (96, 108),
    "ultimo": (109, 121), "bid": (122, 134), "ask": (135, 147), "negocios": (148, 152),
    "quantidade": (153, 170), "volume": (171, 188), "strike": (189, 201), "indopc": (202, 202),
    "vencimento": (203, 210), "fatcot": (211, 217),
}
_PRECOS = ["abertura", "maxima", "minima", "medio", "ultimo", "bid", "ask", "volume", "strike"]

# Raiz do código -> ação-objeto, quando a regra automática (maior volume) não serve
OBJETO_MANUAL: dict[str, str] = {}


# ---- Download e leitura -------------------------------------------------------------
def _baixar(pregao: date) -> bytes | None:
    url = URL.format(pregao)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    for contexto in (None, ssl._create_unverified_context()):
        # O servidor da B3 às vezes apresenta cadeia de certificado incompleta; dado público,
        # então tenta sem verificação só nesse caso.
        try:
            with urllib.request.urlopen(req, timeout=60, context=contexto) as r:
                conteudo = r.read()
            return conteudo if conteudo[:2] == b"PK" else None  # 'PK' = arquivo zip
        except ssl.SSLError:
            continue
        except Exception:
            return None
    return None


def interpretar(texto: str) -> pd.DataFrame:
    """Converte o TXT de largura fixa nas linhas de mercado à vista, calls e puts."""
    linhas = [l for l in texto.splitlines() if l[:2] == "01" and l[24:27] in MERCADOS]
    if not linhas:
        return pd.DataFrame(columns=list(_CAMPOS))
    df = pd.DataFrame({k: [l[a - 1:b] for l in linhas] for k, (a, b) in _CAMPOS.items()})
    df["codneg"] = df["codneg"].str.strip()
    df["especi"] = df["especi"].str.strip()
    df["mercado"] = df["tpmerc"].map(MERCADOS)
    fat = pd.to_numeric(df["fatcot"], errors="coerce").replace(0, 1).fillna(1)
    for c in _PRECOS:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    for c in ("abertura", "maxima", "minima", "medio", "ultimo", "bid", "ask"):
        df[c] = df[c] / fat  # cotação por lote de mil -> unitária
    for c in ("negocios", "quantidade"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce")
    df["vencimento"] = pd.to_datetime(df["vencimento"], format="%Y%m%d", errors="coerce")
    for c in ("bid", "ask", "ultimo"):
        df.loc[df[c] <= 0, c] = np.nan  # sem oferta / sem negócio vem zerado
    return df.drop(columns=["tpmerc", "fatcot"])


def carregar_pregao(pregao: date) -> pd.DataFrame | None:
    """COTAHIST de um pregão (do disco ou da B3). None se ainda não foi publicado."""
    arq = PASTA / f"COTAHIST_D{pregao:%Y%m%d}.csv"
    if arq.exists():
        return pd.read_csv(arq, parse_dates=["data", "vencimento"], dtype={"indopc": str})
    conteudo = _baixar(pregao)
    if conteudo is None:
        return None
    with zipfile.ZipFile(io.BytesIO(conteudo)) as z:
        texto = z.read(z.namelist()[0]).decode("latin-1")
    df = interpretar(texto)
    if df.empty:
        return None
    PASTA.mkdir(parents=True, exist_ok=True)
    df.to_csv(arq, index=False)
    return df


def ultimo_pregao(hoje: date | None = None) -> tuple[date, pd.DataFrame] | None:
    """O COTAHIST mais recente disponível (o do dia sai só à noite)."""
    for d in cal.pregoes_anteriores(hoje or date.today(), DIAS_PROCURA):
        df = carregar_pregao(d)
        if df is not None:
            return d, df
    return None


# ---- Cadeia por ativo ----------------------------------------------------------------
def objetos(df: pd.DataFrame) -> dict[str, str]:
    """Raiz de 4 letras -> ação-objeto. Escolhe a ação de maior volume com aquela raiz
    (PETR -> PETR4, ITUB -> ITUB4). Use OBJETO_MANUAL para corrigir exceções."""
    vista = df[df["mercado"] == "Vista"].copy()
    vista["raiz"] = vista["codneg"].str[:4]
    escolha = vista.sort_values("volume", ascending=False).drop_duplicates("raiz")
    mapa = dict(zip(escolha["raiz"], escolha["codneg"]))
    mapa.update(OBJETO_MANUAL)
    return mapa


def cadeia(df: pd.DataFrame, ativo: str, pregao: date, selic: float, vol_ref: float | None = None) -> pd.DataFrame:
    """Todas as opções do ativo com prazo, distância do strike, vol implícita e gregas.

    vol_ref (anual, ex.: a vol EWMA do ativo) é usada nas gregas quando a opção não tem
    preço suficiente para calcular a vol implícita.
    """
    raiz = ativo[:4]
    if objetos(df).get(raiz) != ativo:
        return pd.DataFrame()
    vista = df[(df["mercado"] == "Vista") & (df["codneg"] == ativo)]
    if vista.empty or pd.isna(vista["ultimo"].iloc[0]):
        return pd.DataFrame()
    spot = float(vista["ultimo"].iloc[0])

    op = df[df["mercado"].isin(["Call", "Put"]) & (df["codneg"].str[:4] == raiz)].copy()
    op = op[op["vencimento"] >= pd.Timestamp(pregao)]
    if op.empty:
        return pd.DataFrame()

    op["du"] = [cal.dias_uteis(pregao, v) for v in op["vencimento"]]
    op["semanal"] = [cal.e_semanal(v) for v in op["vencimento"]]
    op["T"] = op["du"] / bs.DIAS_ANO
    op["spot"] = spot
    op["dist"] = op["strike"] / spot - 1
    otm = (op["mercado"] == "Call") == (op["strike"] > spot)
    op["moneyness"] = np.where(np.isclose(op["strike"], spot, rtol=0.01), "ATM", np.where(otm, "OTM", "ITM"))
    op["meio"] = np.where(op["bid"].notna() & op["ask"].notna(), (op["bid"] + op["ask"]) / 2, np.nan)
    op["spread"] = (op["ask"] - op["bid"]) / op["meio"]
    op["ref"] = op["meio"].fillna(op["ultimo"])  # preço usado para a vol implícita

    r = bs.taxa_continua(selic)
    op["iv"] = bs.vol_implicita(op["ref"].fillna(0).to_numpy(), spot, op["strike"].to_numpy(),
                                op["T"].to_numpy(), r, op["mercado"].to_numpy())
    vol_gregas = op["iv"].fillna(vol_ref if vol_ref else np.nan).to_numpy()
    g = bs.gregas(spot, op["strike"].to_numpy(), op["T"].to_numpy(), r, vol_gregas, op["mercado"].to_numpy())
    for k, v in g.items():
        op[k] = v
    op["tipo"] = op["mercado"]
    colunas = ["codneg", "tipo", "vencimento", "du", "strike", "dist", "moneyness", "spot", "bid", "ask",
               "meio", "ultimo", "minima", "maxima", "negocios", "volume", "spread", "iv", "delta",
               "gama", "theta", "vega", "prob_exercicio", "semanal", "especi", "indopc"]
    return op[colunas].rename(columns={"codneg": "opcao"}).sort_values(["vencimento", "tipo", "strike"])


def resumo_liquidez(df: pd.DataFrame) -> pd.DataFrame:
    """Negócios e volume em opções por ação-objeto no pregão (para ranquear o que é operável)."""
    mapa = objetos(df)
    op = df[df["mercado"].isin(["Call", "Put"])].copy()
    op["ativo"] = op["codneg"].str[:4].map(mapa)
    return (op.groupby("ativo")
              .agg(series=("codneg", "count"), negocios=("negocios", "sum"), volume=("volume", "sum"))
              .sort_values("volume", ascending=False))
