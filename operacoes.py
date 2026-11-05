"""Operações abertas: guarda só o ativo e se a posição é Call ou Put (operacoes.csv)."""
import re
from pathlib import Path

import pandas as pd

ARQUIVO = Path(__file__).parent / "operacoes.csv"
COLUNAS = ["ativo", "tipo"]
TIPOS = ["Call", "Put"]
# 4 letras + final de ação (3, 4, 5, 6), unit/ETF (11) ou BDR (32 a 35, 39): PETR4, BOVA11...
_PADRAO_TICKER = re.compile(r"^[A-Z]{4}(3|4|5|6|11|32|33|34|35|39)$")


def carregar_operacoes() -> pd.DataFrame:
    """Lê as operações abertas (uma linha por ativo/tipo)."""
    if not ARQUIVO.exists():
        return pd.DataFrame(columns=COLUNAS)
    df = pd.read_csv(ARQUIVO)
    if "tipo" not in df.columns and "estrutura" in df.columns:  # CSV no formato antigo
        df["tipo"] = df["estrutura"].str.contains("Put", case=False).map({True: "Put", False: "Call"})
    return df[COLUNAS].dropna().reset_index(drop=True)


def normalizar_ativo(texto: str) -> str:
    t = texto.strip().upper()
    return t[:-3] if t.endswith(".SA") else t


def adicionar_operacao(ativo: str, tipo: str) -> None:
    """Valida e grava a nova operação no CSV. Levanta ValueError se algo estiver errado."""
    ativo = normalizar_ativo(ativo)
    if not _PADRAO_TICKER.match(ativo):
        raise ValueError("Ticker inválido. Use o formato PETR4 ou BOVA11.")
    if tipo not in TIPOS:
        raise ValueError("Escolha Call ou Put.")

    ops = carregar_operacoes()
    if ((ops["ativo"] == ativo) & (ops["tipo"] == tipo)).any():
        raise ValueError(f"{ativo} já está nas operações como {tipo}.")

    nova = pd.DataFrame([{"ativo": ativo, "tipo": tipo}])
    (nova if ops.empty else pd.concat([ops, nova], ignore_index=True)).to_csv(ARQUIVO, index=False)