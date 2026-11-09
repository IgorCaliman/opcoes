"""Operações abertas: guarda só o ativo e se a posição é Call ou Put (operacoes.csv)."""
import os
import re
import tempfile
import threading
from pathlib import Path

import pandas as pd

ARQUIVO = Path(__file__).parent / "operacoes.csv"
COLUNAS = ["ativo", "tipo"]
TIPOS = ["Call", "Put"]
# 4 letras + final de ação (3, 4, 5, 6), unit/ETF (11) ou BDR (32 a 35, 39): PETR4, BOVA11...
_PADRAO_TICKER = re.compile(r"^[A-Z]{4}(3|4|5|6|11|32|33|34|35|39)$")

# Evita que duas sessões do app leiam-modifiquem-gravem ao mesmo tempo e uma apague a mudança da outra.
_TRAVA = threading.Lock()


def carregar_operacoes() -> pd.DataFrame:
    """Lê as operações abertas (uma linha por ativo/tipo)."""
    if not ARQUIVO.exists():
        return pd.DataFrame(columns=COLUNAS)
    df = pd.read_csv(ARQUIVO, encoding="utf-8")
    if "tipo" not in df.columns and "estrutura" in df.columns:  # CSV no formato antigo
        df["tipo"] = df["estrutura"].str.contains("Put", case=False).map({True: "Put", False: "Call"})
    return df[COLUNAS].dropna().reset_index(drop=True)


def normalizar_ativo(texto: str) -> str:
    t = texto.strip().upper()
    return t[:-3] if t.endswith(".SA") else t


def _gravar(df: pd.DataFrame) -> None:
    """Grava o CSV de forma atômica: escreve num arquivo temporário e só então troca o original.
    Se o app cair no meio da gravação, o operacoes.csv antigo continua intacto (não fica vazio/corrompido)."""
    ARQUIVO.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ARQUIVO.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df[COLUNAS].to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ARQUIVO)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def adicionar_operacao(ativo: str, tipo: str) -> None:
    """Valida e grava a nova operação no CSV. Levanta ValueError se algo estiver errado."""
    ativo = normalizar_ativo(ativo)
    if not _PADRAO_TICKER.match(ativo):
        raise ValueError("Ticker inválido. Use o formato PETR4 ou BOVA11.")
    if tipo not in TIPOS:
        raise ValueError("Escolha Call ou Put.")

    with _TRAVA:
        ops = carregar_operacoes()  # relê o arquivo na hora de gravar, nunca usa uma cópia antiga
        if ((ops["ativo"] == ativo) & (ops["tipo"] == tipo)).any():
            raise ValueError(f"{ativo} já está nas operações como {tipo}.")
        nova = pd.DataFrame([{"ativo": ativo, "tipo": tipo}])
        _gravar(pd.concat([ops, nova], ignore_index=True))


def remover_operacoes(itens: list[tuple[str, str]]) -> int:
    """Remove operações fechadas, dadas como lista de (ativo, tipo). Devolve quantas linhas saíram."""
    with _TRAVA:
        ops = carregar_operacoes()
        alvo = {(normalizar_ativo(a), t) for a, t in itens}
        sai = [(a, t) in alvo for a, t in zip(ops["ativo"], ops["tipo"])]
        if not any(sai):
            raise ValueError("Operação não encontrada (talvez já tenha sido removida).")
        _gravar(ops[[not s for s in sai]])
        return sum(sai)
