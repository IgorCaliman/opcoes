"""Regime de mercado e exposição das operações abertas.

Regime: combina o HiLo do BOVA11 com a "amplitude" do universo (quantos ativos estão em
sinal Call, acima da MM200 e da MM20). Alta com amplitude baixa é alta estreita, puxada
por poucos papéis; vale o mesmo para a baixa.

Exposição: as operações só têm ativo e Call/Put (sem tamanho), então cada uma pesa igual.
- Beta de cada ativo contra o BOVA11 e exposição beta líquida do livro (Call = +1, Put = −1).
- Correlação ajustada pela direção: duas puts em ativos correlacionados são a MESMA aposta;
  uma call e uma put em ativos correlacionados se protegem.
- Apostas independentes: n² / Σ dᵢdⱼρᵢⱼ. Onze operações muito correlacionadas na mesma
  direção podem equivaler a duas ou três apostas de verdade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INDICE = "BOVA11"
INDICES = {"BOVA11", "SMAL11"}  # ficam fora da amplitude
JANELA_BETA = 120               # pregões usados em beta e correlação
JANELA_AMPLITUDE = 120          # pregões no gráfico de amplitude
AMPLA, ESTREITA = 0.60, 0.40    # % do universo em Call que separa alta/baixa ampla de estreita

# Ticker -> (setor, grupo de fator). Ajuste à vontade; ticker sem entrada vira "Outros".
SETORES = {
    "BOVA11": ("Índice", "Índice"), "SMAL11": ("Índice small caps", "Índice"),
    "PETR3": ("Petróleo", "Petróleo"), "PETR4": ("Petróleo", "Petróleo"), "PRIO3": ("Petróleo", "Petróleo"),
    "CSAN3": ("Combustíveis/holding", "Petróleo"),
    "VALE3": ("Mineração", "Metálicas"), "CMIN3": ("Mineração", "Metálicas"), "BRAP4": ("Mineração", "Metálicas"),
    "CSNA3": ("Siderurgia", "Metálicas"), "GGBR4": ("Siderurgia", "Metálicas"), "USIM5": ("Siderurgia", "Metálicas"),
    "SUZB3": ("Papel e celulose", "Exportadoras"), "KLBN11": ("Papel e celulose", "Exportadoras"),
    "EMBJ3": ("Aeroespacial", "Exportadoras"), "WEGE3": ("Bens de capital", "Exportadoras"),
    "BEEF3": ("Proteína", "Exportadoras"), "MBRF3": ("Proteína", "Exportadoras"),
    "ITUB4": ("Bancos", "Financeiro"), "BBDC4": ("Bancos", "Financeiro"), "BBDC3": ("Bancos", "Financeiro"),
    "BBAS3": ("Bancos", "Financeiro"), "BPAC11": ("Bancos", "Financeiro"), "ITSA4": ("Holding bancária", "Financeiro"),
    "B3SA3": ("Bolsa", "Financeiro"), "BBSE3": ("Seguros", "Financeiro"),
    "MGLU3": ("Varejo", "Domésticas cíclicas"), "LREN3": ("Varejo moda", "Domésticas cíclicas"),
    "CEAB3": ("Varejo moda", "Domésticas cíclicas"), "AZZA3": ("Varejo moda", "Domésticas cíclicas"),
    "ASAI3": ("Varejo alimentar", "Domésticas cíclicas"), "NATU3": ("Cosméticos", "Domésticas cíclicas"),
    "MRVE3": ("Construção", "Domésticas cíclicas"), "CYRE3": ("Construção", "Domésticas cíclicas"),
    "DIRR3": ("Construção", "Domésticas cíclicas"), "JHSF3": ("Imobiliário", "Domésticas cíclicas"),
    "ALOS3": ("Shoppings", "Domésticas cíclicas"), "MULT3": ("Shoppings", "Domésticas cíclicas"),
    "COGN3": ("Educação", "Domésticas cíclicas"), "ANIM3": ("Educação", "Domésticas cíclicas"),
    "RENT3": ("Locação", "Domésticas cíclicas"), "VAMO3": ("Locação", "Domésticas cíclicas"),
    "POMO4": ("Bens de capital", "Domésticas cíclicas"),
    "RDOR3": ("Saúde", "Defensivas"), "HAPV3": ("Saúde", "Domésticas cíclicas"),
    "ABEV3": ("Bebidas", "Defensivas"), "TOTS3": ("Tecnologia", "Defensivas"),
    "AXIA3": ("Energia elétrica", "Utilities"), "CMIG4": ("Energia elétrica", "Utilities"),
    "EQTL3": ("Energia elétrica", "Utilities"), "SBSP3": ("Saneamento", "Utilities"),
}


def setor(ativo: str) -> tuple[str, str]:
    return SETORES.get(ativo, ("Outros", "Outros"))


def _retornos(historicos: dict[str, pd.DataFrame], ativos, janela: int) -> pd.DataFrame:
    """Retornos log diários alinhados por data, últimos `janela` pregões."""
    series = {a: np.log(historicos[a]["Close"]).diff() for a in ativos if a in historicos}
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="all").tail(janela)


# ---- Regime -----------------------------------------------------------------------
def amplitude(historicos: dict[str, pd.DataFrame], dias: int = JANELA_AMPLITUDE) -> pd.DataFrame:
    """% do universo (sem índices) em HiLo Call, acima da MM200 e acima da MM20, por dia."""
    tend, mm200, mm20 = {}, {}, {}
    for a, h in historicos.items():
        if a in INDICES:
            continue
        h = h.tail(dias + 5)
        tend[a] = (h["HiLo_tendencia"] == 1).where(h["HiLo_tendencia"] != 0)
        mm200[a] = (h["Close"] > h["MM200"]).where(h["MM200"].notna())
        mm20[a] = (h["Close"] > h["MM20"]).where(h["MM20"].notna())
    if not tend:
        return pd.DataFrame()
    out = pd.DataFrame({
        "pct_call": pd.DataFrame(tend).astype(float).mean(axis=1),
        "pct_mm200": pd.DataFrame(mm200).astype(float).mean(axis=1),
        "pct_mm20": pd.DataFrame(mm20).astype(float).mean(axis=1),
        "n": pd.DataFrame(tend).notna().sum(axis=1),
    })
    return out[out["n"] >= max(3, 0.5 * out["n"].max())].tail(dias)


def regime(indice: pd.DataFrame | None, amp: pd.DataFrame) -> dict:
    """Rótulo do regime, direção (+1, −1 ou 0) e explicação."""
    if indice is None or amp.empty:
        return {"rotulo": "Indefinido", "direcao": 0, "texto": "Sem dados do índice ou do universo."}
    u = indice.iloc[-1]
    t = int(u["HiLo_tendencia"])
    pc, p200 = float(amp["pct_call"].iloc[-1]), float(amp["pct_mm200"].iloc[-1])
    d200 = u["Close"] / u["MM200"] - 1 if pd.notna(u["MM200"]) else np.nan
    base = (f"{INDICE} em {'Call' if t == 1 else 'Put' if t == -1 else 'indefinido'} no HiLo"
            + ("" if pd.isna(d200) else f", {d200 * 100:+.1f}% da MM200".replace(".", ","))
            + f"; {pc * 100:.0f}% do universo em Call e {p200 * 100:.0f}% acima da MM200.")
    if t == 1 and pc >= AMPLA:
        return {"rotulo": "Alta ampla", "direcao": 1,
                "texto": base + " Tendência de alta disseminada: calls têm o vento a favor."}
    if t == 1:
        return {"rotulo": "Alta estreita", "direcao": 1,
                "texto": base + " Índice sobe com poucos papéis: calls só nos líderes, puts em quem já está fraco."}
    if t == -1 and pc <= ESTREITA:
        return {"rotulo": "Baixa ampla", "direcao": -1,
                "texto": base + " Queda disseminada: puts têm o vento a favor; calls são contra a maré."}
    if t == -1:
        return {"rotulo": "Baixa estreita", "direcao": -1,
                "texto": base + " Índice cai, mas boa parte do universo resiste: puts seletivas, sem exagero."}
    return {"rotulo": "Misto", "direcao": 0, "texto": base + " Sem direção clara do índice."}


def mercado_a_favor(tipo: str, reg: dict) -> str:
    """'A favor', 'Contra' ou 'Neutro' para uma ideia Call/Put dado o regime."""
    if reg.get("direcao", 0) == 0:
        return "Neutro"
    d = 1 if str(tipo).lower().startswith("c") else -1
    return "A favor" if d == reg["direcao"] else "Contra"


# ---- Exposição ---------------------------------------------------------------------
def betas(historicos: dict[str, pd.DataFrame], ativos, janela: int = JANELA_BETA) -> pd.Series:
    if INDICE not in historicos:
        return pd.Series(np.nan, index=list(ativos))
    r = _retornos(historicos, list(dict.fromkeys(list(ativos) + [INDICE])), janela)
    var = r[INDICE].var()
    return pd.Series({a: (r[a].cov(r[INDICE]) / var if a in r and var > 0 else np.nan) for a in ativos})


def correlacao(historicos: dict[str, pd.DataFrame], ativos, janela: int = JANELA_BETA) -> pd.DataFrame:
    r = _retornos(historicos, ativos, janela)
    return r.corr(min_periods=int(janela * 0.6)) if not r.empty else pd.DataFrame()


def exposicao(ops: pd.DataFrame, historicos: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """(tabela por operação, resumo do livro, matriz de correlação ajustada pela direção)."""
    if ops.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    ops = ops.copy()
    ops["dir"] = np.where(ops["tipo"] == "Call", 1, -1)
    ops["rotulo"] = ops["ativo"] + " (" + ops["tipo"].str[0] + ")"
    ops[["setor", "grupo"]] = [setor(a) for a in ops["ativo"]]
    b = betas(historicos, ops["ativo"].unique())
    ops["beta"] = ops["ativo"].map(b)
    ops["vol"] = [float(historicos[a]["Vol_EWMA"].iloc[-1]) if a in historicos else np.nan for a in ops["ativo"]]
    ops["hilo"] = [int(historicos[a]["HiLo_tendencia"].iloc[-1]) if a in historicos else 0 for a in ops["ativo"]]
    ops["hilo_a_favor"] = np.where(ops["hilo"] == 0, "—", np.where(ops["hilo"] == ops["dir"], "Sim", "Não"))

    c = correlacao(historicos, ops["ativo"].unique())
    if c.empty:
        adj = pd.DataFrame()
        n_eff = np.nan
    else:
        rho = np.array(c.reindex(index=ops["ativo"], columns=ops["ativo"]).to_numpy(), dtype=float)
        np.fill_diagonal(rho, 1.0)
        d = ops["dir"].to_numpy(dtype=float)
        m = np.outer(d, d) * np.nan_to_num(rho, nan=0.0)
        adj = pd.DataFrame(m, index=ops["rotulo"], columns=ops["rotulo"])
        soma = m.sum()
        n_eff = len(d) ** 2 / soma if soma > 0 else np.nan

    beta_liq = float(np.nansum(ops["dir"] * ops["beta"]) / len(ops))
    resumo = {
        "n": len(ops), "call": int((ops["dir"] == 1).sum()), "put": int((ops["dir"] == -1).sum()),
        "beta_liquido": beta_liq, "apostas_independentes": n_eff,
        "contra_hilo": int((ops["hilo_a_favor"] == "Não").sum()),
    }
    return ops, resumo, adj


def cenario(ops: pd.DataFrame, mov_indice: float) -> pd.DataFrame:
    """Movimento esperado de cada ativo (beta × movimento do índice) e se vai a favor da operação."""
    out = ops[["ativo", "tipo", "dir", "beta"]].copy()
    out["mov_esperado"] = out["beta"] * mov_indice
    out["efeito"] = out["dir"] * out["mov_esperado"]
    out["situacao"] = np.where(out["efeito"].isna(), "—", np.where(out["efeito"] >= 0, "A favor", "Contra"))
    return out.sort_values("efeito")
