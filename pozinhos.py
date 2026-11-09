"""Pozinhos: opções fora do dinheiro (OTM) muito baratas, tratadas como bilhetes de
assimetria. A lista vem do COTAHIST (opcoes_dados.py); aqui cada pozinho ganha
métricas para separar bilhete com chance real de bilhete perdido:

- Alinhamento com o HiLo: call em ativo com sinal Call, put em ativo com sinal Put.
- Breakeven e distância até ele (strike ± prêmio pago).
- σ até o breakeven: quantos desvios-padrão (vol EWMA do ativo) o preço precisa andar
  até o vencimento. Acima de ~2σ é aposta de cauda.
- Prob. B&S: probabilidade neutra a risco (vol implícita, ou a EWMA se não houver) de
  vencer além do breakeven. É o que o preço da opção "embute".
- Prob. empírica: no histórico do próprio ativo, em quantas janelas de mesmo prazo o
  preço terminou além do breakeven (no vencimento) e em quantas tocou o breakeven em
  algum momento (toque: relevante porque você zera antes com 60-80% de lucro).
- Retorno esperado empírico: aplica cada retorno histórico de mesmo prazo ao preço de
  hoje, calcula o payoff da opção no vencimento e tira a média, dividido pelo prêmio.
  Positivo = no histórico, esse bilhete pagou mais do que custou. Inclui caudas gordas,
  mas as janelas se sobrepõem e o passado pode não se repetir.
- Multiplicador em 2σ: quantas vezes o prêmio a opção vale no vencimento se o ativo
  andar 2σ a favor.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import black_scholes as bs
import opcoes_dados

PRECO_MAX = 0.15     # teto do pozinho (R$)
DU_MAX = 60          # prazo máximo considerado (dias úteis)
MIN_JANELAS = 100    # abaixo disso as probabilidades empíricas não são mostradas
FAIXAS = {"Até R$ 0,05": (0.0, 0.05), "R$ 0,06 a 0,10": (0.055, 0.10), "R$ 0,11 a 0,15": (0.105, 0.15)}


# ---- Histórico do ativo ------------------------------------------------------------
def janelas(hist: pd.DataFrame, du: int) -> dict | None:
    """Retornos de `du` pregões à frente (no fechamento e no extremo do período), a partir
    de cada dia do histórico ajustado. Calculado uma vez por (ativo, prazo)."""
    if du < 1 or len(hist) < du + MIN_JANELAS:
        return None
    c = hist["Close"].to_numpy()
    h = hist["High"].to_numpy()
    l = hist["Low"].to_numpy()
    n = len(c) - du
    fim = c[du:du + n] / c[:n] - 1
    # máxima/mínima nos `du` pregões seguintes a cada dia
    h_fwd = pd.Series(h).rolling(du).max().shift(-du).to_numpy()[:n]
    l_fwd = pd.Series(l).rolling(du).min().shift(-du).to_numpy()[:n]
    return {"fim": fim, "max": h_fwd / c[:n] - 1, "min": l_fwd / c[:n] - 1, "n": n}


def metricas_empiricas(jan: dict | None, spot: float, strike: float, premio: float,
                       be: float, direcao: int) -> tuple[float, float, float, int]:
    """(prob. no vencimento além do breakeven, prob. de toque, retorno esperado, nº de janelas)."""
    if jan is None:
        return np.nan, np.nan, np.nan, 0
    d_be = be / spot - 1
    if direcao == 1:
        p_venc = float(np.mean(jan["fim"] >= d_be))
        p_toque = float(np.nanmean(jan["max"] >= d_be))
        payoff = np.maximum(spot * (1 + jan["fim"]) - strike, 0)
    else:
        p_venc = float(np.mean(jan["fim"] <= d_be))
        p_toque = float(np.nanmean(jan["min"] <= d_be))
        payoff = np.maximum(strike - spot * (1 + jan["fim"]), 0)
    ret = float(payoff.mean() / premio - 1)
    return p_venc, p_toque, ret, jan["n"]


# ---- Lista ------------------------------------------------------------------------
def listar(cotahist: pd.DataFrame, pregao: date, selic: float, historicos: dict[str, pd.DataFrame],
           analises: dict[str, dict], preco_max: float = PRECO_MAX, du_max: int = DU_MAX) -> pd.DataFrame:
    """Todos os pozinhos dos ativos em `historicos`, com as métricas acima.

    historicos: {ativo: DataFrame de indicadores.calcular (ajustado)}
    analises:   {ativo: dict de estrategias.analisar} (sinal HiLo e força)
    """
    r = bs.taxa_continua(selic)
    linhas = []
    for ativo, hist in historicos.items():
        vol = hist["Vol_EWMA"].iloc[-1] if "Vol_EWMA" in hist.columns else np.nan
        vol = float(vol) if pd.notna(vol) and vol > 0 else None
        cad = opcoes_dados.cadeia(cotahist, ativo, pregao, selic, vol)
        if cad.empty:
            continue
        cad["premio"] = cad["ask"].fillna(cad["ultimo"])  # quanto custaria entrar
        cad = cad[(cad["moneyness"] == "OTM") & (cad["premio"] > 0) & (cad["premio"] <= preco_max + 1e-9)
                  & (cad["du"] >= 1) & (cad["du"] <= du_max)]
        if cad.empty:
            continue

        an = analises.get(ativo, {})
        tend = an.get("tend", 0)
        cache_jan: dict[int, dict | None] = {}
        for o in cad.itertuples():
            direcao = 1 if o.tipo == "Call" else -1
            be = o.strike + direcao * o.premio
            T = o.du / bs.DIAS_ANO
            vol_bs = o.iv if pd.notna(o.iv) else vol
            if vol_bs:
                prob_bs = float(bs.gregas(o.spot, be, T, r, vol_bs, o.tipo)["prob_exercicio"])
            else:
                prob_bs = np.nan
            sig_be = (abs(np.log(be / o.spot)) / (vol * np.sqrt(T))) if vol else np.nan
            mult_2s = np.nan
            if vol:
                alvo = o.spot * np.exp(direcao * 2 * vol * np.sqrt(T))
                mult_2s = max(direcao * (alvo - o.strike), 0) / o.premio

            if o.du not in cache_jan:
                cache_jan[o.du] = janelas(hist, o.du)
            p_venc, p_toque, ret_emp, n_jan = metricas_empiricas(cache_jan[o.du], o.spot, o.strike,
                                                                  o.premio, be, direcao)
            linhas.append({
                "opcao": o.opcao, "ativo": ativo, "tipo": o.tipo, "vencimento": o.vencimento, "du": o.du,
                "strike": o.strike, "spot": o.spot, "dist": o.dist, "premio": o.premio, "bid": o.bid,
                "ask": o.ask, "ultimo": o.ultimo, "negocios": o.negocios, "volume": o.volume,
                "breakeven": be, "dist_be": be / o.spot - 1, "sigma_be": sig_be, "iv": o.iv, "delta": o.delta,
                "prob_bs": prob_bs, "prob_venc": p_venc, "prob_toque": p_toque, "ret_esperado": ret_emp,
                "janelas": n_jan, "mult_2s": mult_2s,
                "sinal": {1: "Call", -1: "Put"}.get(tend, "—"), "forca": an.get("score", np.nan),
                "alinhado": tend == direcao,
                "razao_prob": p_venc / prob_bs if (prob_bs and prob_bs > 0 and pd.notna(p_venc)) else np.nan,
            })
    if not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas)
    df.loc[df["janelas"] < MIN_JANELAS, ["prob_venc", "prob_toque", "ret_esperado", "razao_prob"]] = np.nan
    return df.sort_values("ret_esperado", ascending=False, na_position="last").reset_index(drop=True)


def _n(x: float, casas: int = 1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def leitura(o: pd.Series) -> str:
    """Resumo em texto de um pozinho."""
    lado = "alta" if o["tipo"] == "Call" else "queda"
    partes = [f"{o['tipo']} strike {_n(o['strike'], 2)}, {o['du']} d.u., prêmio R$ {_n(o['premio'], 2)}: "
              f"paga a partir de {_n(o['breakeven'], 2)}, {lado} de {_n(abs(o['dist_be']) * 100)}% "
              "até o vencimento"]
    if pd.notna(o["sigma_be"]):
        partes.append(f"isso é {_n(o['sigma_be'])}σ pela vol EWMA do ativo"
                      + (" (aposta de cauda)" if o["sigma_be"] > 2 else ""))
    if pd.notna(o["prob_venc"]):
        bs_txt = "" if pd.isna(o["prob_bs"]) else f"; o preço embute {_n(o['prob_bs'] * 100)}%"
        partes.append(f"no histórico, terminou além do breakeven em {_n(o['prob_venc'] * 100)}% das janelas "
                      f"e tocou nele em {_n(o['prob_toque'] * 100)}%{bs_txt}")
        partes.append(f"retorno esperado empírico de {o['ret_esperado'] * 100:+.0f}% sobre o prêmio "
                      f"({o['janelas']} janelas)")
    else:
        partes.append("histórico curto demais para as probabilidades empíricas")
    partes.append("alinhado com o HiLo" if o["alinhado"] else f"contra o HiLo (sinal {o['sinal']})")
    if pd.isna(o["bid"]):
        partes.append("sem oferta de compra no fechamento: pode ser difícil sair antes do vencimento")
    return ". ".join(p[0].upper() + p[1:] for p in partes) + "."
