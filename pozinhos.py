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

Regra de saída (base dos "hot picks"): segurar pozinho até o vencimento quase sempre
perde no histórico; o que você faz é zerar quando a opção sobe 60-80%. Para cada janela
histórica de mesmo prazo, o preço do ativo percorre o caminho real daquele período e a
opção é reavaliada todo dia por Black-Scholes (vol = média entre a implícita da opção e a
EWMA do ativo). Se em algum fechamento ela valer prêmio × (1 + alvo), conta como acerto e
a saída é no alvo; senão, fica com o valor no vencimento. Daí saem:
- Chance de bater o alvo (hist.).
- EV com saída: retorno médio por real investido seguindo a regra. É o critério dos hot picks.

Versão líquida (base da aba "alta prob. lucro"): a mesma simulação, mas a venda antes do
vencimento é feita no bid, não no valor teórico. O bid estimado é valor × (1 − spread/2),
com o spread relativo de hoje ((ask − bid) / meio). Assim, para embolsar +30% sobre o ask
pago, a opção precisa valer bem mais que +30% no modelo quando o spread é largo. Em alvos
baixos (+30%, +50%) é isso que separa chance real de chance de planilha. No vencimento o
valor é o intrínseco (exercício), sem desconto.
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
ALVOS_SAIDA = (0.3, 0.5, 0.6, 0.8, 1.0, 2.0)   # ganho sobre o prêmio em que a operação é zerada
ALVO_PADRAO = 0.8
VERSAO_CALCULO = "2026-11-12"          # mude ao alterar o cálculo: invalida o cache do app
TETO_IV = 3.0                        # implícita acima de 3× a EWMA é tratada como 3× (ruído de pozinho)
SPREAD_TETO = 1.9                    # spread relativo máximo usado no desconto do bid (2 = bid zero)


def col_prob(alvo: float) -> str:
    return f"prob_alvo_{int(round(alvo * 100))}"


def col_ev(alvo: float) -> str:
    return f"ev_saida_{int(round(alvo * 100))}"


def col_prob_liq(alvo: float) -> str:
    return f"prob_liq_{int(round(alvo * 100))}"


def col_ev_liq(alvo: float) -> str:
    return f"ev_liq_{int(round(alvo * 100))}"


def colunas_saida() -> list[str]:
    return [c for a in ALVOS_SAIDA for c in (col_prob(a), col_ev(a), col_prob_liq(a), col_ev_liq(a))]


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


def caminhos(hist: pd.DataFrame, du: int) -> np.ndarray | None:
    """Matriz (janelas × du) com o preço relativo de cada fechamento nos `du` pregões seguintes
    a cada dia do histórico: C[t+j] / C[t]."""
    if du < 1 or len(hist) < du + MIN_JANELAS:
        return None
    c = hist["Close"].to_numpy()
    n = len(c) - du
    idx = np.arange(n)[:, None] + np.arange(1, du + 1)[None, :]
    return c[idx] / c[:n, None]


def metricas_saida(rel: np.ndarray | None, spot: float, strike: float, premio: float, tipo: str,
                   vol: float | None, r: float, du: int, spread: float | None = None) -> dict:
    """Chance de bater cada alvo e EV seguindo a regra de saída, no caminho histórico.

    Bruta: saída no valor teórico. Líquida: saída antes do vencimento no bid estimado,
    valor × (1 − spread/2); no último pregão vale o intrínseco."""
    vazio = {k: np.nan for k in colunas_saida()}
    if rel is None or not vol or premio <= 0:
        return vazio
    S = spot * rel
    T = (du - np.arange(1, du + 1)) / bs.DIAS_ANO       # prazo restante em cada fechamento
    valor = bs.preco(S, strike, np.broadcast_to(T, S.shape), r, vol, tipo)
    pico = valor.max(axis=1)
    final = valor[:, -1] / premio - 1

    s = spread if (spread is not None and pd.notna(spread) and spread >= 0) else None
    if s is None:
        pico_liq = None
    else:
        desconto = 1 - min(s, SPREAD_TETO) / 2
        antes = valor[:, :-1] * desconto if du > 1 else np.zeros((len(valor), 0))
        pico_liq = np.maximum(antes.max(axis=1) if antes.shape[1] else 0.0, valor[:, -1])

    out = {}
    for a in ALVOS_SAIDA:
        bateu = pico >= premio * (1 + a)
        out[col_prob(a)] = float(bateu.mean())
        out[col_ev(a)] = float(np.where(bateu, a, final).mean())
        if pico_liq is None:
            out[col_prob_liq(a)] = np.nan
            out[col_ev_liq(a)] = np.nan
        else:
            bateu_liq = pico_liq >= premio * (1 + a)
            out[col_prob_liq(a)] = float(bateu_liq.mean())
            out[col_ev_liq(a)] = float(np.where(bateu_liq, a, final).mean())
    return out


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
        cache_cam: dict[int, np.ndarray | None] = {}
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
            if o.du not in cache_cam:
                cache_cam[o.du] = caminhos(hist, o.du)
            vol_cam = vol
            if vol and pd.notna(o.iv):
                vol_cam = 0.5 * (min(o.iv, TETO_IV * vol) + vol)
            spread_o = o.spread if pd.notna(o.bid) else None   # sem bid não há como estimar a venda
            saida = metricas_saida(cache_cam[o.du], o.spot, o.strike, o.premio, o.tipo, vol_cam, r, o.du,
                                   spread_o)
            linhas.append({
                "opcao": o.opcao, "ativo": ativo, "tipo": o.tipo, "vencimento": o.vencimento, "du": o.du,
                "strike": o.strike, "spot": o.spot, "dist": o.dist, "premio": o.premio, "bid": o.bid,
                "ask": o.ask, "ultimo": o.ultimo, "negocios": o.negocios, "volume": o.volume,
                "spread": o.spread, "semanal": bool(o.semanal),
                "breakeven": be, "dist_be": be / o.spot - 1, "sigma_be": sig_be, "iv": o.iv, "delta": o.delta,
                "prob_bs": prob_bs, "prob_venc": p_venc, "prob_toque": p_toque, "ret_esperado": ret_emp,
                "janelas": n_jan, "mult_2s": mult_2s,
                "sinal": {1: "Call", -1: "Put"}.get(tend, "—"), "forca": an.get("score", np.nan),
                "alinhado": tend == direcao,
                "razao_prob": p_venc / prob_bs if (prob_bs and prob_bs > 0 and pd.notna(p_venc)) else np.nan,
                **saida,
            })
    if not linhas:
        return pd.DataFrame()
    df = pd.DataFrame(linhas)
    hist_cols = ["prob_venc", "prob_toque", "ret_esperado", "razao_prob"] + colunas_saida()
    df.loc[df["janelas"] < MIN_JANELAS, hist_cols] = np.nan
    return df.sort_values(col_ev(ALVO_PADRAO), ascending=False, na_position="last").reset_index(drop=True)


# ---- Hot picks ----------------------------------------------------------------------
SPREAD_MAX = 1.0      # (ask − bid) / meio: acima disso a saída antecipada custa caro demais
PROB_MIN = 0.10       # chance mínima de bater o alvo para não ser só bilhete de loteria


def classificar(ev: float, prob: float) -> str:
    if pd.isna(ev):
        return "—"
    if ev >= 0.15 and prob >= 0.20:
        return "Quente"
    if ev > 0:
        return "Morno"
    return "Frio"


def hot_picks(poz: pd.DataFrame, alvo: float, tipos=("Call", "Put"), du_max: int = 30, sem_semanais: bool = True,
              so_alinhados: bool = True, mercado_fn=None, so_mercado: bool = False) -> pd.DataFrame:
    """Pozinhos negociáveis (bid, negócio no pregão, spread aceitável), ordenados pelo EV com saída."""
    if poz.empty:
        return poz
    ev, pr = col_ev(alvo), col_prob(alvo)
    f = (poz["tipo"].isin(tipos) & (poz["du"] <= du_max) & poz["bid"].notna() & (poz["negocios"] > 0)
         & (poz["spread"] <= SPREAD_MAX) & (poz[pr] >= PROB_MIN) & poz[ev].notna())
    if sem_semanais:
        f &= ~poz["semanal"]
    if so_alinhados:
        f &= poz["alinhado"]
    out = poz[f].copy()
    if mercado_fn is not None:
        out["mercado"] = out["tipo"].map(mercado_fn)
        if so_mercado:
            out = out[out["mercado"] != "Contra"]
    out["ev"], out["prob_alvo"] = out[ev], out[pr]
    out["nota"] = [classificar(e, p) for e, p in zip(out["ev"], out["prob_alvo"])]
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


# ---- Alta probabilidade de lucro ----------------------------------------------------------
ALVO_ALTA_PROB = 0.3        # alvo padrão da aba: lucro menor, mais provável
PROB_MIN_ALTA = 0.5         # chance líquida mínima padrão
SPREAD_MAX_ALTA = 0.5       # spread máximo padrão: em alvo baixo, spread largo come o lucro


def classificar_prob(prob: float, ev: float) -> str:
    if pd.isna(prob):
        return "—"
    if prob >= 0.65 and ev > 0:
        return "Muito alta"
    if prob >= 0.5 and ev > 0:
        return "Alta"
    if ev > 0:
        return "Moderada"
    return "Sem vantagem"


def alta_prob(poz: pd.DataFrame, alvo: float = ALVO_ALTA_PROB, prob_min: float = PROB_MIN_ALTA,
              spread_max: float = SPREAD_MAX_ALTA, ev_positivo: bool = True, tipos=("Call", "Put"),
              du_max: int = 30, sem_semanais: bool = True, so_alinhados: bool = True, mercado_fn=None,
              so_mercado: bool = False) -> pd.DataFrame:
    """Pozinhos com maior chance líquida (venda no bid) de bater um alvo de lucro modesto.

    Ordena pela chance líquida e desempata pelo EV líquido. Com `ev_positivo`, descarta os
    que têm chance alta mas EV negativo (ganha pouco muitas vezes e perde tudo nas outras,
    acima do que compensa)."""
    pr, ev = col_prob_liq(alvo), col_ev_liq(alvo)
    if poz.empty or pr not in poz.columns or ev not in poz.columns:
        return poz.iloc[0:0]
    f = (poz["tipo"].isin(tipos) & (poz["du"] <= du_max) & poz["bid"].notna() & (poz["negocios"] > 0)
         & (poz["spread"] <= spread_max) & poz[pr].notna() & (poz[pr] >= prob_min))
    if ev_positivo:
        f &= poz[ev] > 0
    if sem_semanais:
        f &= ~poz["semanal"]
    if so_alinhados:
        f &= poz["alinhado"]
    out = poz[f].copy()
    if mercado_fn is not None:
        out["mercado"] = out["tipo"].map(mercado_fn)
        if so_mercado:
            out = out[out["mercado"] != "Contra"]
    out["prob_alvo"], out["ev"] = out[pr], out[ev]
    out["prob_bruta"] = out[col_prob(alvo)]
    out["nota"] = [classificar_prob(p, e) for p, e in zip(out["prob_alvo"], out["ev"])]
    return out.sort_values(["prob_alvo", "ev"], ascending=[False, False]).reset_index(drop=True)


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
    ev80, p80 = o.get(col_ev(ALVO_PADRAO)), o.get(col_prob(ALVO_PADRAO))
    if ev80 is not None and pd.notna(ev80):
        partes.append(f"com a regra de zerar em +{ALVO_PADRAO * 100:.0f}%, bateu o alvo em {_n(p80 * 100)}% das "
                      f"janelas e o EV ficou em {ev80 * 100:+.0f}% por real investido")
    p30 = o.get(col_prob_liq(ALVO_ALTA_PROB))
    if p30 is not None and pd.notna(p30):
        partes.append(f"vendendo no bid, chegou a +{ALVO_ALTA_PROB * 100:.0f}% líquido em {_n(p30 * 100)}% "
                      "das janelas")
    if pd.isna(o["bid"]):
        partes.append("sem oferta de compra no fechamento: pode ser difícil sair antes do vencimento")
    return ". ".join(p[0].upper() + p[1:] for p in partes) + "."
