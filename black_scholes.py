"""Black-Scholes para opções sobre ações da B3 (vetorizado com numpy).

Convenções:
- Tempo em dias úteis / 252 (padrão do mercado brasileiro).
- Taxa contínua r = ln(1 + Selic anual). Sem dividendos (q = 0): perto de uma data ex
  o modelo fica menos preciso, especialmente em calls.
- Opções de ação na B3 são americanas; para calls sem provento no prazo o preço é
  igual ao europeu, e para puts a diferença é pequena nos prazos curtos que você opera.
"""
from __future__ import annotations

import json
import math
import urllib.request

import numpy as np

DIAS_ANO = 252
SELIC_PADRAO = 0.15  # usada só se a API do Banco Central não responder
_URL_SELIC = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"

def _erf(x: np.ndarray) -> np.ndarray:
    """Função erro, aproximação de Abramowitz-Stegun 7.1.26 (erro máximo ~1,5e-7), em numpy puro."""
    sinal = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-x * x)
    return sinal * y


def N(x):
    """Distribuição normal acumulada."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def n(x):
    """Densidade da normal."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def selic_meta() -> float:
    """Meta Selic anual (ex.: 0.15) pela série 432 do SGS/Banco Central."""
    try:
        with urllib.request.urlopen(_URL_SELIC, timeout=10) as r:
            dados = json.loads(r.read().decode("utf-8"))
        return float(str(dados[-1]["valor"]).replace(",", ".")) / 100
    except Exception:
        return SELIC_PADRAO


def taxa_continua(selic_anual: float) -> float:
    return math.log1p(selic_anual)


def _arr(*xs):
    return [np.asarray(x, dtype=float) for x in xs]


def _sinal(tipo) -> np.ndarray:
    """+1 para call, -1 para put. Aceita 'Call'/'Put'/'C'/'P' (escalar ou array)."""
    t = np.asarray(tipo).astype(str)
    return np.where(np.char.upper(np.char.strip(t)).astype(str).astype("U1") == "C", 1.0, -1.0)


def _d1d2(S, K, T, r, vol):
    with np.errstate(divide="ignore", invalid="ignore"):
        raiz = vol * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * vol * vol) * T) / raiz
    return d1, d1 - raiz


def preco(S, K, T, r, vol, tipo):
    """Prêmio teórico. T em anos (dias úteis / 252)."""
    S, K, T, vol = _arr(S, K, T, vol)
    s = _sinal(tipo)
    d1, d2 = _d1d2(S, K, T, r, vol)
    teorico = s * (S * N(s * d1) - K * np.exp(-r * T) * N(s * d2))
    intrinseco = np.maximum(s * (S - K), 0.0)
    return np.where((T <= 0) | (vol <= 0), intrinseco, teorico)


def gregas(S, K, T, r, vol, tipo) -> dict:
    """Delta, gama, theta (por dia útil), vega (por 1 ponto de vol) e prob. de exercício."""
    S, K, T, vol = _arr(S, K, T, vol)
    s = _sinal(tipo)
    d1, d2 = _d1d2(S, K, T, r, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(s > 0, N(d1), N(d1) - 1)
        gama = n(d1) / (S * vol * np.sqrt(T))
        theta_ano = (-S * n(d1) * vol / (2 * np.sqrt(T)) - s * r * K * np.exp(-r * T) * N(s * d2))
        vega = S * n(d1) * np.sqrt(T) / 100
    return {
        "delta": delta,
        "gama": gama,
        "theta": theta_ano / DIAS_ANO,
        "vega": vega,
        "prob_exercicio": N(s * d2),  # probabilidade (neutra a risco) de vencer dentro do dinheiro
    }


def vol_implicita(premio, S, K, T, r, tipo, vmin: float = 0.01, vmax: float = 5.0, iteracoes: int = 80):
    """Vol implícita por bisseção (vetorizada). NaN se o prêmio estiver fora do intervalo possível."""
    premio, S, K, T = _arr(premio, S, K, T)
    premio, S, K, T = np.broadcast_arrays(premio, S, K, T)
    tipo = np.broadcast_to(np.asarray(tipo), premio.shape)
    lo = np.full(premio.shape, vmin)
    hi = np.full(premio.shape, vmax)
    p_lo, p_hi = preco(S, K, T, r, lo, tipo), preco(S, K, T, r, hi, tipo)
    valido = (premio > 0) & (T > 0) & (premio >= p_lo - 1e-9) & (premio <= p_hi)
    for _ in range(iteracoes):
        meio = 0.5 * (lo + hi)
        acima = preco(S, K, T, r, meio, tipo) > premio
        hi = np.where(acima, meio, hi)
        lo = np.where(acima, lo, meio)
    return np.where(valido, 0.5 * (lo + hi), np.nan)
