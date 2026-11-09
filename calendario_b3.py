"""Calendário de pregões da B3: feriados nacionais e contagem de dias úteis.

Os feriados móveis (Carnaval, Sexta-feira Santa, Corpus Christi) saem da data da
Páscoa, então o calendário funciona para qualquer ano sem precisar de lista manual.
A B3 também não abre em 24/12 e 31/12.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd


def pascoa(ano: int) -> date:
    """Domingo de Páscoa (algoritmo de Meeus/Jones/Butcher)."""
    a, b, c = ano % 19, ano // 100, ano % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(ano, mes, dia)


@lru_cache(maxsize=None)
def feriados(ano: int) -> tuple[date, ...]:
    p = pascoa(ano)
    fixos = [(1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 24), (12, 25), (12, 31)]
    if ano >= 2024:
        fixos.append((11, 20))  # Dia da Consciência Negra, feriado nacional desde 2024
    moveis = [p - timedelta(days=48), p - timedelta(days=47),  # Carnaval (segunda e terça)
              p - timedelta(days=2),                             # Sexta-feira Santa
              p + timedelta(days=60)]                            # Corpus Christi
    return tuple(sorted({date(ano, m, d) for m, d in fixos} | set(moveis)))


def _feriados_entre(inicio: date, fim: date) -> list[str]:
    anos = range(min(inicio.year, fim.year), max(inicio.year, fim.year) + 1)
    return [d.isoformat() for a in anos for d in feriados(a)]


def _data(x) -> date:
    return pd.Timestamp(x).date()


def e_pregao(d) -> bool:
    d = _data(d)
    return bool(np.is_busday(d.isoformat(), holidays=_feriados_entre(d, d)))


def dias_uteis(inicio, fim) -> int:
    """Pregões em (inicio, fim]: conta o dia do vencimento, não conta o dia de hoje."""
    a, b = _data(inicio), _data(fim)
    if b <= a:
        return 0
    hol = _feriados_entre(a, b)
    return int(np.busday_count((a + timedelta(days=1)).isoformat(), (b + timedelta(days=1)).isoformat(),
                               holidays=hol))


def pregoes_anteriores(ref, n: int) -> list[date]:
    """Os `n` pregões mais recentes até `ref` (inclusive), do mais novo para o mais velho."""
    d, saida = _data(ref), []
    while len(saida) < n:
        if e_pregao(d):
            saida.append(d)
        d -= timedelta(days=1)
    return saida
