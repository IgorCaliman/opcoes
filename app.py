import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from datetime import date

import alvos
import black_scholes as bs
import cotacoes
import estrategias
import indicadores
import mercado
import opcoes_dados
import operacoes as ops_mod
import pozinhos as pz
from indicadores import HILO_PERIODOS, IFR_PERIODOS, calcular, distancias_mm, sinal

# Lista principal de ativos (sem o ".SA"). Ativos de operações abertas que não
# estejam aqui são adicionados automaticamente a ela.
ATIVOS_BASE = [
    "BEEF3", "BRAP4", "COGN3", "CSAN3", "CSNA3", "GGBR4", "MBRF3",
    "MGLU3", "MRVE3", "PETR4", "PRIO3", "RENT3", "USIM5", "WEGE3",
]

# Ativos só para acompanhamento: ficam na aba "Acompanhamento" e NÃO entram
# na lista principal nem na aba Consolidado.
WATCHLIST = [
    # Primeira lista
    "BOVA11", "PETR4", "SMAL11", "BBAS3", "VALE3", "BBDC4", "ITUB4", "CYRE3",
    "MGLU3", "B3SA3", "PRIO3", "ABEV3", "BPAC11", "CMIN3", "WEGE3", "CSNA3",
    "SUZB3", "PETR3", "EMBJ3", "ITSA4", "CEAB3", "BEEF3", "LREN3",
    "VAMO3",
    # Segunda lista
    "BBSE3", "TOTS3", "USIM5", "RENT3", "AXIA3", "COGN3", "MBRF3", "MRVE3",
    "GGBR4", "AZZA3", "SBSP3", "CSAN3", "BBDC3", "CMIG4", "NATU3", "BRAP4",
    "POMO4", "DIRR3", "EQTL3", "ALOS3", "KLBN11", "RDOR3", "JHSF3", "ASAI3",
    "HAPV3",
    # Vindo dos prints de posições em opções
    "MULT3",
]
WATCHLIST = list(dict.fromkeys(WATCHLIST))  # remove repetidos, mantém a ordem

# Cores escolhidas para funcionar nos temas claro e escuro.
COR_HILO_ALTA = "#00c853"
COR_HILO_BAIXA = "#ff1744"
COR_MM9 = "#a855f7"
COR_MM20 = "#facc15"
COR_MM200 = "#f08080"
COR_IFR = "#00bcd4"

st.set_page_config(page_title="Acompanhamento de Mercado", layout="wide")
st.title("Acompanhamento de Mercado")

OPERACOES = ops_mod.carregar_operacoes()
ATIVOS = sorted(set(ATIVOS_BASE) | set(OPERACOES["ativo"]))
# Universo das abas de ideias: lista principal + acompanhados
UNIVERSO = sorted(set(ATIVOS) | set(WATCHLIST))


@st.cache_data(ttl=900, show_spinner="Atualizando cotações...")
def sincronizar(tickers: tuple) -> tuple[str, list[str]]:
    """Atualiza o cache em disco de todos os ativos de uma vez (download em lote).

    Devolve (versão, tickers sem dados). A versão entra na chave de cache das funções
    abaixo: quando as cotações são atualizadas, os indicadores são recalculados juntos.
    """
    sem_dados = cotacoes.atualizar(list(tickers))
    return pd.Timestamp.now().isoformat(), sem_dados


VERSAO, SEM_DADOS = sincronizar(tuple(UNIVERSO))


@st.cache_data(ttl=900, show_spinner=False)
def _carregar(ticker: str, versao: str) -> pd.DataFrame:
    """Histórico ajustado por proventos (lido do disco) + indicadores."""
    return calcular(cotacoes.carregar_ajustado(ticker))


def carregar(ticker: str) -> pd.DataFrame:
    return _carregar(ticker, VERSAO)


@st.cache_data(ttl=900, show_spinner="Analisando ativos...")
def _analisar_universo(tickers: tuple, versao: str) -> list[dict]:
    """Resumo de indicadores de cada ativo (usa os dados já em cache)."""
    saida = []
    for t in tickers:
        try:
            saida.append({"ativo": t, **estrategias.analisar(_carregar(t, versao))})
        except Exception:
            pass
    return saida


def analisar_universo(tickers: tuple) -> list[dict]:
    return _analisar_universo(tickers, VERSAO)


@st.cache_data(ttl=900, show_spinner=False)
def _alvos_ativo(ticker: str, direcao: int, versao: str) -> dict:
    try:
        return alvos.calcular(_carregar(ticker, versao), direcao)
    except Exception:
        return {"ok": False}


def alvos_ativo(ticker: str, direcao: int) -> dict:
    """Alvo (Bollinger), stop curto (ATR) e histórico condicional do ativo na direção da ideia."""
    return _alvos_ativo(ticker, direcao, VERSAO)


def _historicos(versao: str) -> dict[str, pd.DataFrame]:
    saida = {}
    for t in UNIVERSO:
        try:
            saida[t] = _carregar(t, versao)
        except Exception:
            pass
    return saida


@st.cache_data(ttl=900, show_spinner="Lendo o regime de mercado...")
def _contexto_mercado(versao: str) -> tuple[dict, pd.DataFrame]:
    hist = _historicos(versao)
    amp = mercado.amplitude(hist)
    return mercado.regime(hist.get(mercado.INDICE), amp), amp


REGIME, AMPLITUDE = _contexto_mercado(VERSAO)


@st.cache_data(ttl=3600, show_spinner="Baixando opções da B3 (COTAHIST)...")
def pregao_opcoes(hoje: date):
    """(data do pregão, DataFrame do COTAHIST) mais recente, ou None."""
    return opcoes_dados.ultimo_pregao(hoje)


@st.cache_data(ttl=86400, show_spinner=False)
def selic_atual() -> float:
    return bs.selic_meta()


def grafico(df: pd.DataFrame, ticker: str, candles: int) -> go.Figure:
    d = df.tail(candles)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker,
    ), row=1, col=1)

    # HiLo: verde na tendência de alta (Call), vermelho na de baixa (Put)
    for tendencia, cor, nome in (
        (1, COR_HILO_ALTA, f"HiLo {HILO_PERIODOS} (alta)"),
        (-1, COR_HILO_BAIXA, f"HiLo {HILO_PERIODOS} (baixa)"),
    ):
        fig.add_trace(go.Scatter(
            x=d.index, y=d["HiLo"].where(d["HiLo_tendencia"] == tendencia),
            name=nome, mode="lines",
            line=dict(color=cor, width=2, shape="hv"),
        ), row=1, col=1)

    for n, cor in ((9, COR_MM9), (20, COR_MM20), (200, COR_MM200)):
        fig.add_trace(go.Scatter(
            x=d.index, y=d[f"MM{n}"], name=f"MM{n}", mode="lines",
            line=dict(color=cor, width=1.5),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=d.index, y=d["IFR"], name=f"IFR {IFR_PERIODOS}",
        line=dict(color=COR_IFR, width=1.5),
    ), row=2, col=1)
    for nivel in (30, 70):
        fig.add_hline(y=nivel, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
    )
    return fig


def brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct_sinal(v: float) -> str:
    return f"{v * 100:+.1f}%".replace(".", ",")


def cor_dist(v: float) -> str:
    """Distância às médias: verde acima, vermelho abaixo (satura em 15%)."""
    if pd.isna(v):
        return ""
    intensidade = min(abs(v) / 0.15, 1.0) * 0.6
    rgb = "46, 160, 90" if v >= 0 else "220, 70, 60"
    return f"background-color: rgba({rgb}, {intensidade:.2f})"


def ou_traco(fmt):
    """Aplica `fmt` ao valor, ou mostra '—' se for nulo."""
    return lambda v: "—" if pd.isna(v) else fmt(v)


def cor_estrutura(v: str) -> str:
    if "Put" in v:
        return "background-color: rgba(220, 70, 60, 0.25)"
    return "background-color: rgba(46, 160, 90, 0.25)"


def cor_sinal(v: str) -> str:
    return {"Call": "background-color:#c6efce;color:#006100",
            "Put": "background-color:#ffc7ce;color:#9c0006"}.get(v, "")


AVISO = (
    "Ideias geradas por regras técnicas (HiLo, IFR e médias móveis), não são recomendação de investimento. "
    "Não consideram preço, liquidez, volatilidade implícita nem vencimento das opções: "
    "valide na cadeia de opções antes de operar."
)


def posicao_aberta(ativo: str) -> str:
    o = OPERACOES[OPERACOES["ativo"] == ativo]
    return ", ".join(sorted(set(o["tipo"]))) if not o.empty else "—"


@st.dialog("Adicionar operação")
def dialogo_adicionar():
    ativo = st.text_input("Ativo", placeholder="Ex.: PETR4")
    tipo = st.radio("Tipo", ops_mod.TIPOS, horizontal=True)
    if st.button("Salvar", type="primary"):
        try:
            ops_mod.adicionar_operacao(ativo, tipo)
        except ValueError as e:
            st.error(str(e))
        else:
            st.rerun()


@st.dialog("Fechar operação")
def dialogo_fechar():
    if OPERACOES.empty:
        st.caption("Nenhuma operação aberta.")
        return
    opcoes = [f"{r.ativo} — {r.tipo}" for r in OPERACOES.itertuples()]
    escolhidas = st.multiselect("Operações encerradas", opcoes, placeholder="Escolha uma ou mais")
    if st.button("Excluir", type="primary", disabled=not escolhidas):
        itens = [tuple(o.split(" — ")) for o in escolhidas]
        try:
            ops_mod.remover_operacoes(itens)
        except ValueError as e:
            st.error(str(e))
        else:
            st.rerun()


def estilo_ideias(tab: pd.DataFrame):
    mapa = {
        "Último": brl, "HiLo (stop ref.)": brl,
        "Força": lambda v: f"{int(v):+d}",
        f"IFR {IFR_PERIODOS}": lambda v: f"{v:.1f}",
        "Dist. MM20": pct_sinal, "Dist. MM200": pct_sinal,
    }
    est = tab.style.format({c: ou_traco(f) for c, f in mapa.items() if c in tab.columns})
    dist = [c for c in ("Dist. MM20", "Dist. MM200") if c in tab.columns]
    if dist:
        est = est.map(cor_dist, subset=dist)
    if "Mercado" in tab.columns:
        est = est.map(cor_mercado, subset=["Mercado"])
    return est


def cor_mercado(v: str) -> str:
    return {"A favor": "background-color: rgba(46, 160, 90, 0.35)",
            "Contra": "background-color: rgba(220, 70, 60, 0.35)"}.get(v, "")


def explicacoes(itens: list[tuple[str, str]]):
    """Lista de 'ativo — porquê'. O $ é escapado para o markdown não virar fórmula."""
    for ativo, texto in itens:
        st.markdown(f"- **{ativo}**: " + texto.replace("$", "\\$"))


# ---- Alvo e gestão de saída (aba direcional) ---------------------------------
VERDE = "background-color: rgba(46, 160, 90, 0.45)"
AMARELO = "background-color: rgba(210, 153, 34, 0.45)"
VERMELHO = "background-color: rgba(220, 70, 60, 0.45)"
VERMELHO_FORTE = "background-color: rgba(220, 70, 60, 0.70)"
COR_LEITURA = {
    "Com espaço": VERDE, "Esticado": AMARELO, "Acima do alvo": AMARELO, "Abaixo do alvo": AMARELO,
    "Sinal de saída": VERMELHO, "Stop curto rompido": VERMELHO_FORTE,
}
COL_MED = f"Mediana {alvos.HORIZONTE_FWD}d (hist.)"
COL_ACERTO = f"% positivo {alvos.HORIZONTE_FWD}d"


def cor_rr(v: float) -> str:
    if pd.isna(v):
        return ""
    return VERDE if v >= 2 else AMARELO if v >= alvos.RR_MINIMA else VERMELHO


def tabela_alvos(itens: list, direcao: int):
    """Tabela de alvo/stop logo abaixo dos candidatos, com a leitura por ativo num expander."""
    linhas, textos = [], []
    for an, _, _ in itens:
        r = alvos_ativo(an["ativo"], direcao)
        textos.append((an["ativo"], alvos.texto(r)))
        ok = r.get("ok", False)
        v = (lambda k, sinal_=1: sinal_ * r[k] if ok else np.nan)
        linhas.append({
            "Ativo": an["ativo"], "Leitura": alvos.status(r),
            "Alvo (Bollinger)": v("alvo"), "Espaço até alvo": v("espaco"),
            "Stop curto (ATR)": v("stop"), "Risco até stop": v("risco", -1),
            "Alvo/stop": v("rr"), "%B": v("pctb"), "Percentil esticamento": v("percentil"),
            COL_MED: v("med_fwd"), COL_ACERTO: v("acerto_fwd"),
            "Posição aberta": posicao_aberta(an["ativo"]),
        })
    tab = pd.DataFrame(linhas)
    estilo = (
        tab.style
        .map(lambda s: COR_LEITURA.get(s, ""), subset=["Leitura"])
        .map(cor_rr, subset=["Alvo/stop"])
        .format({
            "Alvo (Bollinger)": ou_traco(brl), "Stop curto (ATR)": ou_traco(brl),
            "Espaço até alvo": ou_traco(pct_sinal), "Risco até stop": ou_traco(pct_sinal),
            COL_MED: ou_traco(pct_sinal),
            "Alvo/stop": ou_traco(lambda x: f"{x:.1f}x".replace(".", ",")),
            "%B": ou_traco(lambda x: f"{x * 100:.0f}%"),
            "Percentil esticamento": ou_traco(lambda x: f"{x:.0f}"),
            COL_ACERTO: ou_traco(lambda x: f"{x:.0f}%"),
        })
    )
    st.markdown("**Alvo e gestão de saída**")
    st.dataframe(estilo, hide_index=True, width="stretch", height=40 + 35 * len(tab))
    with st.expander("Leitura de alvo e saída por ativo"):
        explicacoes(textos)


# Barra lateral: controles de dados (valem para todas as abas)
with st.sidebar:
    st.header("Dados")
    if st.button("Atualizar cotações", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    if st.button("Refazer histórico completo", width="stretch"):
        cotacoes.limpar_cache()
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "Atualizar baixa só os últimos dias, em lote. Desdobramentos e grupamentos são "
        "detectados sozinhos; Refazer apaga a pasta dados/ e baixa tudo de novo."
    )
    st.caption("Preços e indicadores ajustados por proventos (dividendos e JCP).")
    if SEM_DADOS:
        st.warning("Sem dados: " + ", ".join(SEM_DADOS))

(aba_ativo, aba_consolidado, aba_operacoes, aba_mercado, aba_watch,
 aba_direcional, aba_travas, aba_cadeia, aba_pozinhos, aba_hot) = st.tabs([
    "Ativo por ativo", "Consolidado", "Operações abertas", "Mercado e exposição", "Acompanhamento",
    "Acompanhamento - direcional", "Acompanhamento - travas", "Cadeia de opções", "Pozinhos",
    "Pozinhos - hot picks",
])

with aba_ativo:
    col1, col2 = st.columns([1, 1])
    ticker = col1.selectbox("Ativo", ATIVOS, index=ATIVOS.index("PETR4") if "PETR4" in ATIVOS else 0)
    candles = col2.slider("Candles no gráfico", 60, 500, 180, step=20)

    try:
        with st.spinner("Carregando cotações..."):
            df = carregar(ticker)
    except Exception as e:
        st.error(f"Não foi possível carregar {ticker}: {e}")
        st.stop()

    ult = df.iloc[-1]
    s = sinal(df)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Último", f"R$ {ult['Close_bruto']:.2f}")
    m2.metric("Sinal HiLo", s)
    m3.metric(f"HiLo {HILO_PERIODOS}", f"{ult['HiLo']:.2f}" if pd.notna(ult["HiLo"]) else "—")
    m4.metric(f"IFR {IFR_PERIODOS}", f"{ult['IFR']:.1f}")
    m5.metric("MM9 / MM20", f"{ult['MM9']:.2f} / {ult['MM20']:.2f}")
    m6.metric("MM200", f"{ult['MM200']:.2f}" if pd.notna(ult["MM200"]) else "—")

    st.plotly_chart(grafico(df, ticker, candles), width="stretch")
    vol_txt = f" Vol. EWMA anualizada: {ult['Vol_EWMA'] * 100:.0f}%." if pd.notna(ult["Vol_EWMA"]) else ""
    st.caption(f"Último candle: {df.index[-1]:%d/%m/%Y}. Gráfico e indicadores ajustados por proventos.{vol_txt}")

with aba_consolidado:
    linhas = []
    barra = st.progress(0.0, text="Carregando cotações...")
    for i, t in enumerate(ATIVOS, start=1):
        try:
            linhas.append({"Ativo": t, "Sinal": sinal(carregar(t))})
        except Exception:
            linhas.append({"Ativo": t, "Sinal": "Sem dados"})
        barra.progress(i / len(ATIVOS), text=f"Carregando cotações... {t}")
    barra.empty()
    tabela = pd.DataFrame(linhas)

    col_call, col_put = st.columns(2)
    for coluna, nome in ((col_call, "Call"), (col_put, "Put")):
        sub = tabela[tabela["Sinal"] == nome][["Ativo"]]
        with coluna:
            st.subheader(f"{nome} ({len(sub)})")
            if sub.empty:
                st.caption("Nenhum ativo.")
            else:
                st.dataframe(
                    sub.style.map(lambda _, n=nome: cor_sinal(n), subset=["Ativo"]),
                    hide_index=True, width="stretch",
                    height=40 + 35 * len(sub),
                )
    sem_sinal = tabela[~tabela["Sinal"].isin(["Call", "Put"])]["Ativo"]
    if len(sem_sinal):
        st.caption("Sem sinal ou sem dados: " + ", ".join(sem_sinal))

with aba_operacoes:
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("Adicionar operação", type="primary"):
        dialogo_adicionar()
    if c2.button("Fechar operação", disabled=OPERACOES.empty):
        dialogo_fechar()

    if OPERACOES.empty:
        st.caption("Nenhuma operação aberta.")
    else:
        n_call = int((OPERACOES["tipo"] == "Call").sum())
        n_put = int((OPERACOES["tipo"] == "Put").sum())
        st.caption(f"{len(OPERACOES)} operações abertas ({n_call} Call, {n_put} Put)")
        exibir = OPERACOES.rename(columns={"ativo": "Ativo", "tipo": "Tipo"})
        st.dataframe(
            exibir.style.map(cor_estrutura, subset=["Tipo"]),
            hide_index=True, width="stretch",
            height=40 + 35 * len(exibir),
        )

with aba_watch:
    st.caption("Ativos só para acompanhamento — não fazem parte da lista principal.")
    colunas = ["Ativo", "Último", "Sinal HiLo", "HiLo", f"IFR {IFR_PERIODOS}",
               "Dist. MM9", "Dist. MM20", "Dist. MM200", "Média MMs", "Dist. média MMs"]
    linhas = []
    barra = st.progress(0.0, text="Carregando cotações...")
    for i, t in enumerate(WATCHLIST, start=1):
        try:
            d = carregar(t)
            u = d.iloc[-1]
            dm = distancias_mm(u)
            linhas.append({
                "Ativo": t, "Último": u["Close"], "Sinal HiLo": sinal(d),
                "HiLo": u["HiLo"], f"IFR {IFR_PERIODOS}": u["IFR"],
                "Dist. MM9": dm["dist_mm9"], "Dist. MM20": dm["dist_mm20"],
                "Dist. MM200": dm["dist_mm200"], "Média MMs": dm["media_mm"],
                "Dist. média MMs": dm["dist_media"],
            })
        except Exception:
            linhas.append({"Ativo": t, "Sinal HiLo": "Sem dados"})
        barra.progress(i / len(WATCHLIST), text=f"Carregando cotações... {t}")
    barra.empty()
    tab_watch = pd.DataFrame(linhas).reindex(columns=colunas)

    cols_dist = ["Dist. MM9", "Dist. MM20", "Dist. MM200", "Dist. média MMs"]
    st.dataframe(
        tab_watch.style
        .map(cor_sinal, subset=["Sinal HiLo"])
        .map(cor_dist, subset=cols_dist)
        .format({
            "Último": ou_traco(brl),
            "HiLo": ou_traco(brl),
            f"IFR {IFR_PERIODOS}": ou_traco(lambda v: f"{v:.1f}"),
            "Média MMs": ou_traco(brl),
            **{c: ou_traco(pct_sinal) for c in cols_dist},
        }),
        hide_index=True, width="stretch",
        height=40 + 35 * len(tab_watch),
    )
    st.caption(
        "Dist. = distância do último fechamento a cada média (positivo: acima, negativo: abaixo). "
        "Média MMs = média simples de MM9, MM20 e MM200; Dist. média MMs = distância do preço a essa média."
    )

    st.divider()
    w1, w2 = st.columns([1, 1])
    t_watch = w1.selectbox("Ver gráfico do ativo", WATCHLIST, key="watch_ativo")
    c_watch = w2.slider("Candles no gráfico", 60, 500, 180, step=20, key="watch_candles")
    try:
        st.plotly_chart(grafico(carregar(t_watch), t_watch, c_watch), width="stretch")
    except Exception as e:
        st.error(f"Não foi possível carregar {t_watch}: {e}")

with aba_direcional:
    st.info(f"Regime de mercado: **{REGIME['rotulo']}**. {REGIME['texto']}")
    st.caption(AVISO)
    st.caption(
        "Força de −6 a +6: HiLo vale 2 pontos; MM9, MM20, MM200 e o alinhamento entre elas valem 1 cada. "
        f"Ideia direcional: força de {estrategias.SCORE_DIRECIONAL} ou mais (em módulo), com IFR fora de "
        f"sobrecompra ({estrategias.IFR_ALTO}) e sobrevenda ({estrategias.IFR_BAIXO})."
    )
    st.caption(
        f"Alvo e saída: o alvo é a banda de Bollinger ({alvos.BB_PERIODOS}, {alvos.BB_DESVIOS:g}σ) do lado da "
        f"operação; o stop curto é o Chandelier ({alvos.CHANDELIER_JANELA} candles, {alvos.CHANDELIER_MULT:g}×ATR "
        f"{alvos.ATR_PERIODOS}), mais perto que o HiLo. Alvo/stop = espaço até o alvo ÷ risco até o stop. "
        "%B: 0% na banda inferior, 100% na superior. Percentil de esticamento: onde a distância atual à MM20 "
        f"cai no histórico do ativo, no sentido da operação. Mediana e % positivo: retorno a favor "
        f"{alvos.HORIZONTE_FWD} candles depois, em situações parecidas no passado (mesma faixa de %B, preço do "
        "mesmo lado da MM20). Valores do ativo, não da opção: theta e volatilidade implícita ficam de fora."
    )
    analises = analisar_universo(tuple(UNIVERSO))
    grupos = {"call": [], "put": [], "obs": []}
    for an in analises:
        tipo, rotulo, texto = estrategias.direcional(an)
        if tipo:
            grupos[tipo].append((an, rotulo, texto))
    grupos["call"].sort(key=lambda x: -x[0]["score"])
    grupos["put"].sort(key=lambda x: x[0]["score"])
    grupos["obs"].sort(key=lambda x: -abs(x[0]["score"]))

    for tipo, titulo in (("call", "Candidatos a Call"), ("put", "Candidatos a Put")):
        itens = grupos[tipo]
        st.subheader(f"{titulo} ({len(itens)})")
        if not itens:
            st.caption("Nenhum ativo atende às regras hoje.")
            continue
        tab = pd.DataFrame([{
            "Ativo": an["ativo"], "Operação": rot, "Último": an["preco"], "Força": an["score"],
            "HiLo (stop ref.)": an["hilo"], f"IFR {IFR_PERIODOS}": an["ifr"],
            "Dist. MM20": an["d20"], "Dist. MM200": an["d200"], "Dias no sinal": an["dias"],
            "Mercado": mercado.mercado_a_favor(tipo, REGIME),
            "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, rot, _ in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, width="stretch",
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], txt) for an, _, txt in itens])
        tabela_alvos(itens, 1 if tipo == "call" else -1)

    itens = grupos["obs"]
    st.subheader(f"Em observação: tendência forte, mas esticada ({len(itens)})")
    if not itens:
        st.caption("Nenhum ativo nessa situação hoje.")
    else:
        tab = pd.DataFrame([{
            "Ativo": an["ativo"], "Situação": rot, "Último": an["preco"], "Força": an["score"],
            f"IFR {IFR_PERIODOS}": an["ifr"], "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, rot, _ in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, width="stretch",
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], txt) for an, _, txt in itens])

with aba_travas:
    st.info(f"Regime de mercado: **{REGIME['rotulo']}**. {REGIME['texto']}")
    st.caption(AVISO)
    st.caption(
        f"σ = volatilidade diária EWMA (λ = {indicadores.EWMA_LAMBDA:g}) × raiz de {estrategias.HORIZONTE} × "
        "preço (movimento esperado de cerca de 1 mês). Os strikes são referências calculadas sobre o ativo: "
        "use o strike listado mais próximo na cadeia."
    )
    analises = analisar_universo(tuple(UNIVERSO))
    por_estrutura = {e: [] for e in estrategias.ESTRUTURAS}
    for an in analises:
        for ideia in estrategias.travas(an):
            por_estrutura[ideia["estrutura"]].append((an, ideia))

    for estrutura in estrategias.ESTRUTURAS:
        itens = sorted(por_estrutura[estrutura], key=lambda x: -abs(x[0]["score"]))
        st.subheader(f"{estrutura} ({len(itens)})")
        st.caption(estrategias.DESCRICAO[estrutura])
        if not itens:
            st.caption("Nenhum ativo atende às regras hoje.")
            continue
        tab = pd.DataFrame([{
            "Ativo": an["ativo"], "Último": an["preco"], "Força": an["score"],
            f"IFR {IFR_PERIODOS}": an["ifr"], "HiLo (stop ref.)": an["hilo"],
            "Strikes de referência": ideia["strikes"],
            "Mercado": ("Neutro" if estrutura == estrategias.NEUTRA else
                        mercado.mercado_a_favor("call" if estrutura in (estrategias.ALTA_DEB, estrategias.ALTA_CRED)
                                                else "put", REGIME)),
            "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, ideia in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, width="stretch",
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], ideia["texto"]) for an, ideia in itens])


# ---- Cadeia de opções (dados de fim de dia da B3) -------------------------------------
def fmt_cadeia(tab: pd.DataFrame):
    pct0 = ou_traco(lambda v: f"{v * 100:.0f}%")
    num2 = ou_traco(lambda v: f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    return tab.style.format({
        "Strike": num2, "Dist.": ou_traco(pct_sinal), "Bid": num2, "Ask": num2, "Último": num2,
        "Volume": ou_traco(lambda v: f"{v:,.0f}".replace(",", ".")), "Spread": pct0, "Vol. impl.": pct0,
        "Delta": ou_traco(lambda v: f"{v:+.2f}".replace(".", ",")), "Prob. exercício": pct0,
        "Theta/dia": ou_traco(lambda v: f"{v:+.3f}".replace(".", ",")),
    }).map(cor_estrutura, subset=["Tipo"])


with aba_cadeia:
    st.caption(
        "Fonte: arquivo COTAHIST da B3 (oficial, gratuito), publicado no fim de cada pregão: os dados são "
        "do fechamento do último pregão disponível, não em tempo real. Bid/ask = melhores ofertas no "
        "fechamento. Vol. implícita por Black-Scholes sobre o meio do book (ou o último negócio, se não "
        "houver oferta dos dois lados), com taxa = Selic meta e prazo em dias úteis/252."
    )
    resultado = pregao_opcoes(date.today())
    if resultado is None:
        st.error("Não foi possível baixar o COTAHIST da B3 nos últimos pregões. Tente de novo mais tarde.")
    else:
        pregao, cotahist = resultado
        selic = selic_atual()
        com_opcoes = set(opcoes_dados.resumo_liquidez(cotahist).index)
        lista = [t for t in UNIVERSO if t in com_opcoes] or sorted(com_opcoes)

        c1, c2, c3 = st.columns([1, 1, 1])
        ativo_op = c1.selectbox("Ativo", lista, index=lista.index("PETR4") if "PETR4" in lista else 0,
                                key="cadeia_ativo")
        vol_ref = None
        try:
            v = carregar(ativo_op)["Vol_EWMA"].iloc[-1]
            vol_ref = float(v) if pd.notna(v) else None
        except Exception:
            pass
        cad = opcoes_dados.cadeia(cotahist, ativo_op, pregao, selic, vol_ref)

        if cad.empty:
            st.caption("Sem opções para esse ativo no pregão.")
        else:
            vencs = cad.drop_duplicates("vencimento")[["vencimento", "du"]]
            rotulos = {f"{r.vencimento:%d/%m/%Y} ({r.du} d.u.)": r.vencimento for r in vencs.itertuples()}
            venc_sel = rotulos[c2.selectbox("Vencimento", list(rotulos), key="cadeia_venc")]
            tipos = c3.multiselect("Tipo", ["Call", "Put"], default=["Call", "Put"], key="cadeia_tipo")
            f1, f2 = st.columns([2, 1])
            faixa = f1.slider("Distância do strike ao preço (%)", -50, 50, (-20, 20), step=5, key="cadeia_faixa")
            so_negociadas = f2.checkbox("Só com negócio no pregão", value=False, key="cadeia_neg")

            sub = cad[(cad["vencimento"] == venc_sel) & cad["tipo"].isin(tipos)
                      & cad["dist"].between(faixa[0] / 100, faixa[1] / 100)]
            if so_negociadas:
                sub = sub[sub["negocios"] > 0]

            do_venc = cad[cad["vencimento"] == venc_sel]
            atm = do_venc.loc[(do_venc["dist"].abs() <= 0.03), "iv"].dropna()
            iv_atm = float(atm.mean()) if len(atm) else np.nan
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Preço (fech. B3)", brl(float(cad["spot"].iloc[0])))
            m2.metric("Pregão", f"{pregao:%d/%m/%Y}")
            m3.metric("Selic meta", f"{selic * 100:.2f}%".replace(".", ","))
            m4.metric("Vol. impl. ATM", "—" if pd.isna(iv_atm) else f"{iv_atm * 100:.0f}%")
            m5.metric("Vol. EWMA do ativo", "—" if vol_ref is None else f"{vol_ref * 100:.0f}%",
                      None if (vol_ref is None or pd.isna(iv_atm)) else
                      f"IV/vol = {iv_atm / vol_ref:.2f}x".replace(".", ","), delta_color="off")

            tab = sub.rename(columns={
                "opcao": "Opção", "tipo": "Tipo", "strike": "Strike", "dist": "Dist.", "moneyness": "Situação",
                "bid": "Bid", "ask": "Ask", "ultimo": "Último", "negocios": "Negócios", "volume": "Volume",
                "spread": "Spread", "iv": "Vol. impl.", "delta": "Delta", "prob_exercicio": "Prob. exercício",
                "theta": "Theta/dia",
            })[["Opção", "Tipo", "Strike", "Dist.", "Situação", "Bid", "Ask", "Último", "Negócios", "Volume",
                "Spread", "Vol. impl.", "Delta", "Prob. exercício", "Theta/dia"]]
            st.dataframe(fmt_cadeia(tab), hide_index=True, width="stretch",
                         height=min(40 + 35 * len(tab), 700))
            st.caption(
                "Dist. = strike ÷ preço − 1. Prob. exercício = N(d2), probabilidade neutra a risco de vencer "
                "dentro do dinheiro. Theta em R$ por dia útil. Quando não dá para calcular a vol implícita, "
                "as gregas usam a vol EWMA do ativo."
            )

            fig = go.Figure()
            for tipo, cor in (("Call", COR_HILO_ALTA), ("Put", COR_HILO_BAIXA)):
                d = do_venc[(do_venc["tipo"] == tipo) & do_venc["iv"].notna()]
                fig.add_trace(go.Scatter(x=d["strike"], y=d["iv"] * 100, mode="lines+markers", name=tipo,
                                         line=dict(color=cor), text=d["opcao"],
                                         hovertemplate="%{text}<br>Strike %{x:.2f}<br>IV %{y:.0f}%"))
            fig.add_vline(x=float(cad["spot"].iloc[0]), line_dash="dash", line_color="gray")
            if vol_ref:
                fig.add_hline(y=vol_ref * 100, line_dash="dot", line_color=COR_MM20,
                              annotation_text="vol EWMA", annotation_position="top left")
            fig.update_layout(title="Smile de volatilidade implícita", height=380, xaxis_title="Strike",
                              yaxis_title="Vol. implícita (%)", margin=dict(l=10, r=10, t=40, b=10),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

        with st.expander("Liquidez em opções por ativo no pregão"):
            liq = opcoes_dados.resumo_liquidez(cotahist).head(40).reset_index()
            liq["No universo"] = liq["ativo"].isin(UNIVERSO).map({True: "Sim", False: ""})
            st.dataframe(
                liq.rename(columns={"ativo": "Ativo", "series": "Séries", "negocios": "Negócios",
                                    "volume": "Volume (R$)"})
                .style.format({"Volume (R$)": lambda v: f"{v:,.0f}".replace(",", ".")}),
                hide_index=True, width="stretch",
            )


# ---- Pozinhos -------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Calculando pozinhos (simulando a regra de saída no histórico)...")
def _listar_pozinhos(hoje: date, versao: str, selic: float) -> tuple[date | None, pd.DataFrame]:
    resultado = pregao_opcoes(hoje)
    if resultado is None:
        return None, pd.DataFrame()
    pregao, cotahist = resultado
    analises = {an["ativo"]: an for an in _analisar_universo(tuple(UNIVERSO), versao)}
    return pregao, pz.listar(cotahist, pregao, selic, _historicos(versao), analises)


def cor_retorno(v: float) -> str:
    if pd.isna(v):
        return ""
    return VERDE if v > 0 else VERMELHO if v < -0.5 else AMARELO


def cor_sigma(v: float) -> str:
    if pd.isna(v):
        return ""
    return VERDE if v <= 1 else AMARELO if v <= 2 else VERMELHO


def cor_nota(v: str) -> str:
    return {"Quente": VERDE, "Morno": AMARELO, "Frio": VERMELHO}.get(v, "")


def pct_txt(v: float, casas: int = 0, sinal: bool = False) -> str:
    if pd.isna(v):
        return "—"
    fmt = f"{{:{'+' if sinal else ''}.{casas}f}}%"
    return fmt.format(v * 100).replace(".", ",")


def num_txt(v: float, casas: int = 2) -> str:
    return "—" if pd.isna(v) else f"{v:.{casas}f}".replace(".", ",")


ALVO_ROTULOS = {f"+{a * 100:.0f}%": a for a in pz.ALVOS_SAIDA}
EV80, PR80 = pz.col_ev(pz.ALVO_PADRAO), pz.col_prob(pz.ALVO_PADRAO)

COLS_POZ_RESUMO = {
    "opcao": "Opção", "ativo": "Ativo", "tipo": "Tipo", "vencimento": "Venc.", "du": "d.u.",
    "strike": "Strike", "dist": "Dist.", "premio": "Prêmio", "bid": "Bid", "negocios": "Negócios",
    "sigma_be": "σ até breakeven", PR80: "Chance +80% (hist.)", EV80: "EV saída +80%", "mercado": "Mercado",
}
COLS_POZ_EXTRA = {
    "sinal": "Sinal HiLo", "forca": "Força", "spread": "Spread", "prob_bs": "Prob. B&S",
    "prob_venc": "Prob. venc. (hist.)", "prob_toque": "Prob. toque (hist.)",
    "ret_esperado": "Ret. no venc. (hist.)", "mult_2s": "Mult. em 2σ",
}
ORDENS_POZ = {
    "EV com saída em +80%": (EV80, False), "Chance de bater +80%": (PR80, False),
    "Retorno no vencimento (hist.)": ("ret_esperado", False), "Mult. em 2σ": ("mult_2s", False),
    "Menos σ até o breakeven": ("sigma_be", True), "Mais negociados": ("negocios", False),
}
FORMATOS_POZ = {
    "Venc.": lambda v: f"{v:%d/%m/%y}", "Strike": ou_traco(lambda v: num_txt(v)),
    "Dist.": ou_traco(pct_sinal), "Prêmio": ou_traco(lambda v: num_txt(v)), "Bid": ou_traco(lambda v: num_txt(v)),
    "Força": ou_traco(lambda v: f"{int(v):+d}"), "σ até breakeven": ou_traco(lambda v: num_txt(v, 1) + "σ"),
    "Spread": lambda v: pct_txt(v), "Prob. B&S": lambda v: pct_txt(v, 1), "Prob. venc. (hist.)": lambda v: pct_txt(v, 1),
    "Prob. toque (hist.)": lambda v: pct_txt(v, 1), "Chance +80% (hist.)": lambda v: pct_txt(v, 1),
    "EV saída +80%": lambda v: pct_txt(v, 0, True), "Ret. no venc. (hist.)": lambda v: pct_txt(v, 0, True),
    "Mult. em 2σ": ou_traco(lambda v: num_txt(v, 1) + "x"),
}


def estilo_pozinhos(tab: pd.DataFrame):
    est = tab.style.format({c: f for c, f in FORMATOS_POZ.items() if c in tab.columns})
    for col, fn in (("Tipo", cor_estrutura), ("Sinal HiLo", cor_sinal), ("Mercado", cor_mercado),
                    ("σ até breakeven", cor_sigma), ("EV saída +80%", cor_retorno),
                    ("Ret. no venc. (hist.)", cor_retorno), ("Nota", cor_nota), ("EV com saída", cor_retorno)):
        if col in tab.columns:
            est = est.map(fn, subset=[col])
    return est


def a_favor_regime(tipo: str) -> str:
    return mercado.mercado_a_favor(tipo, REGIME)


with aba_pozinhos:
    st.caption(
        "Pozinhos = opções fora do dinheiro com prêmio de até R$ 0,15 (ask do fechamento, ou o último negócio), "
        "nos ativos da lista principal e do acompanhamento. Dados de fim de dia do COTAHIST da B3. "
        "Colunas 'hist.' vêm do histórico ajustado do próprio ativo, em janelas com o mesmo prazo da opção."
    )
    pregao_poz, poz = _listar_pozinhos(date.today(), VERSAO, selic_atual())
    if pregao_poz is None:
        st.error("Não foi possível baixar o COTAHIST da B3 nos últimos pregões. Tente de novo mais tarde.")
    elif poz.empty:
        st.caption("Nenhum pozinho encontrado nos ativos do universo.")
    else:
        f1, f2, f3, f4 = st.columns([1, 2, 2, 2])
        tipos_poz = f1.multiselect("Tipo", ["Call", "Put"], default=["Call", "Put"], key="poz_tipo")
        faixas = f2.multiselect("Faixa de prêmio", list(pz.FAIXAS), default=list(pz.FAIXAS), key="poz_faixa")
        du_max = f3.slider("Prazo máximo (dias úteis)", 1, pz.DU_MAX, 30, key="poz_du")
        ordem = f4.selectbox("Ordenar por", list(ORDENS_POZ), key="poz_ordem")
        g1, g2, g3, g4, g5 = st.columns(5)
        so_alinhados = g1.checkbox("Só alinhados com o HiLo", value=True, key="poz_alinhado")
        sem_semanais = g2.checkbox("Sem vencimentos semanais", value=True, key="poz_semanal",
                                   help="Tira as séries que não vencem na 3ª sexta do mês (códigos com W1, W2...).")
        com_bid = g3.checkbox("Exige oferta de compra", value=True, key="poz_bid",
                              help="Só opções com bid no fechamento: dá para vender antes do vencimento.")
        com_neg = g4.checkbox("Só com negócio no pregão", value=True, key="poz_neg")
        detalhado = g5.toggle("Todas as colunas", value=False, key="poz_detalhado")

        filtro = poz["tipo"].isin(tipos_poz) & (poz["du"] <= du_max)
        if faixas:
            na_faixa = pd.Series(False, index=poz.index)
            for nome in faixas:
                lo, hi = pz.FAIXAS[nome]
                na_faixa |= poz["premio"].between(lo, hi + 1e-9)
            filtro &= na_faixa
        if so_alinhados:
            filtro &= poz["alinhado"]
        if sem_semanais:
            filtro &= ~poz["semanal"]
        if com_bid:
            filtro &= poz["bid"].notna()
        if com_neg:
            filtro &= poz["negocios"] > 0
        sel = poz[filtro].copy()
        sel["mercado"] = sel["tipo"].map(a_favor_regime)
        col, asc = ORDENS_POZ[ordem]
        sel = sel.sort_values(col, ascending=asc, na_position="last")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pregão", f"{pregao_poz:%d/%m/%Y}")
        m2.metric("Pozinhos no filtro", f"{len(sel)} de {len(poz)}")
        m3.metric("EV positivo saindo em +80%", int((sel[EV80] > 0).sum()))
        m4.metric("Chance média de bater +80%", pct_txt(sel[PR80].mean(), 1))

        if sel.empty:
            st.caption("Nenhum pozinho com esses filtros.")
        else:
            cols = {**COLS_POZ_RESUMO, **(COLS_POZ_EXTRA if detalhado else {})}
            tab = sel[list(cols)].rename(columns=cols)
            st.dataframe(estilo_pozinhos(tab), hide_index=True, width="stretch", height=min(40 + 35 * len(tab), 600))
            st.caption(
                "Chance +80% = em quantas janelas históricas a opção chegou a valer o prêmio + 80% em algum "
                "fechamento antes do vencimento. EV saída +80% = retorno médio por real investido zerando nesse "
                "alvo (ou levando ao vencimento se não bater). Detalhes em 'Como ler'."
            )

            st.markdown("**Detalhe do pozinho**")
            escolha = st.selectbox("Opção", sel["opcao"].tolist(), key="poz_detalhe")
            o = sel[sel["opcao"] == escolha].iloc[0]
            st.markdown(pz.leitura(o).replace("$", "\\$"))
            jan = pz.janelas(_carregar(o["ativo"], VERSAO), int(o["du"]))
            if jan is not None:
                fig = go.Figure(go.Histogram(x=jan["fim"] * 100, nbinsx=60, marker_color=COR_IFR,
                                             name=f"Retorno em {o['du']} d.u."))
                fig.add_vline(x=o["dist_be"] * 100, line_dash="dash",
                              line_color=COR_HILO_ALTA if o["tipo"] == "Call" else COR_HILO_BAIXA,
                              annotation_text="breakeven", annotation_position="top")
                fig.update_layout(
                    title=f"{o['ativo']}: retornos históricos em {o['du']} pregões ({jan['n']} janelas)",
                    height=330, xaxis_title="Retorno (%)", yaxis_title="Janelas", showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, width="stretch")

        with st.expander("Como ler as colunas"):
            st.markdown(
                "- **σ até breakeven**: quanto o ativo precisa andar até o vencimento, em desvios-padrão da vol "
                "EWMA. Até 1σ (verde) é movimento comum; acima de 2σ (vermelho) é aposta de cauda.\n"
                "- **Chance +80% (hist.)**: o caminho de cada janela histórica de mesmo prazo é aplicado ao preço de "
                "hoje e a opção é reavaliada todo dia por Black-Scholes; conta quantas vezes ela chegou a valer o "
                "prêmio + 80%.\n"
                "- **EV saída +80%**: retorno médio seguindo a regra (zera no alvo, senão leva ao vencimento). É a "
                "métrica que conversa com o jeito que você opera.\n"
                "- **Ret. no venc. (hist.)**: o mesmo, mas segurando até o vencimento. Quase sempre negativo em "
                "pozinho: é por isso que a saída no alvo importa.\n"
                "- **Prob. B&S / venc. / toque**: chance embutida no preço vs. frequência histórica de terminar ou "
                "tocar além do breakeven.\n"
                "- Limitações: saída avaliada no fechamento, sem custo de spread na venda, histórico de ~3 anos com "
                "janelas sobrepostas. Use como triagem."
            )


# ---- Pozinhos - hot picks -----------------------------------------------------------------
def card_pick(o: pd.Series, alvo: float) -> None:
    with st.container(border=True):
        cor = COR_HILO_ALTA if o["tipo"] == "Call" else COR_HILO_BAIXA
        nota_cor = {"Quente": "#00c853", "Morno": "#facc15", "Frio": "#ff1744"}.get(o["nota"], "#9e9e9e")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-size:1.15rem;font-weight:700'>{o['opcao']}</span>"
            f"<span style='background:{nota_cor};color:#000;padding:2px 8px;border-radius:6px;"
            f"font-weight:600;font-size:0.8rem'>{o['nota']}</span></div>"
            f"<div style='color:{cor};font-weight:600'>{o['ativo']} · {o['tipo']}</div>",
            unsafe_allow_html=True,
        )
        lado = "subir" if o["tipo"] == "Call" else "cair"
        st.caption(
            f"Strike {num_txt(o['strike'])} · venc. {o['vencimento']:%d/%m} ({o['du']} d.u.) · "
            f"prêmio R\\$ {num_txt(o['premio'])} (bid {num_txt(o['bid'])}) · {o['negocios']} negócios"
        )
        k1, k2, k3 = st.columns(3)
        k1.metric(f"Chance de +{alvo * 100:.0f}%", pct_txt(o["prob_alvo"]))
        k2.metric("EV com saída", pct_txt(o["ev"], 0, True))
        k3.metric("σ até breakeven", num_txt(o["sigma_be"], 1) + "σ")
        st.caption(
            f"Precisa {lado} {pct_txt(abs(o['dist_be']), 1)} até o vencimento para pagar o prêmio. "
            f"HiLo {'a favor' if o['alinhado'] else 'contra'} · mercado {o['mercado'].lower()} · "
            f"no vencimento (sem saída) o histórico dá {pct_txt(o['ret_esperado'], 0, True)}."
        )


with aba_hot:
    st.caption(
        "Os pozinhos com melhor relação entre chance de bater o seu alvo de saída e o ganho quando bate, "
        "pelo histórico do próprio ativo. Só entram opções negociáveis: com bid, com negócio no pregão, spread de "
        f"até {pz.SPREAD_MAX * 100:.0f}% do meio e pelo menos {pz.PROB_MIN * 100:.0f}% de chance de bater o alvo."
    )
    pregao_hot, poz_hot = _listar_pozinhos(date.today(), VERSAO, selic_atual())
    if pregao_hot is None:
        st.error("Não foi possível baixar o COTAHIST da B3 nos últimos pregões. Tente de novo mais tarde.")
    elif poz_hot.empty:
        st.caption("Nenhum pozinho encontrado nos ativos do universo.")
    else:
        h1, h2, h3, h4 = st.columns([1, 1, 2, 1])
        tipos_hot = h1.multiselect("Tipo", ["Call", "Put"], default=["Call", "Put"], key="hot_tipo")
        alvo_rot = h2.selectbox("Zerar com", list(ALVO_ROTULOS), index=list(ALVO_ROTULOS.values()).index(pz.ALVO_PADRAO),
                                key="hot_alvo")
        du_hot = h3.slider("Prazo máximo (dias úteis)", 1, pz.DU_MAX, 30, key="hot_du")
        n_picks = h4.number_input("Quantos picks", 2, 20, 6, step=2, key="hot_n")
        i1, i2, i3, i4 = st.columns(4)
        sem_sem_hot = i1.checkbox("Sem vencimentos semanais", value=True, key="hot_semanal")
        alinh_hot = i2.checkbox("Só alinhados com o HiLo", value=True, key="hot_alinhado")
        merc_hot = i3.checkbox("Excluir os contra o regime de mercado", value=False, key="hot_mercado")
        um_por_ativo = i4.checkbox("Um pick por ativo", value=False, key="hot_um_por_ativo",
                                   help="Nos cartões, mostra só o melhor pozinho de cada ativo.")
        # Opções vêm do universo inteiro (não dos picks filtrados) para a seleção não sumir ao mudar outros filtros.
        excluir_hot = st.multiselect(
            "Excluir ativos", sorted(poz_hot["ativo"].unique()), key="hot_excluir", placeholder="Ex.: BBAS3",
            help="Tira o ativo dos cartões e do ranking, para ver os próximos da fila.",
        )
        alvo = ALVO_ROTULOS[alvo_rot]

        picks = pz.hot_picks(poz_hot, alvo, tuple(tipos_hot), du_hot, sem_sem_hot, alinh_hot, a_favor_regime, merc_hot)
        if excluir_hot:
            picks = picks[~picks["ativo"].isin(excluir_hot)].reset_index(drop=True)
        positivos = picks[picks["ev"] > 0]
        txt_excl = f" · excluídos: {', '.join(excluir_hot)}" if excluir_hot else ""
        st.markdown(f"Pregão de **{pregao_hot:%d/%m/%Y}** · regime **{REGIME['rotulo']}** · "
                    f"{len(positivos)} de {len(picks)} pozinhos negociáveis com EV positivo zerando em {alvo_rot}"
                    f"{txt_excl}.")

        if picks.empty:
            st.info("Nenhum pozinho negociável com esses filtros.")
        else:
            if positivos.empty:
                st.warning("Nenhum pozinho com EV positivo hoje. Abaixo, os menos piores: só se tiver convicção própria.")
                base = picks
            else:
                base = positivos
            if um_por_ativo:
                base = base.drop_duplicates("ativo")  # já vem ordenado por EV: fica o melhor de cada ativo
            mostrar = base.head(int(n_picks))
            colunas = st.columns(2)
            for i, (_, o) in enumerate(mostrar.iterrows()):
                with colunas[i % 2]:
                    card_pick(o, alvo)

            with st.expander(f"Ranking completo ({len(picks)} pozinhos negociáveis)"):
                rk = picks.rename(columns={
                    "nota": "Nota", "opcao": "Opção", "ativo": "Ativo", "tipo": "Tipo", "vencimento": "Venc.",
                    "du": "d.u.", "strike": "Strike", "premio": "Prêmio", "bid": "Bid", "negocios": "Negócios",
                    "sigma_be": "σ até breakeven", "prob_alvo": "Chance do alvo", "ev": "EV com saída",
                    "mercado": "Mercado",
                })[["Nota", "Opção", "Ativo", "Tipo", "Venc.", "d.u.", "Strike", "Prêmio", "Bid", "Negócios",
                    "σ até breakeven", "Chance do alvo", "EV com saída", "Mercado"]]
                st.dataframe(
                    estilo_pozinhos(rk).format({"Chance do alvo": lambda v: pct_txt(v, 1),
                                                "EV com saída": lambda v: pct_txt(v, 0, True)}),
                    hide_index=True, width="stretch", height=min(40 + 35 * len(rk), 600),
                )

        with st.expander("Como os hot picks são escolhidos"):
            st.markdown(
                "1. **Universo**: pozinhos da aba ao lado, só os negociáveis (bid, negócio no pregão, spread "
                "aceitável), sem semanais e alinhados ao HiLo, se marcado. Ativos em 'Excluir ativos' ficam de fora.\n"
                "2. **Simulação da sua regra**: para cada janela histórica com o mesmo prazo da opção, o ativo "
                "refaz o caminho daquele período a partir do preço de hoje, e a opção é reavaliada todo dia "
                "(Black-Scholes, vol = média entre a implícita e a EWMA do ativo). Se ela chega a valer o prêmio + "
                "alvo, a operação é zerada ali; se não, vale o que valer no vencimento.\n"
                "3. **EV com saída** = média do resultado de todas as janelas, por real investido. É a nota "
                "principal: junta a chance de chegar lá e o quanto se ganha.\n"
                "4. **Nota**: Quente = EV ≥ +15% e chance ≥ 20%; Morno = EV positivo; Frio = EV negativo.\n\n"
                "O histórico é de ~3 anos, as janelas se sobrepõem e a saída é avaliada no fechamento sem custo de "
                "spread. Um EV de +10% no histórico não é garantia de nada: é o melhor ponto de partida para a "
                "sua análise, não o fim dela."
            )


# ---- Mercado e exposição -------------------------------------------------------------
COR_REGIME = {"Alta ampla": "#00c853", "Alta estreita": "#9ccc65", "Misto": "#9e9e9e",
              "Baixa estreita": "#ff8a65", "Baixa ampla": "#ff1744", "Indefinido": "#9e9e9e"}


@st.cache_data(ttl=900, show_spinner="Calculando exposição...")
def _exposicao(versao: str, ops_csv: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    ops = pd.read_json(io.StringIO(ops_csv))
    hist = _historicos(versao)
    return mercado.exposicao(ops, hist)


with aba_mercado:
    # Regime
    st.subheader("Regime de mercado")
    cor = COR_REGIME.get(REGIME["rotulo"], "#9e9e9e")
    st.markdown(f"<span style='background:{cor};color:#000;padding:4px 10px;border-radius:6px;"
                f"font-weight:600'>{REGIME['rotulo']}</span>", unsafe_allow_html=True)
    st.markdown(REGIME["texto"])
    if not AMPLITUDE.empty:
        try:
            ind = carregar(mercado.INDICE).tail(len(AMPLITUDE) + 5)
        except Exception:
            ind = None
        u = AMPLITUDE.iloc[-1]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("% em Call (HiLo)", f"{u['pct_call'] * 100:.0f}%",
                  f"{(u['pct_call'] - AMPLITUDE['pct_call'].iloc[-6]) * 100:+.0f} p.p. em 5 dias"
                  if len(AMPLITUDE) > 6 else None)
        r2.metric("% acima da MM200", f"{u['pct_mm200'] * 100:.0f}%")
        r3.metric("% acima da MM20", f"{u['pct_mm20'] * 100:.0f}%")
        r4.metric("Ativos na amplitude", int(u["n"]))

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for col, nome, cor_l in (("pct_call", "% em Call (HiLo)", COR_HILO_ALTA),
                                 ("pct_mm200", "% acima MM200", COR_MM200),
                                 ("pct_mm20", "% acima MM20", COR_MM20)):
            fig.add_trace(go.Scatter(x=AMPLITUDE.index, y=AMPLITUDE[col] * 100, name=nome,
                                     line=dict(color=cor_l, width=2)), secondary_y=False)
        if ind is not None:
            ind = ind[ind.index >= AMPLITUDE.index[0]]
            fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name=mercado.INDICE,
                                     line=dict(color="gray", width=1.5, dash="dot")), secondary_y=True)
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_yaxes(title_text="% do universo", range=[0, 100], secondary_y=False)
        fig.update_yaxes(title_text=mercado.INDICE, secondary_y=True, showgrid=False)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Amplitude calculada sobre o universo (lista principal + acompanhamento, sem índices). "
            f"Alta/baixa 'ampla' quando {mercado.AMPLA * 100:.0f}% ou mais (ou {mercado.ESTREITA * 100:.0f}% ou "
            f"menos) do universo está em Call. As abas de ideias e de pozinhos mostram a coluna Mercado: "
            "se a ideia está a favor ou contra o regime."
        )

    # Exposição
    st.divider()
    st.subheader("Exposição das operações abertas")
    if OPERACOES.empty:
        st.caption("Nenhuma operação aberta.")
    else:
        exp, resumo, adj = _exposicao(VERSAO, OPERACOES.to_json())
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Operações", f"{resumo['call']} Call · {resumo['put']} Put")
        e2.metric("Beta líquido médio", f"{resumo['beta_liquido']:+.2f}".replace(".", ","),
                  help="Média de (Call = +1, Put = −1) × beta contra o BOVA11. −1 = o livro se comporta como "
                       "uma put no índice; 0 = neutro ao mercado.")
        n_eff = resumo["apostas_independentes"]
        e3.metric("Apostas independentes", "—" if pd.isna(n_eff) else f"{n_eff:.1f}".replace(".", ","),
                  help="n² ÷ Σ dᵢdⱼρᵢⱼ. Quanto mais as operações andam juntas na mesma direção, menor o número.")
        e4.metric("Com HiLo contra a posição", resumo["contra_hilo"])

        tab = exp.rename(columns={"ativo": "Ativo", "tipo": "Tipo", "setor": "Setor", "grupo": "Grupo",
                                  "beta": "Beta", "vol": "Vol. EWMA", "hilo_a_favor": "HiLo a favor"})
        tab = tab[["Ativo", "Tipo", "Setor", "Grupo", "Beta", "Vol. EWMA", "HiLo a favor"]]
        tab["Mercado"] = tab["Tipo"].map(lambda t: mercado.mercado_a_favor(t, REGIME))
        st.dataframe(
            tab.style.map(cor_estrutura, subset=["Tipo"])
            .map(lambda v: cor_mercado({"Sim": "A favor", "Não": "Contra"}.get(v, "")), subset=["HiLo a favor"])
            .map(cor_mercado, subset=["Mercado"])
            .format({"Beta": ou_traco(lambda v: f"{v:.2f}".replace(".", ",")),
                     "Vol. EWMA": ou_traco(lambda v: f"{v * 100:.0f}%")}),
            hide_index=True, width="stretch", height=40 + 35 * len(tab),
        )

        x1, x2 = st.columns([1, 1])
        por_grupo = exp.groupby(["grupo", "tipo"]).size().unstack(fill_value=0)
        fig_g = go.Figure()
        for tipo, cor_b in (("Call", COR_HILO_ALTA), ("Put", COR_HILO_BAIXA)):
            if tipo in por_grupo:
                fig_g.add_trace(go.Bar(y=por_grupo.index, x=por_grupo[tipo] * (1 if tipo == "Call" else -1),
                                       name=tipo, orientation="h", marker_color=cor_b))
        fig_g.update_layout(title="Operações por grupo (Put à esquerda, Call à direita)", barmode="relative",
                            height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        x1.plotly_chart(fig_g, width="stretch")

        if not adj.empty:
            fig_c = go.Figure(go.Heatmap(z=adj.to_numpy(), x=adj.columns, y=adj.index, zmin=-1, zmax=1, zmid=0,
                                         colorscale="RdYlGn", reversescale=True,
                                         hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
            fig_c.update_layout(title="Correlação ajustada pela direção", height=360,
                                margin=dict(l=10, r=10, t=40, b=10))
            x2.plotly_chart(fig_c, width="stretch")
        st.caption(
            f"Beta e correlação com retornos diários dos últimos {mercado.JANELA_BETA} pregões (preços ajustados). "
            "No mapa, vermelho = as duas operações ganham e perdem juntas (mesma aposta); verde = uma protege a "
            "outra. Cada operação pesa igual, já que o app não guarda tamanho."
        )

        st.markdown("**Cenário: e se o índice andar?**")
        mov = st.slider(f"Movimento do {mercado.INDICE} em um dia (%)", -8.0, 8.0, 3.0, step=0.5, key="cen_mov")
        cen = mercado.cenario(exp, mov / 100)
        contra = cen[cen["situacao"] == "Contra"]
        mov_txt = f"{mov:+.1f}".replace(".", ",")
        st.caption(f"Com o {mercado.INDICE} {mov_txt}%, {len(contra)} de {len(cen)} operações andariam contra, "
                   "pelo beta de cada ativo.")
        st.dataframe(
            cen.rename(columns={"ativo": "Ativo", "tipo": "Tipo", "beta": "Beta",
                                "mov_esperado": "Mov. esperado do ativo", "situacao": "Situação"})
            [["Ativo", "Tipo", "Beta", "Mov. esperado do ativo", "Situação"]]
            .style.map(cor_estrutura, subset=["Tipo"]).map(cor_mercado, subset=["Situação"])
            .format({"Beta": ou_traco(lambda v: f"{v:.2f}".replace(".", ",")),
                     "Mov. esperado do ativo": ou_traco(pct_sinal)}),
            hide_index=True, width="stretch", height=40 + 35 * len(cen),
        )
