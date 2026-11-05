import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import cotacoes
import estrategias
import operacoes as ops_mod
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


@st.cache_data(ttl=900, show_spinner=False)
def carregar(ticker: str) -> pd.DataFrame:
    """Histórico (do cache em disco, atualizado só nos últimos dias) + indicadores."""
    return calcular(cotacoes.obter(ticker))


@st.cache_data(ttl=900, show_spinner="Analisando ativos...")
def analisar_universo(tickers: tuple) -> list[dict]:
    """Resumo de indicadores de cada ativo (usa os dados já em cache)."""
    saida = []
    for t in tickers:
        try:
            saida.append({"ativo": t, **estrategias.analisar(carregar(t))})
        except Exception:
            pass
    return saida


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


def estilo_ideias(tab: pd.DataFrame):
    mapa = {
        "Último": brl, "HiLo (stop ref.)": brl,
        "Força": lambda v: f"{int(v):+d}",
        f"IFR {IFR_PERIODOS}": lambda v: f"{v:.1f}",
        "Dist. MM20": pct_sinal, "Dist. MM200": pct_sinal,
    }
    est = tab.style.format({c: ou_traco(f) for c, f in mapa.items() if c in tab.columns})
    dist = [c for c in ("Dist. MM20", "Dist. MM200") if c in tab.columns]
    return est.map(cor_dist, subset=dist) if dist else est


def explicacoes(itens: list[tuple[str, str]]):
    """Lista de 'ativo — porquê'. O $ é escapado para o markdown não virar fórmula."""
    for ativo, texto in itens:
        st.markdown(f"- **{ativo}**: " + texto.replace("$", "\\$"))


# Barra lateral: controles de dados (valem para todas as abas)
with st.sidebar:
    st.header("Dados")
    if st.button("Atualizar cotações", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    if st.button("Refazer histórico completo", use_container_width=True):
        cotacoes.limpar_cache()
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "Atualizar baixa só os últimos dias. Refazer apaga a pasta dados/ e "
        "baixa tudo de novo (use após desdobramentos/grupamentos)."
    )

(aba_ativo, aba_consolidado, aba_operacoes, aba_watch,
 aba_direcional, aba_travas) = st.tabs([
    "Ativo por ativo", "Consolidado", "Operações abertas", "Acompanhamento",
    "Acompanhamento - direcional", "Acompanhamento - travas",
])

with aba_ativo:
    col1, col2 = st.columns([1, 1])
    ticker = col1.selectbox("Ativo", ATIVOS, index=ATIVOS.index("PETR4"))
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
    m1.metric("Último", f"R$ {ult['Close']:.2f}")
    m2.metric("Sinal HiLo", s)
    m3.metric(f"HiLo {HILO_PERIODOS}", f"{ult['HiLo']:.2f}" if pd.notna(ult["HiLo"]) else "—")
    m4.metric(f"IFR {IFR_PERIODOS}", f"{ult['IFR']:.1f}")
    m5.metric("MM9 / MM20", f"{ult['MM9']:.2f} / {ult['MM20']:.2f}")
    m6.metric("MM200", f"{ult['MM200']:.2f}" if pd.notna(ult["MM200"]) else "—")

    st.plotly_chart(grafico(df, ticker, candles), use_container_width=True)
    st.caption(f"Último candle: {df.index[-1]:%d/%m/%Y}")

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
                    hide_index=True, use_container_width=True,
                    height=40 + 35 * len(sub),
                )
    sem_sinal = tabela[~tabela["Sinal"].isin(["Call", "Put"])]["Ativo"]
    if len(sem_sinal):
        st.caption("Sem sinal ou sem dados: " + ", ".join(sem_sinal))

with aba_operacoes:
    if st.button("Adicionar operação", type="primary"):
        dialogo_adicionar()

    if OPERACOES.empty:
        st.caption("Nenhuma operação aberta.")
    else:
        n_call = int((OPERACOES["tipo"] == "Call").sum())
        n_put = int((OPERACOES["tipo"] == "Put").sum())
        st.caption(f"{len(OPERACOES)} operações abertas ({n_call} Call, {n_put} Put)")
        exibir = OPERACOES.rename(columns={"ativo": "Ativo", "tipo": "Tipo"})
        st.dataframe(
            exibir.style.map(cor_estrutura, subset=["Tipo"]),
            hide_index=True, use_container_width=True,
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
        hide_index=True, use_container_width=True,
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
        st.plotly_chart(grafico(carregar(t_watch), t_watch, c_watch), use_container_width=True)
    except Exception as e:
        st.error(f"Não foi possível carregar {t_watch}: {e}")

with aba_direcional:
    st.caption(AVISO)
    st.caption(
        "Força de −6 a +6: HiLo vale 2 pontos; MM9, MM20, MM200 e o alinhamento entre elas valem 1 cada. "
        f"Ideia direcional: força de {estrategias.SCORE_DIRECIONAL} ou mais (em módulo), com IFR fora de "
        f"sobrecompra ({estrategias.IFR_ALTO}) e sobrevenda ({estrategias.IFR_BAIXO})."
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
            "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, rot, _ in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, use_container_width=True,
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], txt) for an, _, txt in itens])

    itens = grupos["obs"]
    st.subheader(f"Em observação: tendência forte, mas esticada ({len(itens)})")
    if not itens:
        st.caption("Nenhum ativo nessa situação hoje.")
    else:
        tab = pd.DataFrame([{
            "Ativo": an["ativo"], "Situação": rot, "Último": an["preco"], "Força": an["score"],
            f"IFR {IFR_PERIODOS}": an["ifr"], "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, rot, _ in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, use_container_width=True,
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], txt) for an, _, txt in itens])

with aba_travas:
    st.caption(AVISO)
    st.caption(
        f"σ = desvio-padrão dos retornos diários dos últimos {estrategias.HORIZONTE} candles × raiz de "
        f"{estrategias.HORIZONTE} × preço (movimento esperado de cerca de 1 mês). Os strikes são referências "
        "calculadas sobre o ativo: use o strike listado mais próximo na cadeia."
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
            "Strikes de referência": ideia["strikes"], "Posição aberta": posicao_aberta(an["ativo"]),
        } for an, ideia in itens])
        st.dataframe(estilo_ideias(tab), hide_index=True, use_container_width=True,
                     height=40 + 35 * len(tab))
        explicacoes([(an["ativo"], ideia["texto"]) for an, ideia in itens])
