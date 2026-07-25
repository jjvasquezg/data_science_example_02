"""Storytelling interactivo sobre el agro colombiano."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent / ".env")


st.set_page_config(
    page_title="Agro Colombia | Storytelling",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "datasets" / "agro_colombia.csv"
TECH_ORDER = ["Bajo", "Medio", "Alto", "Muy Alto"]
GREEN = "#146b4d"
GREEN_DARK = "#0b4f3b"
GREEN_LIGHT = "#d9eee5"
GOLD = "#b66a16"
INK = "#172b27"
GROQ_MODEL = "llama-3.3-70b-versatile"
CHAT_SYSTEM_PROMPT = """Eres un asistente conversacional experto en cultura general e historia mundial.
Responde en español, con claridad y rigor. Explica el contexto, las causas y las consecuencias
cuando sea útil. Si una fecha o dato es discutible, indícalo explícitamente. No inventes fuentes,
citas ni hechos; reconoce cuando no tengas suficiente certeza. Mantén las respuestas accesibles,
interesantes y de extensión moderada. Puedes responder preguntas de seguimiento usando el contexto
de la conversación, pero mantén el foco en cultura general e historia mundial."""


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    data["Fecha_Ultima_Auditoria"] = pd.to_datetime(data["Fecha_Ultima_Auditoria"])
    data["Rendimiento_Ton_Ha"] = data["Produccion_Anual_Ton"] / data["Area_Hectareas"]
    data["Ingreso_Estimado_COP"] = (
        data["Produccion_Anual_Ton"] * data["Precio_Venta_Por_Ton_COP"]
    )
    data["Riego"] = np.where(
        data["Sistema_Riego_Tecnificado"], "Tecnificado", "Convencional"
    )
    data["Nivel_Tecnificacion"] = pd.Categorical(
        data["Nivel_Tecnificacion"], categories=TECH_ORDER, ordered=True
    )
    return data


def money(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f} mil M"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f} M"
    return f"${value:,.0f}"


def chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=45, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color=INK),
        legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#dce8e3")
    fig.update_yaxes(gridcolor="#e8f0ed", zeroline=False)
    return fig


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        return None
    return Groq(api_key=api_key)


def reset_chat():
    st.session_state.chat_messages = []


st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #172b27; }
    [data-testid="stSidebar"] { background: #edf5f1; border-right: 1px solid #d6e5de; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] small { color: #17352d !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #17352d !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #45635c !important; }
    .hero { background: linear-gradient(115deg,#123c31 0%,#146b4d 70%,#2f8a63 100%);
            padding: 2.2rem 2.5rem; border-radius: 18px; color: white; margin-bottom: 1.4rem; }
    .hero h1 { font-size: 2.8rem; margin: 0; letter-spacing: -1px; }
    .hero p { font-size: 1.1rem; max-width: 760px; margin: .65rem 0 0; color: #e3f3e6; }
    .eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: .75rem; font-weight: 700; color: #b8e0bd; }
    .chapter { color: #0b6044; text-transform: uppercase; letter-spacing: 1.6px; font-size: .75rem; font-weight: 800; margin-top: 1.2rem; }
    .story { color: #36564d; font-size: 1.05rem; line-height: 1.55; }
    div[data-testid="stMetric"] { background: #f4f8f6; padding: 1rem 1.1rem; border-radius: 13px; border: 1px solid #cfe1d9; }
    div[data-testid="stMetric"] label { color: #45635c; }
    div[data-testid="stMetricValue"] { color: #123c31; }
    </style>
    """,
    unsafe_allow_html=True,
)

df = load_data()

with st.sidebar:
    st.markdown("## 🌱 Agro Colombia")
    st.caption("Explora qué hace productiva y rentable a una finca.")
    st.markdown("### Filtros")
    departments = st.multiselect(
        "Departamento", sorted(df["Departamento"].unique()), default=sorted(df["Departamento"].unique())
    )
    crops = st.multiselect(
        "Tipo de cultivo", sorted(df["Tipo_Cultivo"].unique()), default=sorted(df["Tipo_Cultivo"].unique())
    )
    tech = st.multiselect("Nivel de tecnificación", TECH_ORDER, default=TECH_ORDER)
    irrigation = st.radio("Sistema de riego", ["Todos", "Tecnificado", "Convencional"], index=0)
    st.divider()
    st.caption("Fuente: agro_colombia.csv · 500 registros · datos de referencia 2025")

filtered = df[
    df["Departamento"].isin(departments)
    & df["Tipo_Cultivo"].isin(crops)
    & df["Nivel_Tecnificacion"].isin(tech)
].copy()
if irrigation != "Todos":
    filtered = filtered[filtered["Riego"] == irrigation]

st.markdown(
    '<div class="hero"><div class="eyebrow">Panorama productivo · Colombia</div>'
    '<h1>La tierra cuenta una historia</h1>'
    '<p>500 fincas, cinco cultivos y una pregunta: ¿dónde están las oportunidades para producir mejor, vender mejor y cerrar brechas?</p></div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No hay fincas que coincidan con los filtros seleccionados.")
    st.stop()

total_area = filtered["Area_Hectareas"].sum()
total_production = filtered["Produccion_Anual_Ton"].sum()
total_income = filtered["Ingreso_Estimado_COP"].sum()
avg_yield = total_production / total_area

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fincas analizadas", f"{len(filtered):,}", f"{len(filtered) / len(df):.0%} del universo")
k2.metric("Producción anual", f"{total_production:,.0f} t")
k3.metric("Rendimiento promedio", f"{avg_yield:.2f} t/ha")
k4.metric("Ingreso estimado", money(total_income))

st.markdown('<div class="chapter">Capítulo 1 · El mapa productivo</div><h2>La producción no se reparte por igual</h2>', unsafe_allow_html=True)
st.markdown('<p class="story">El primer patrón aparece al mirar el territorio: algunos departamentos concentran volumen, mientras otros destacan por el valor generado por hectárea.</p>', unsafe_allow_html=True)

by_dep = filtered.groupby("Departamento", as_index=False).agg(
    Produccion=("Produccion_Anual_Ton", "sum"),
    Area=("Area_Hectareas", "sum"),
    Ingreso=("Ingreso_Estimado_COP", "sum"),
    Fincas=("ID_Finca", "count"),
)
by_dep["Rendimiento"] = by_dep["Produccion"] / by_dep["Area"]
left, right = st.columns([1.15, 1])
with left:
    fig = px.bar(by_dep.sort_values("Produccion"), x="Produccion", y="Departamento", orientation="h", text_auto=".0f", color="Rendimiento", color_continuous_scale=["#b7d9ca", GREEN_DARK])
    fig.update_layout(coloraxis_colorbar_title="t/ha", xaxis_title="Toneladas", yaxis_title="")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
with right:
    fig = px.scatter(by_dep, x="Rendimiento", y="Ingreso", size="Area", color="Departamento", text="Departamento", hover_data=["Fincas", "Produccion"])
    fig.update_traces(textposition="top center")
    fig.update_yaxes(title="Ingreso estimado (COP)", tickprefix="$", tickformat="~s")
    fig.update_xaxes(title="Rendimiento (t/ha)")
    st.plotly_chart(chart_layout(fig), use_container_width=True)

st.markdown('<div class="chapter">Capítulo 2 · La palanca tecnológica</div><h2>Más tecnología, más consistencia</h2>', unsafe_allow_html=True)
st.markdown('<p class="story">La tecnificación no solo cambia el promedio: permite comparar sistemas y detectar qué tan grande es la brecha entre una finca convencional y una finca equipada.</p>', unsafe_allow_html=True)

tech_summary = filtered.groupby(["Nivel_Tecnificacion", "Riego"], observed=False, as_index=False).agg(
    Rendimiento=("Rendimiento_Ton_Ha", "mean"), Precio=("Precio_Venta_Por_Ton_COP", "mean"), Fincas=("ID_Finca", "count")
)
left, right = st.columns(2)
with left:
    fig = px.bar(tech_summary, x="Nivel_Tecnificacion", y="Rendimiento", color="Riego", barmode="group", text_auto=".2f", category_orders={"Nivel_Tecnificacion": TECH_ORDER}, color_discrete_map={"Tecnificado": GREEN_DARK, "Convencional": "#9aaea6"})
    fig.update_layout(yaxis_title="Toneladas por hectárea", xaxis_title="", legend_title="Riego")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
with right:
    fig = px.box(filtered, x="Nivel_Tecnificacion", y="Rendimiento_Ton_Ha", color="Riego", category_orders={"Nivel_Tecnificacion": TECH_ORDER}, color_discrete_map={"Tecnificado": GREEN_DARK, "Convencional": "#9aaea6"})
    fig.update_layout(yaxis_title="Rendimiento (t/ha)", xaxis_title="", showlegend=False)
    st.plotly_chart(chart_layout(fig), use_container_width=True)

st.markdown('<div class="chapter">Capítulo 3 · La economía del cultivo</div><h2>No todo volumen significa el mismo valor</h2>', unsafe_allow_html=True)
st.markdown('<p class="story">El precio por tonelada reordena el panorama. Un cultivo puede producir menos toneladas y aun así aportar más ingreso estimado cuando combina rendimiento y valor de mercado.</p>', unsafe_allow_html=True)

by_crop = filtered.groupby("Tipo_Cultivo", as_index=False).agg(
    Produccion=("Produccion_Anual_Ton", "sum"), Ingreso=("Ingreso_Estimado_COP", "sum"), Precio=("Precio_Venta_Por_Ton_COP", "mean"), Rendimiento=("Rendimiento_Ton_Ha", "mean")
)
fig = px.bar(by_crop.sort_values("Ingreso"), x="Ingreso", y="Tipo_Cultivo", orientation="h", color="Precio", text="Rendimiento", color_continuous_scale=["#e7c778", "#b66a32"])
fig.update_traces(texttemplate="%{text:.1f} t/ha", textposition="outside")
fig.update_layout(xaxis_title="Ingreso estimado (COP)", yaxis_title="", coloraxis_colorbar_title="Precio/t (COP)", xaxis_tickprefix="$", xaxis_tickformat="~s")
st.plotly_chart(chart_layout(fig, 410), use_container_width=True)

st.markdown('<div class="chapter">Capítulo 4 · La oportunidad</div><h2>¿Dónde priorizar la próxima intervención?</h2>', unsafe_allow_html=True)
st.markdown('<p class="story">La oportunidad es mayor donde el rendimiento está por debajo del promedio, pero existe una base productiva suficiente para que una mejora tenga impacto.</p>', unsafe_allow_html=True)

priority = by_dep.copy()
priority["Brecha_vs_promedio"] = avg_yield - priority["Rendimiento"]
priority["Prioridad"] = np.where(priority["Brecha_vs_promedio"] > 0, "Cerrar brecha", "Referente")
priority = priority.sort_values(["Prioridad", "Brecha_vs_promedio"], ascending=[True, False])
fig = px.bar(priority.sort_values("Brecha_vs_promedio"), x="Brecha_vs_promedio", y="Departamento", orientation="h", color="Prioridad", text="Rendimiento", color_discrete_map={"Cerrar brecha": GOLD, "Referente": GREEN})
fig.update_traces(texttemplate="%{text:.2f} t/ha", textposition="outside")
fig.update_layout(xaxis_title="Brecha de rendimiento frente al promedio (t/ha)", yaxis_title="")
st.plotly_chart(chart_layout(fig), use_container_width=True)

with st.expander("Ver datos y definiciones"):
    st.caption("Ingreso estimado = producción anual × precio de venta por tonelada. Rendimiento = producción anual ÷ área.")
    display = filtered[["ID_Finca", "Departamento", "Tipo_Cultivo", "Area_Hectareas", "Produccion_Anual_Ton", "Rendimiento_Ton_Ha", "Riego", "Nivel_Tecnificacion", "Precio_Venta_Por_Ton_COP", "Tipo_Suelo", "Fecha_Ultima_Auditoria"]].copy()
    display.columns = ["Finca", "Departamento", "Cultivo", "Área (ha)", "Producción (t)", "Rendimiento (t/ha)", "Riego", "Tecnificación", "Precio/t (COP)", "Suelo", "Última auditoría"]
    st.dataframe(display, use_container_width=True, hide_index=True)

st.markdown('<div class="chapter">Asistente de conocimiento</div><h2>Conversa con HistoriaBot</h2>', unsafe_allow_html=True)
st.markdown('<p class="story">Pregunta por civilizaciones, guerras, personajes, inventos, fechas o conexiones entre hechos históricos.</p>', unsafe_allow_html=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

chat_client = get_groq_client()
if chat_client is None:
    st.warning("No se encontró GROQ_API_KEY. Configura esta variable en tu archivo .env para activar el asistente.")
else:
    chat_col, action_col = st.columns([5, 1])
    with action_col:
        st.button("Limpiar chat", on_click=reset_chat, use_container_width=True)

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ejemplo: ¿Por qué cayó el Imperio romano de Occidente?")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        api_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        api_messages.extend(st.session_state.chat_messages)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                completion = chat_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=api_messages,
                    temperature=0.35,
                    max_tokens=900,
                )
                answer = completion.choices[0].message.content
                response_placeholder.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as error:
                response_placeholder.error(
                    "No pude responder en este momento. Revisa la clave de Groq, la cuota disponible "
                    "o la conexión e inténtalo de nuevo."
                )
                st.caption(f"Detalle técnico: {type(error).__name__}")
