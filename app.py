"""Storytelling interactivo sobre el agro colombiano."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Agro Colombia | Storytelling",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "datasets" / "agro_colombia.csv"
TECH_ORDER = ["Bajo", "Medio", "Alto", "Muy Alto"]
GREEN = "#1f7a5a"
GOLD = "#e5a93d"
INK = "#16342f"


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


st.markdown(
    """
    <style>
    .stApp { background: #f5f8f5; }
    [data-testid="stSidebar"] { background: #16342f; }
    [data-testid="stSidebar"] * { color: #f4faf6 !important; }
    .hero { background: linear-gradient(115deg,#16342f 0%,#1f7a5a 70%,#579b68 100%);
            padding: 2.2rem 2.5rem; border-radius: 18px; color: white; margin-bottom: 1.4rem; }
    .hero h1 { font-size: 2.8rem; margin: 0; letter-spacing: -1px; }
    .hero p { font-size: 1.1rem; max-width: 760px; margin: .65rem 0 0; color: #e3f3e6; }
    .eyebrow { text-transform: uppercase; letter-spacing: 2px; font-size: .75rem; font-weight: 700; color: #b8e0bd; }
    .chapter { color: #1f7a5a; text-transform: uppercase; letter-spacing: 1.6px; font-size: .75rem; font-weight: 800; margin-top: 1.2rem; }
    .story { color: #45635c; font-size: 1.05rem; line-height: 1.55; }
    div[data-testid="stMetric"] { background: white; padding: 1rem 1.1rem; border-radius: 13px; border: 1px solid #e1ece6; }
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
    fig = px.bar(by_dep.sort_values("Produccion"), x="Produccion", y="Departamento", orientation="h", text_auto=".0f", color="Rendimiento", color_continuous_scale=["#b9d9bd", GREEN])
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
    fig = px.bar(tech_summary, x="Nivel_Tecnificacion", y="Rendimiento", color="Riego", barmode="group", text_auto=".2f", category_orders={"Nivel_Tecnificacion": TECH_ORDER}, color_discrete_map={"Tecnificado": GREEN, "Convencional": "#c3d0cb"})
    fig.update_layout(yaxis_title="Toneladas por hectárea", xaxis_title="", legend_title="Riego")
    st.plotly_chart(chart_layout(fig), use_container_width=True)
with right:
    fig = px.box(filtered, x="Nivel_Tecnificacion", y="Rendimiento_Ton_Ha", color="Riego", category_orders={"Nivel_Tecnificacion": TECH_ORDER}, color_discrete_map={"Tecnificado": GREEN, "Convencional": "#c3d0cb"})
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
