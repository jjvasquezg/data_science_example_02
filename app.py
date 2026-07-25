from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).parent / ".env")


st.set_page_config(
    page_title="Text2EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#1c2b25"
DEEP = "#0f5c45"
TEAL = "#1f8a70"
MINT = "#dff3ea"
SAND = "#f7f3e8"
AMBER = "#b46a1f"
SOFT_BORDER = "#cfe3d8"
GROQ_MODEL = "llama-3.3-70b-versatile"

SAMPLE_TEXT = """En 2023, Colombia exporto cafe por 3.2 millones de sacos y genero 1.450 millones USD.
Brasil exporto 8.1 millones de sacos y genero 3.980 millones USD.
Vietnam alcanzo 2.7 millones de sacos y 2.150 millones USD.
En 2024, Colombia subio a 3.5 millones de sacos con 1.620 millones USD, mientras Brasil llego a
8.4 millones de sacos y 4.120 millones USD."""


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


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, #eef8f3 0%, transparent 28%),
                linear-gradient(180deg, #fcfdfc 0%, #f5fbf7 100%);
            color: #1c2b25;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #103e31 0%, #0f5c45 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: #f4fbf7 !important;
        }
        .hero {
            background: linear-gradient(120deg, #123c31 0%, #1f8a70 70%, #7dc7ac 100%);
            border: 1px solid rgba(255,255,255,0.16);
            padding: 2rem 2.2rem;
            border-radius: 22px;
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 18px 40px rgba(15, 92, 69, 0.12);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
            letter-spacing: -0.04em;
        }
        .hero p {
            margin: 0.75rem 0 0;
            max-width: 860px;
            color: #e8fbf2;
            font-size: 1.02rem;
            line-height: 1.6;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.75rem;
            font-weight: 700;
            color: #caf3df;
            margin-bottom: 0.65rem;
        }
        .card {
            background: rgba(255,255,255,0.85);
            border: 1px solid #cfe3d8;
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid #cfe3d8;
            border-radius: 16px;
            padding: 0.9rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #4a665d;
        }
        div[data-testid="stMetricValue"] {
            color: #123c31;
        }
        .section-kicker {
            color: #0f5c45;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height: int = 420):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font=dict(color=INK, family="Segoe UI, sans-serif"),
        legend=dict(orientation="h", y=1.08, x=0),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor="#e3efe8", linecolor=SOFT_BORDER)
    fig.update_yaxes(gridcolor="#e3efe8", linecolor=SOFT_BORDER)
    return fig


def normalize_column_name(name: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(name).strip())
    cleaned = re.sub(r"[^\w_]", "", cleaned)
    return cleaned or "columna"


def detect_json_payload(text: str):
    match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def coerce_number(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    lowered = text.lower().replace("usd", "").replace("cop", "")
    multiplier = 1.0
    if "mil millones" in lowered:
        multiplier = 1_000_000_000
        lowered = lowered.replace("mil millones", "")
    elif "millones" in lowered:
        multiplier = 1_000_000
        lowered = lowered.replace("millones", "")
    elif "mil" in lowered and "%" not in lowered:
        multiplier = 1_000
        lowered = lowered.replace("mil", "")

    lowered = lowered.replace("$", "").replace("%", "").replace("por ciento", "")
    lowered = lowered.strip()
    lowered = re.sub(r"[^0-9,.\-]", "", lowered)
    if not lowered:
        return np.nan

    if "," in lowered and "." in lowered:
        if lowered.rfind(",") > lowered.rfind("."):
            lowered = lowered.replace(".", "").replace(",", ".")
        else:
            lowered = lowered.replace(",", "")
    elif lowered.count(",") == 1 and lowered.count(".") == 0:
        lowered = lowered.replace(",", ".")
    else:
        lowered = lowered.replace(",", "")

    try:
        return float(lowered) * multiplier
    except ValueError:
        return np.nan


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [normalize_column_name(col) for col in cleaned.columns]

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            numeric_try = cleaned[column].map(coerce_number)
            valid_ratio = numeric_try.notna().mean()
            if valid_ratio >= 0.7:
                cleaned[column] = numeric_try
            else:
                dt_try = pd.to_datetime(cleaned[column], errors="coerce")
                if dt_try.notna().mean() >= 0.7:
                    cleaned[column] = dt_try

    return cleaned


def local_extract_records(text: str):
    clauses = [
        chunk.strip(" .;:\n\t")
        for chunk in re.split(r"[\n]+|(?<=[.;])\s+", text)
        if re.search(r"\d", chunk)
    ]
    records = []
    value_pattern = re.compile(
        r"(?P<value>\$?\d[\d\.,]*)\s*(?P<unit>%|por ciento|millones|mil millones|mil|USD|COP|sacos|toneladas|tm|kg|ha|hectareas|casos|personas|empresas|puntos)?",
        flags=re.IGNORECASE,
    )
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    for clause in clauses:
        year_match = year_pattern.search(clause)
        year_value = int(year_match.group(0)) if year_match else None
        matches = list(value_pattern.finditer(clause))
        if not matches:
            continue

        prefix = clause[: matches[0].start()].strip(" ,:-")
        entity = prefix if prefix else "Registro"
        entity = re.sub(
            r"\b(en|durante|para|de|del|la|el)\b\s*$",
            "",
            entity,
            flags=re.IGNORECASE,
        ).strip()

        for index, match in enumerate(matches, start=1):
            value_raw = f"{match.group('value')} {match.group('unit') or ''}".strip()
            records.append(
                {
                    "registro": entity[:120] or f"Registro {index}",
                    "medida": f"valor_{index}",
                    "valor": coerce_number(value_raw),
                    "unidad": (match.group("unit") or "").strip(),
                    "anio": year_value,
                    "texto_origen": clause,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def extract_with_groq(text: str, client: Groq) -> pd.DataFrame:
    prompt = f"""
Convierte el texto en una tabla JSON util para analisis. Reglas:
- Responde solo JSON valido.
- Usa este formato exacto:
{{
  "dataset": [
    {{"columna_1": "...", "columna_2": 123, "columna_3": "..."}}
  ]
}}
- Cada fila debe representar una observacion.
- Separa entidad, periodo, metrica, valor y unidad cuando existan.
- Los numeros deben ir como numeros, no como texto.
- Si hay monedas o porcentajes, conserva una columna de unidad.
- No inventes filas ni valores.
- Si el texto no permite una tabla rica, devuelve una tabla minima con columnas:
  registro, valor, unidad, texto_origen.

Texto:
\"\"\"{text}\"\"\"
"""
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Eres un extractor de datos. Devuelves solo JSON valido y sin markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1800,
    )
    content = completion.choices[0].message.content or ""
    payload = detect_json_payload(content)
    if payload is None or "dataset" not in payload:
        raise ValueError("Groq no devolvio un JSON utilizable")
    return pd.DataFrame(payload["dataset"])


def extract_structured_table(text: str, extraction_mode: str, client: Groq | None):
    source_used = "Local"
    raw_df = pd.DataFrame()
    extraction_error = None

    if extraction_mode in {"Auto", "Groq"} and client is not None:
        try:
            raw_df = extract_with_groq(text, client)
            source_used = "Groq"
        except Exception as error:
            extraction_error = error
            if extraction_mode == "Groq":
                raise

    if raw_df.empty:
        raw_df = local_extract_records(text)
        source_used = "Local"

    if raw_df.empty:
        raise ValueError("No se pudieron extraer registros con el texto proporcionado.")

    return sanitize_dataframe(raw_df), source_used, extraction_error


def append_to_master_table(master_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if master_df.empty:
        return new_df.reset_index(drop=True)
    combined = pd.concat([master_df, new_df], ignore_index=True, sort=False)
    return sanitize_dataframe(combined)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return pd.DataFrame()
    summary = numeric_df.describe().T.reset_index().rename(columns={"index": "columna"})
    summary["faltantes"] = numeric_df.isna().sum().values
    return summary


def categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = df.select_dtypes(include=["object", "category"]).copy()
    rows = []
    for column in cat_df.columns:
        series = cat_df[column].dropna().astype(str)
        top = series.value_counts().head(5)
        rows.append(
            {
                "columna": column,
                "unicos": series.nunique(),
                "mas_frecuente": top.index[0] if not top.empty else "",
                "frecuencia": int(top.iloc[0]) if not top.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_eda(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Filas", f"{len(df):,}")
    k2.metric("Columnas", f"{df.shape[1]:,}")
    k3.metric("Numericas", f"{len(numeric_cols):,}")
    k4.metric("Categoricas/fecha", f"{len(cat_cols) + len(datetime_cols):,}")

    tabs = st.tabs(["Vista general", "Distribuciones", "Relaciones", "Calidad", "Datos"])

    with tabs[0]:
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("### Resumen numerico")
            profile = numeric_profile(df)
            if profile.empty:
                st.info("No hay columnas numericas suficientes para estadistica descriptiva.")
            else:
                st.dataframe(profile, use_container_width=True, hide_index=True)
        with right:
            st.markdown("### Resumen categorico")
            cat_profile = categorical_profile(df)
            if cat_profile.empty:
                st.info("No hay columnas categoricas para perfilar.")
            else:
                st.dataframe(cat_profile, use_container_width=True, hide_index=True)

        if numeric_cols:
            plot_col = numeric_cols[0]
            fig = px.box(
                df,
                y=plot_col,
                color_discrete_sequence=[DEEP],
                points="outliers",
                title=f"Dispersion de {plot_col}",
            )
            st.plotly_chart(chart_layout(fig, 380), use_container_width=True)

    with tabs[1]:
        if not numeric_cols:
            st.info("No hay columnas numericas para visualizar distribuciones.")
        else:
            selected_numeric = st.selectbox("Columna numerica", numeric_cols, key="dist_col")
            left, right = st.columns(2)
            with left:
                hist = px.histogram(
                    df,
                    x=selected_numeric,
                    nbins=20,
                    color_discrete_sequence=[TEAL],
                    title=f"Histograma de {selected_numeric}",
                )
                st.plotly_chart(chart_layout(hist, 380), use_container_width=True)
            with right:
                fig = px.violin(
                    df,
                    y=selected_numeric,
                    box=True,
                    color_discrete_sequence=[AMBER],
                    title=f"Violin plot de {selected_numeric}",
                )
                st.plotly_chart(chart_layout(fig, 380), use_container_width=True)

        if cat_cols and numeric_cols:
            st.markdown("### Corte por categoria")
            category_col = st.selectbox("Categoria", cat_cols, key="cat_cut")
            measure_col = st.selectbox("Medida", numeric_cols, key="measure_cut")
            grouped = (
                df.groupby(category_col, dropna=False)[measure_col]
                .mean()
                .reset_index()
                .sort_values(measure_col, ascending=False)
                .head(12)
            )
            bar = px.bar(
                grouped,
                x=measure_col,
                y=category_col,
                orientation="h",
                color=measure_col,
                color_continuous_scale=[MINT, DEEP],
                title=f"Promedio de {measure_col} por {category_col}",
            )
            st.plotly_chart(chart_layout(bar, 420), use_container_width=True)

    with tabs[2]:
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("Eje X", numeric_cols, key="rel_x")
            y_axis = st.selectbox(
                "Eje Y",
                [col for col in numeric_cols if col != x_axis] or numeric_cols,
                key="rel_y",
            )
            color_col = st.selectbox("Color", ["Ninguno"] + cat_cols, key="rel_color")
            scatter = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=None if color_col == "Ninguno" else color_col,
                trendline=None,
                color_discrete_sequence=[DEEP, AMBER, TEAL],
                title=f"Relacion entre {x_axis} y {y_axis}",
            )
            st.plotly_chart(chart_layout(scatter, 430), use_container_width=True)

            corr = df[numeric_cols].corr(numeric_only=True)
            heat = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale=[[0, "#f3efe4"], [0.5, "#7dc7ac"], [1, "#0f5c45"]],
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr.values, 2),
                    texttemplate="%{text}",
                )
            )
            heat.update_layout(title="Matriz de correlacion", margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(heat, use_container_width=True)
        else:
            st.info("Se necesitan al menos dos columnas numericas para analizar relaciones.")

    with tabs[3]:
        quality = pd.DataFrame(
            {
                "columna": df.columns,
                "tipo": [str(dtype) for dtype in df.dtypes],
                "faltantes": df.isna().sum().values,
                "faltantes_pct": (df.isna().mean().values * 100).round(2),
                "unicos": df.nunique(dropna=False).values,
            }
        )
        st.dataframe(quality, use_container_width=True, hide_index=True)
        if numeric_cols:
            outlier_rows = []
            for column in numeric_cols:
                series = df[column].dropna()
                if len(series) < 4:
                    continue
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    count = 0
                else:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    count = int(((series < lower) | (series > upper)).sum())
                outlier_rows.append({"columna": column, "outliers_iqr": count})
            if outlier_rows:
                st.markdown("### Posibles outliers")
                st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar CSV extraido",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tabla_extraida.csv",
            mime="text/csv",
        )


apply_theme()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Text to Table + EDA</div>
        <h1>Convierte un parrafo en datos analizables</h1>
        <p>
            Pega un texto con cifras, extrae una tabla estructurada y genera un EDA automatico.
            Si hay clave de Groq disponible, la app intenta construir una tabla mas limpia; si no,
            usa una extraccion local basada en patrones.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

client = get_groq_client()

if "master_table" not in st.session_state:
    st.session_state["master_table"] = pd.DataFrame()
if "entry_counter" not in st.session_state:
    st.session_state["entry_counter"] = 0
if "last_source_used" not in st.session_state:
    st.session_state["last_source_used"] = None

with st.sidebar:
    st.markdown("## Configuracion")
    extraction_mode = st.radio(
        "Metodo de extraccion",
        options=[
            "Auto",
            "Groq",
            "Local",
        ],
        index=0,
    )
    st.caption(
        "Auto usa Groq si detecta `GROQ_API_KEY`; en caso contrario usa un parser local."
    )
    st.divider()
    st.markdown("## Recomendacion")
    st.caption(
        "El texto funciona mejor cuando repite una estructura clara: entidad, periodo, metrica, valor y unidad."
    )
    st.divider()
    st.markdown("## Estado")
    st.caption("Groq disponible" if client else "Groq no configurado")
    current_rows = len(st.session_state["master_table"])
    st.caption(f"Filas acumuladas: {current_rows:,}")


default_text = st.session_state.get("input_text", SAMPLE_TEXT)
default_interview = st.session_state.get("interview_name", "Entrevista 1")
default_respondent = st.session_state.get("respondent_name", f"Persona {st.session_state['entry_counter'] + 1}")

meta_left, meta_right = st.columns(2)
with meta_left:
    interview_name = st.text_input("Tema o entrevista", value=default_interview)
with meta_right:
    respondent_name = st.text_input("Persona o fuente", value=default_respondent)

input_text = st.text_area(
    "Texto de entrada",
    value=default_text,
    height=220,
    placeholder="Ejemplo: En 2024, la region norte vendio 12500 unidades por 4.3 millones USD...",
)
st.session_state["input_text"] = input_text
st.session_state["interview_name"] = interview_name
st.session_state["respondent_name"] = respondent_name

sample_col, add_col, reset_col = st.columns([1, 2.4, 1.2])
with sample_col:
    if st.button("Usar ejemplo", use_container_width=True):
        st.session_state["input_text"] = SAMPLE_TEXT
        st.rerun()
with add_col:
    run_extraction = st.button("Agregar respuesta a la tabla", type="primary", use_container_width=True)
with reset_col:
    reset_interview = st.button("Reiniciar tabla", use_container_width=True)

if reset_interview:
    st.session_state["master_table"] = pd.DataFrame()
    st.session_state["entry_counter"] = 0
    st.session_state["last_source_used"] = None
    st.rerun()

if run_extraction:
    text = input_text.strip()
    if not text:
        st.warning("Ingresa un parrafo con cifras antes de ejecutar la extraccion.")
        st.stop()

    try:
        structured_df, source_used, extraction_error = extract_structured_table(
            text, extraction_mode, client
        )
    except Exception as error:
        if extraction_mode == "Groq":
            st.error("Groq no pudo estructurar el texto en una tabla utilizable.")
            st.caption(f"Detalle tecnico: {type(error).__name__}")
        else:
            st.error("No se pudieron extraer registros con el texto proporcionado.")
            st.caption(f"Detalle tecnico: {type(error).__name__}")
        st.stop()

    st.session_state["entry_counter"] += 1
    entry_id = st.session_state["entry_counter"]
    structured_df["entrevista"] = interview_name.strip() or "Entrevista"
    structured_df["persona"] = respondent_name.strip() or f"Persona {entry_id}"
    structured_df["entrada_id"] = entry_id
    structured_df["metodo_extraccion"] = source_used

    st.session_state["master_table"] = append_to_master_table(
        st.session_state["master_table"],
        structured_df,
    )
    st.session_state["last_source_used"] = source_used

    st.markdown('<div class="section-kicker">Resultado</div>', unsafe_allow_html=True)
    st.success(
        f"Se agregaron {len(structured_df):,} filas de {structured_df['persona'].iloc[0]} usando {source_used}."
    )

    if extraction_error is not None and source_used == "Local":
        st.caption(
            f"Groq no devolvio una estructura valida y se aplico el fallback local ({type(extraction_error).__name__})."
        )

master_df = st.session_state["master_table"]

if not master_df.empty:
    st.markdown('<div class="section-kicker">Dataset acumulado</div>', unsafe_allow_html=True)
    summary_left, summary_mid, summary_right = st.columns(3)
    summary_left.metric("Filas acumuladas", f"{len(master_df):,}")
    summary_mid.metric("Entradas procesadas", f"{master_df['entrada_id'].nunique():,}")
    summary_right.metric(
        "Personas capturadas",
        f"{master_df['persona'].nunique():,}" if "persona" in master_df.columns else "0",
    )
    build_eda(master_df)
else:
    st.info("La tabla acumulada esta vacia. Agrega la primera respuesta para construir el dataset de la entrevista.")
