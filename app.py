from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ad_recommender import (
    DB_PATH,
    RAW_DIR,
    create_sources,
    load_sources,
    read_sources,
    recommend_ads,
    train_models,
)


ROOT = Path(__file__).resolve().parent
with (ROOT / "config.toml").open("rb") as config_file:
    CONFIG = tomllib.load(config_file)["demo"]

st.set_page_config(page_title="Ad Signals", page_icon="A", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f4f5ef; }
    [data-testid="stSidebar"] { background: #e6ede1; }
    h1, h2, h3 { color: #173c32; font-family: Georgia, serif; }
    [data-testid="stMetric"] { background: #fff; border-left: 3px solid #cb713d; padding: 12px 16px; }
    div[data-testid="stDataFrame"] { border: 1px solid #d8ded5; }
    .eyebrow { color: #597067; text-transform: uppercase; font-size: 0.76rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Подготавливаю данные и обучаю модели...")
def get_dashboard_data(seed: int, days: int):
    events, ads, loads = read_sources(seed=seed, days=days)
    metrics, forecasts, best_model = train_models(events)
    return events, ads, loads, metrics, forecasts, best_model


st.sidebar.markdown("## Настройки")
if st.sidebar.button("Обновить синтетические данные", use_container_width=True):
    new_events, new_ads = create_sources(seed=CONFIG["seed"], days=CONFIG["days"])
    load_sources(new_events, new_ads)
    st.cache_data.clear()
    st.rerun()

try:
    events, ads, loads, metrics, forecasts, best_model = get_dashboard_data(
        int(CONFIG["seed"]), int(CONFIG["days"])
    )
except Exception as error:
    st.error(f"Не удалось подготовить аналитику: {error}")
    st.stop()

users = sorted(events["user_id"].dropna().unique())
selected_user = st.sidebar.selectbox("Демо-пользователь", users)
selected_segment = str(
    events.loc[events["user_id"] == selected_user, "segment"].mode().iloc[0]
)

st.markdown('<p class="eyebrow">Медиа · прогноз CTR · демо на синтетических данных</p>', unsafe_allow_html=True)
st.title("Ad Signals")
st.caption("Поведение аудитории превращается в прогноз отклика и подбор объявлений по интересам.")

latest_date = pd.to_datetime(events["timestamp"]).max().date()
overall_ctr = events["clicked"].sum() / max(events["viewed"].sum(), 1)
forecast_rows = forecasts.loc[forecasts["segment"] == selected_segment]
forecast_ctr = forecast_rows["forecast_ctr"].mean()
top_category = (
    recommend_ads(selected_user, events, ads, forecasts).iloc[0]["category"]
)

metric_columns = st.columns(4)
metric_columns[0].metric("События", f"{len(events):,}".replace(",", " "))
metric_columns[1].metric("CTR за период", f"{overall_ctr:.1%}")
metric_columns[2].metric("Прогноз CTR", f"{forecast_ctr:.1%}")
metric_columns[3].metric("Топ интереса", str(top_category).title())

left, right = st.columns([1.25, 1])
with left:
    st.subheader("Отклик аудитории")
    user_events = events.loc[events["segment"] == selected_segment].copy()
    user_events["date"] = pd.to_datetime(user_events["timestamp"]).dt.floor("D")
    history = (
        user_events.groupby("date", as_index=False)
        .agg(impressions=("viewed", "sum"), clicks=("clicked", "sum"))
    )
    history["CTR"] = history["clicks"] / history["impressions"].clip(lower=1)
    history = history.tail(30)
    chart = px.line(history, x="date", y="CTR", markers=True)
    chart.update_traces(line_color="#28715d", marker_color="#cb713d")
    chart.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%",
        xaxis_title=None,
        yaxis_title="CTR",
    )
    st.plotly_chart(chart, width="stretch")
with right:
    st.subheader("Прогноз на следующий день")
    forecast_chart = forecast_rows.sort_values("expected_clicks_per_1000", ascending=True)
    bars = px.bar(
        forecast_chart,
        x="expected_clicks_per_1000",
        y="category",
        orientation="h",
        labels={"expected_clicks_per_1000": "Кликов на 1000 показов", "category": "Категория"},
    )
    bars.update_traces(marker_color="#cb713d")
    bars.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(bars, width="stretch")

st.subheader(f"Подбор рекламы · {selected_user}")
st.caption("Ранжирование учитывает прогноз CTR сегмента и историю просмотров, кликов, лайков и репостов.")
recommendations = recommend_ads(selected_user, events, ads, forecasts).head(5)
for _, ad in recommendations.iterrows():
    ad_left, ad_right = st.columns([3, 1])
    ad_left.markdown(f"**{ad['title']}** · {ad['advertiser']}  \n{ad['category'].title()}")
    ad_right.metric("Ожидаемые клики / 1000", f"{ad['forecast_clicks_per_1000']:.1f}")

st.subheader("Сравнение моделей")
st.caption(f"Лучшая по MAE модель для выдачи: {best_model}. Проверка выполнена на более позднем временном отрезке.")
st.dataframe(
    metrics.style.format(
        {"MAE": "{:.4f}", "RMSE": "{:.4f}", "Клики на 1000 показов": "{:.1f}"}
    ),
    hide_index=True,
    width="stretch",
)

with st.expander("История загрузок источников"):
    st.dataframe(loads, hide_index=True, width="stretch")
    st.caption(
        f"Последняя дата событий: {latest_date}. Источники: журнал взаимодействий и каталог объявлений. "
        f"Хранилище: SQLite ({DB_PATH.name}); исходные CSV: {RAW_DIR.relative_to(ROOT)}."
    )