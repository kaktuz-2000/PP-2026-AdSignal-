from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ad_recommender import (
    DB_PATH,
    FEATURE_COLUMNS,
    RAW_DIR,
    active_model_info,
    activate_model_version,
    analyze_feature_importance,
    data_fingerprint,
    explain_forecast,
    feature_dependence,
    get_experiment_log,
    get_experiment_replays,
    get_latest_model_metrics,
    get_model_registry,
    get_monitoring_log,
    get_quality_log,
    load_sources,
    monitor_model,
    predict_active_model,
    read_sources,
    read_feature_store,
    recommend_ads,
    replay_experiment,
    reset_demo_data,
    rollback_model,
    train_models,
    validate_sources,
)


ROOT = Path(__file__).resolve().parent
with (ROOT / "config.toml").open("rb") as config_file:
    CONFIG = tomllib.load(config_file)
DEMO_CONFIG = CONFIG["demo"]
MODEL_CONFIG = CONFIG.get("model", {})

st.set_page_config(page_title="AdSignal", page_icon="A", layout="wide")
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


@st.cache_data(show_spinner="Загружаю данные и проверяю модель...")
def get_dashboard_data(seed: int, days: int, search_budget: int, promotion_threshold: float):
    events, ads, loads = read_sources(seed=seed, days=days)
    model_info = active_model_info()
    if (
        model_info is None
        or model_info["pipeline_version"] < 3
        or get_latest_model_metrics().empty
    ):
        metrics, forecasts, best_model = train_models(
            events,
            seed=seed,
            search_budget=search_budget,
            promotion_threshold=promotion_threshold,
        )
    else:
        metrics = get_latest_model_metrics()
        forecasts, best_model = predict_active_model(events)
    importance = analyze_feature_importance(events)
    return events, ads, loads, metrics, forecasts, best_model, importance


def refresh_dashboard() -> None:
    st.cache_data.clear()
    st.session_state.pop("monitor_report", None)
    st.session_state.pop("retrain_recommended", None)
    st.session_state.pop("replay_result", None)
    st.rerun()


st.sidebar.markdown("## Данные и настройки")
st.sidebar.caption("Источники можно сгенерировать или заменить двумя CSV-файлами.")
if st.sidebar.button("Сбросить на демо-данные", use_container_width=True):
    try:
        reset_demo_data(seed=int(DEMO_CONFIG["seed"]), days=int(DEMO_CONFIG["days"]))
        refresh_dashboard()
    except Exception as error:
        st.sidebar.error(f"Не удалось обновить демо-данные: {error}")

with st.sidebar.expander("Смоделировать дрейф данных"):
    st.caption("Заменяет журнал на воспроизводимый сценарий со снижением CTR во второй половине периода.")
    if st.button("Создать сценарий дрейфа", use_container_width=True):
        try:
            reset_demo_data(
                seed=int(DEMO_CONFIG["seed"]) + 1,
                days=int(DEMO_CONFIG["days"]),
                drift=True,
            )
            refresh_dashboard()
        except Exception as error:
            st.error(f"Не удалось создать сценарий дрейфа: {error}")

with st.sidebar.expander("Загрузить собственные CSV"):
    with st.form("csv-upload-form"):
        event_file = st.file_uploader("Журнал событий", type=["csv"], key="events-upload")
        ads_file = st.file_uploader("Каталог объявлений", type=["csv"], key="ads-upload")
        replace_submitted = st.form_submit_button("Проверить и заменить источники")
    st.caption(
        "Замените текущие данные. Нужны колонки event_id, timestamp, user_id, "
        "segment, ad_id, category, viewed, clicked, liked, reposted; "
        "каталог: ad_id, category, title, advertiser."
    )
    if replace_submitted:
        if event_file is None or ads_file is None:
            st.error("Выберите оба CSV-файла.")
        else:
            try:
                uploaded_events = pd.read_csv(event_file)
                uploaded_ads = pd.read_csv(ads_file)
                quality_report = validate_sources(uploaded_events, uploaded_ads)
                st.dataframe(quality_report, hide_index=True, width="stretch")
                if (quality_report["Результат"] == "ERROR").any():
                    st.error("Исправьте ошибки качества в CSV и загрузите файлы повторно.")
                else:
                    load_sources(uploaded_events, uploaded_ads, replace=True)
                    st.success("Источники проверены и загружены. Пересчитайте модель в разделе «Модели».")
                    refresh_dashboard()
            except Exception as error:
                st.error(f"Не удалось обработать CSV: {error}")

try:
    with st.spinner("Подготавливаю прогноз и диагностику..."):
        (
            events,
            ads,
            loads,
            metrics,
            forecasts,
            best_model,
            feature_importance,
        ) = get_dashboard_data(
            int(DEMO_CONFIG["seed"]),
            int(DEMO_CONFIG["days"]),
            int(MODEL_CONFIG.get("search_budget", 4)),
            float(MODEL_CONFIG.get("promotion_threshold", 0.01)),
        )
except Exception as error:
    st.error(f"Не удалось подготовить аналитику: {error}")
    st.stop()

model_info = active_model_info()
if model_info is None:
    st.error("Нет активной модели. Откройте раздел «Модели» и запустите обучение.")
    st.stop()

current_data_hash_matches = model_info["data_hash"] == data_fingerprint(events)

users = sorted(events["user_id"].dropna().astype(str).unique())
segments = sorted(events["segment"].dropna().astype(str).unique())
categories = sorted(events["category"].dropna().astype(str).unique())
if not users or not segments or not categories:
    st.error("В источниках нет пользователей, сегментов или категорий для отображения.")
    st.stop()
selected_user = st.sidebar.selectbox("Демо-пользователь", users)
user_segment = str(events.loc[events["user_id"].astype(str) == selected_user, "segment"].mode().iloc[0])
selected_segment = st.sidebar.selectbox(
    "Сегмент аудитории", segments, index=segments.index(user_segment)
)
selected_category = st.sidebar.selectbox("Категория", categories)

st.markdown('<p class="eyebrow">Прогноз CTR · рекомендации · контроль моделей</p>', unsafe_allow_html=True)
st.title("AdSignal")
st.caption("Прогноз кликабельности рекламы и подбор объявлений по интересам аудитории.")
if not current_data_hash_matches:
    st.warning(
        "Источники изменились после обучения активной версии. Текущие прогнозы рассчитаны старой моделью; "
        "сначала запустите мониторинг, затем при необходимости — переобучение."
    )

selected_forecast = forecasts.loc[
    (forecasts["segment"] == selected_segment)
    & (forecasts["category"] == selected_category)
]
history_events = events.loc[
    (events["segment"].astype(str) == selected_segment)
    & (events["category"].astype(str) == selected_category)
].copy()
history_events["date"] = pd.to_datetime(history_events["timestamp"]).dt.floor("D")
history = (
    history_events.groupby("date", as_index=False)
    .agg(impressions=("viewed", "sum"), clicks=("clicked", "sum"))
)
history["CTR"] = history["clicks"] / history["impressions"].clip(lower=1)
if selected_forecast.empty:
    next_day_ctr = 0.0
    next_day = pd.Timestamp(history["date"].max()) + pd.Timedelta(days=1)
else:
    next_day_ctr = float(selected_forecast.iloc[0]["forecast_ctr"])
    next_day = pd.Timestamp(selected_forecast.iloc[0]["forecast_date"])

overall_ctr = events["clicked"].sum() / max(events["viewed"].sum(), 1)
metric_columns = st.columns(4)
metric_columns[0].metric("События", f"{len(events):,}".replace(",", " "))
metric_columns[1].metric("CTR за период", f"{overall_ctr:.1%}")
metric_columns[2].metric("Прогноз CTR", f"{next_day_ctr:.1%}")
metric_columns[3].metric("Активная модель", best_model)

overview, model_tab, data_tab = st.tabs(["Обзор", "Модели и контроль", "Данные"])
with overview:
    chart_history = history.tail(30).copy()
    chart_history["Тип"] = "Факт"
    future = pd.DataFrame(
        {"date": [next_day], "CTR": [next_day_ctr], "Тип": ["Прогноз"]}
    )
    response_chart = px.line(
        chart_history,
        x="date",
        y="CTR",
        markers=True,
        title=f"Факт и прогноз · {selected_segment} / {selected_category}",
    )
    response_chart.add_trace(
        go.Scatter(
            x=future["date"],
            y=future["CTR"],
            mode="markers",
            name="Прогноз",
            marker={"size": 12, "color": "#cb713d", "symbol": "diamond"},
        )
    )
    response_chart.update_traces(line_color="#28715d", selector={"mode": "lines+markers"})
    response_chart.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".0%",
        xaxis_title=None,
        yaxis_title="CTR",
    )
    st.plotly_chart(response_chart, width="stretch")

    st.subheader("Прогноз по категориям")
    segment_forecasts = forecasts.loc[forecasts["segment"] == selected_segment].sort_values(
        "expected_clicks_per_1000"
    )
    forecast_chart = px.bar(
        segment_forecasts,
        x="expected_clicks_per_1000",
        y="category",
        orientation="h",
        labels={"expected_clicks_per_1000": "Ожидаемые клики на 1000 показов", "category": "Категория"},
    )
    forecast_chart.update_traces(marker_color="#cb713d")
    forecast_chart.update_layout(height=300, margin=dict(l=8, r=8, t=16, b=8))
    st.plotly_chart(forecast_chart, width="stretch")

    st.subheader(f"Подбор рекламы · {selected_user}")
    recommendations = recommend_ads(selected_user, events, ads, forecasts).head(5)
    st.dataframe(
        recommendations[["title", "advertiser", "category", "forecast_ctr", "forecast_clicks_per_1000", "score"]]
        .rename(
            columns={
                "title": "Объявление",
                "advertiser": "Рекламодатель",
                "category": "Категория",
                "forecast_ctr": "Прогноз CTR",
                "forecast_clicks_per_1000": "Клики / 1000",
                "score": "Рейтинг",
            }
        )
        .style.format({"Прогноз CTR": "{:.1%}", "Клики / 1000": "{:.1f}", "Рейтинг": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )

with model_tab:
    st.subheader("Сравнение моделей на holdout")
    st.caption(
        f"Конфигурация выбирается по validation, holdout используется только для оценки. "
        f"Активная версия {model_info['version_id']}; отметка показывает validation-победителя."
    )
    st.dataframe(
        metrics.style.format(
            {
                "MAE": "{:.4f}",
                "RMSE": "{:.4f}",
                "Validation MAE": "{:.4f}",
                "Клики на 1000 показов": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "MAE/RMSE — ошибка на отложенном временном отрезке; клики на 1000 показов — "
        "понятное прикладное выражение среднего прогноза."
    )
    if not current_data_hash_matches:
        st.info("Сравнительные метрики относятся к последнему запуску обучения; активная версия создана на другом наборе данных.")

    st.subheader("Объяснение отдельного прогноза")
    try:
        event_dates = pd.to_datetime(events["timestamp"]).dt.floor("D")
        forecast_date = st.date_input(
            "Дата для объяснения",
            value=(event_dates.max() + pd.Timedelta(days=1)).date(),
            min_value=(event_dates.min() + pd.Timedelta(days=1)).date(),
            max_value=(event_dates.max() + pd.Timedelta(days=1)).date(),
        )
        explain_ctr, explanation = explain_forecast(
            events, selected_segment, selected_category, forecast_date
        )
        st.metric(
            f"Прогноз CTR · {selected_segment} / {selected_category}",
            f"{explain_ctr:.2%}",
        )
        st.caption(
            "Столбцы показывают изменение прогноза при замене одного признака на типичное значение. "
            "Это локальные независимые эффекты возмущения, не аддитивное разложение."
        )
        explanation_chart = px.bar(
            explanation,
            x="Изменение прогноза, п.п.",
            y="Признак",
            orientation="h",
            color="Изменение прогноза, п.п.",
            color_continuous_scale=["#cb713d", "#e7e8dd", "#28715d"],
        )
        explanation_chart.update_layout(height=320)
        st.plotly_chart(explanation_chart, width="stretch")
        st.dataframe(
            explanation.style.format({"Изменение прогноза, п.п.": "{:+.3f}"}),
            hide_index=True,
            width="stretch",
        )
        dependence_feature = st.selectbox(
            "Признак для анализа зависимости",
            FEATURE_COLUMNS,
            key="dependence-feature",
        )
        dependence = feature_dependence(
            events,
            selected_segment,
            selected_category,
            dependence_feature,
            forecast_date,
        )
        dependence_chart = px.line(
            dependence,
            x="Значение",
            y="Прогноз CTR",
            markers=True,
            title=f"Зависимость прогноза от {dependence_feature}",
        )
        dependence_chart.update_layout(
            yaxis_tickformat=".0%",
            xaxis_title=dependence_feature,
            yaxis_title="Прогноз CTR",
            height=320,
        )
        st.plotly_chart(dependence_chart, width="stretch")
    except Exception as error:
        st.error(f"Не удалось построить объяснение прогноза: {error}")

    st.subheader("Важность признаков во времени")
    st.caption("Сравниваются два независимых метода: падение качества при перемешивании признака и взаимная информация.")
    st.dataframe(feature_importance, hide_index=True, width="stretch")
    importance_metric = (
        "Permutation importance (ΔMAE)"
        if "Permutation importance (ΔMAE)" in feature_importance
        else None
    )
    if importance_metric and not feature_importance.empty:
        importance_chart = px.bar(
            feature_importance,
            x=importance_metric,
            y="Признак",
            color="Период",
            barmode="group",
            orientation="h",
        )
        importance_chart.update_layout(height=360)
        st.plotly_chart(importance_chart, width="stretch")

    st.subheader("Мониторинг дрейфа и качества")
    st.caption(
        "Правило: предложить переобучение, если PSI любого входного признака ≥ 0.20 "
        "или MAE недавнего окна более чем на 25% превышает holdout MAE активной версии."
    )
    if st.button("Проверить дрейф и качество", key="monitor-model"):
        try:
            st.session_state["monitor_report"], st.session_state["retrain_recommended"] = monitor_model(events)
        except Exception as error:
            st.error(f"Не удалось выполнить мониторинг: {error}")
    if "monitor_report" in st.session_state:
        st.dataframe(
            st.session_state["monitor_report"].style.format(
                {"Значение": "{:.4f}", "Порог": "{:.4f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        if st.session_state["retrain_recommended"]:
            st.warning("Найден сигнал дрейфа/ухудшения. Рекомендуется ограниченное переобучение.")
        else:
            st.success("Пороговые условия переобучения не сработали.")
    if st.button("Запустить ограниченный поиск и переобучение", key="retrain-model"):
        try:
            with st.spinner("Сравниваю ограниченный набор конфигураций на временном validation..."):
                train_models(
                    events,
                    seed=int(DEMO_CONFIG["seed"]),
                    search_budget=int(MODEL_CONFIG.get("search_budget", 4)),
                    promotion_threshold=float(MODEL_CONFIG.get("promotion_threshold", 0.01)),
                )
            st.success("Эксперименты сохранены; версия продвинута только при выполнении правила сравнения.")
            st.session_state.pop("monitor_report", None)
            st.session_state.pop("retrain_recommended", None)
            refresh_dashboard()
        except Exception as error:
            st.error(f"Переобучение не выполнено: {error}")

    st.subheader("Версии моделей")
    registry = get_model_registry()
    st.dataframe(registry, hide_index=True, width="stretch")
    inactive_versions = registry.loc[registry["Статус"] != "active", "Версия"].astype(str).tolist() if not registry.empty else []
    if inactive_versions:
        version_to_activate = st.selectbox("Версия для активации", inactive_versions)
        if st.button("Активировать выбранную версию"):
            try:
                activate_model_version(version_to_activate)
                refresh_dashboard()
            except Exception as error:
                st.error(f"Не удалось активировать версию: {error}")
        if st.button("Откатить на предыдущую версию"):
            try:
                rollback_model()
                refresh_dashboard()
            except Exception as error:
                st.error(f"Не удалось выполнить откат: {error}")

    with st.expander("Журнал экспериментов"):
        experiments = get_experiment_log()
        st.dataframe(experiments, hide_index=True, width="stretch")
        if not experiments.empty:
            experiment_id = st.selectbox(
                "Эксперимент для воспроизведения",
                experiments["Эксперимент"].astype(str).tolist(),
            )
            if st.button("Воспроизвести выбранный эксперимент"):
                try:
                    st.session_state["replay_result"] = replay_experiment(experiment_id)
                except Exception as error:
                    st.error(f"Не удалось воспроизвести эксперимент: {error}")
        if "replay_result" in st.session_state:
            st.dataframe(
                st.session_state["replay_result"].style.format(
                    {"MAE": "{:.5f}", "RMSE": "{:.5f}"}
                ),
                hide_index=True,
                width="stretch",
            )
        st.dataframe(get_experiment_replays(), hide_index=True, width="stretch")
    with st.expander("Журнал мониторинга"):
        st.dataframe(get_monitoring_log(), hide_index=True, width="stretch")

with data_tab:
    st.subheader("Качество источников")
    st.dataframe(get_quality_log(), hide_index=True, width="stretch")
    st.subheader("Витрина признаков")
    feature_store = read_feature_store(data_fingerprint(events))
    if feature_store.empty:
        st.info("Витрины для текущего набора пока нет. Запустите переобучение в разделе «Модели».")
    else:
        st.caption(
            f"Сохранено строк признаков: {len(feature_store):,}. "
            "Последний день не имеет целевой метки и используется для прогноза."
        )
        st.dataframe(feature_store.head(200), hide_index=True, width="stretch")
    st.subheader("Журнал загрузок")
    st.dataframe(loads, hide_index=True, width="stretch")
    st.caption(
        f"Последняя дата событий: {pd.to_datetime(events['timestamp']).max().date()}. "
        f"SQLite: {DB_PATH.name}; CSV: {RAW_DIR.relative_to(ROOT)}. "
        f"Модель обучена на текущем наборе: {'да' if current_data_hash_matches else 'нет'}."
    )
