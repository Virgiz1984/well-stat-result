import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Интерактивная визуализация результатов интерпретации ГИС", layout="wide")
st.title("Интерактивная визуализация результатов интерпретации ГИС")

MIN_PER_CATEGORY = 1  # минимум сырых наблюдений на категорию (до агрегации)

@st.cache_data
def load_data():
    df = pd.read_excel("interact_best_well.xlsx", index_col=None)
    return df

def first_existing(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

df = load_data()

# Приведение числовых столбцов (используемых в флагах/фильтрах)
for col in ["frac_rf", "dis_frac_rfn", "flow_frac_test_rf", "kpr_rf", "perm_timur_ad", "SAT_rf", "por_rf", "kvo_rf"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Инженерные флаги (трещиноватость с порогом 0.15 и гейтом по flow_frac_test_rf)
df["COLL_poro_type"] = np.where(
    (df.get("kpr_rf", 0) > 0.1) | (df.get("perm_timur_ad", 0) > 0.1), 1, 0
)
df["COLL_frac_type"] = np.where(
    (df.get("frac_rf", 0) > 0.15) | (df.get("dis_frac_rfn", 0) > 0.15), 1, 0
)
if "flow_frac_test_rf" in df.columns:
    df["COLL_frac_type"] = np.where(df["flow_frac_test_rf"] < 0.5, 0, df["COLL_frac_type"])

# Тип коллектора (включая неколлекторы)
df["COLL_TYPE"] = np.where(df["COLL_poro_type"] == 1, "порово-каверновый", "неколлектор")
df["COLL_TYPE"] = np.where(df["COLL_frac_type"] == 1, "трещинный", df["COLL_TYPE"])
df["COLL_TYPE"] = np.where((df["COLL_frac_type"] == 1) & (df["COLL_poro_type"] == 1), "каверново-трещинный", df["COLL_TYPE"])

# Удобная колонка литологии
df["Litologiya"] = df["Литология по ГИС"]

st.write(f"Строк: {len(df):,}. Колонки: {len(df.columns)}")

# Преднастройки
well_col = first_existing(df, ["well"], default=None)
lith_candidates = ["Litologiya", "Литология по ГИС", "PETRO", "group", "Кластеры ГИС", "coll_type", "fr_type", "NK"]
poro_candidates = ["por_rf", "Prob_poro_res", "porosity", "PHI", "phi", "por"]
reservoir_group_candidates = ["Litologiya", "Литология по ГИС", "PETRO", "group", "Кластеры ГИС", "coll_type", "fr_type", "NK"]
perm_candidates = ["kpr_rf", "perm_timur_ad", "permeability", "perm", "kpr"]

# Сайдбар
with st.sidebar:
    st.header("Настройки")
    if well_col is None:
        well_col = st.selectbox("Колонка скважины", options=df.columns)

    lith_options = [c for c in lith_candidates if c in df.columns] or list(df.columns)
    poro_options = [c for c in poro_candidates if c in df.columns] or list(df.columns)
    reservoir_options = [c for c in reservoir_group_candidates if c in df.columns] or list(df.columns)
    perm_options = [c for c in perm_candidates if c in df.columns] or list(df.columns)

    lith_col = st.selectbox("Колонка литологии", options=lith_options, index=0)
    poro_col = st.selectbox("Колонка пористости", options=poro_options, index=0)
    perm_col = st.selectbox("Колонка проницаемости", options=perm_options, index=0)
    reservoir_col = st.selectbox("Колонка группировки коллекторов (пласты/группы)", options=reservoir_options, index=0)

    st.markdown("---")
    st.subheader("Фильтр по скважинам")
    wells = sorted(df[well_col].dropna().unique().tolist())
    selected_wells = st.multiselect("Скважины", options=wells)
    df_view = df[df[well_col].isin(selected_wells)] if selected_wells else df

    st.markdown("---")
    st.subheader("Фильтр по SAT_rf (вероятность УВ)")
    if "SAT_rf" in df_view.columns:
        sat_series = pd.to_numeric(df_view["SAT_rf"], errors="coerce")
        sat_norm = sat_series / 100.0 if sat_series.dropna().median() > 1.0 else sat_series
        sat_min, sat_max = st.slider("Диапазон SAT_rf", 0.0, 1.0, (0.0, 1.0), 0.01)
        df_view = df_view[sat_norm.between(sat_min, sat_max)]
    else:
        st.caption("Колонка 'SAT_rf' не найдена.")

    st.markdown("---")
    st.subheader("Фильтр коллектора")
    available_flag_cols = [c for c in ["COLL_poro_type", "COLL_frac_type"] if c in df_view.columns]
    default_mode = "flag" if available_flag_cols else "category"
    mode_label = {"flag": "Флаг (COLL*/инженерные)", "category": "Категория"}
    collector_mode = st.radio("Режим", ["flag", "category"],
                              index=0 if default_mode == "flag" else 1,
                              format_func=lambda x: mode_label[x])
    collector_mask = pd.Series(True, index=df_view.index)

    if collector_mode == "flag" and not available_flag_cols:
        st.warning("Флаговых колонок не найдено. Переключаюсь на режим 'Категория'.")
        collector_mode = "category"

    if collector_mode == "flag":
        flag_col = st.selectbox("Флаговая колонка коллектора", options=available_flag_cols)
        coll_vals = df_view[flag_col]
        if is_numeric_dtype(coll_vals):
            coll_threshold = st.number_input("Порог для 'коллектор' (числовой флаг)", value=1.0, step=0.5)
            collector_mask = coll_vals.fillna(0) >= coll_threshold
        else:
            uniq = sorted(coll_vals.dropna().astype(str).unique().tolist())
            default_pos = [v for v in uniq if "колл" in v.lower() or "pay" in v.lower() or v.lower() in {"yes", "true", "1"}]
            chosen = st.multiselect("Значения, считающиеся коллектором", options=uniq, default=default_pos or uniq[:1])
            collector_mask = coll_vals.astype(str).isin(chosen)

    if collector_mode == "category":
        cat_candidates = [reservoir_col] + ([c for c in ["coll_type"] if c in df_view.columns and c != reservoir_col])
        cat_col = st.selectbox("Колонка-категория для выделения коллекторов", options=cat_candidates)
        uniq = sorted(df_view[cat_col].dropna().astype(str).unique().tolist())
        default_pos = [v for v in uniq if "колл" in v.lower() or "прод" in v.lower() or "pay" in v.lower()]
        chosen = st.multiselect("Какие значения считать коллекторами", options=uniq, default=default_pos or uniq)
        collector_mask = df_view[cat_col].astype(str).isin(chosen)

    st.markdown("---")
    st.subheader("Вес для боксплотов")
    weight_mode = st.radio(
        "Режим веса",
        ["equal", "weighted"],
        index=0,
        format_func=lambda v: "Равный вес скважин" if v == "equal" else "Вес ∝ числу сырых точек"
    )
    max_rep = st.slider("Максимум повторов на скважину (для взвешивания)", 10, 500, 100) if weight_mode == "weighted" else 0

# Применим фильтры (коллекторный фильтр НЕ влияет на 1-ю круговую COLL_TYPE)
res_df = df_view[collector_mask].copy()
st.caption(f"После фильтра по скважинам/SAT: {len(df_view):,} строк. Коллекторы: {len(res_df):,} строк")

# Числовые поля (с учётом выбранной проницаемости)
for col in [poro_col, perm_col, "dis_frac_rfn", "SAT_rf", "kvo_rf"]:
    if col in res_df.columns:
        res_df[col] = pd.to_numeric(res_df[col], errors="coerce")

# Масштаб долей (если похоже на %)
for col in [poro_col, "SAT_rf", "kvo_rf"]:
    if col in res_df.columns:
        sample = res_df[col].dropna()
        if not sample.empty and sample.median() > 1.0:
            res_df[col] = res_df[col] / 100.0

phi_unit = sat_unit = kvo_unit = "доля"

# 1) Круговая №1: COLL_TYPE — вес = число сырых точек; подписи: h = сумма/10; не зависит от фильтра коллектора
pie_colltype_src = df_view.dropna(subset=["COLL_TYPE"])
pie_colltype = (
    pie_colltype_src
    .groupby("COLL_TYPE", as_index=False)
    .size()
    .rename(columns={"size": "n_points"})
    .sort_values("n_points", ascending=False)
)
pie_colltype["thickness_m"] = pie_colltype["n_points"] / 10.0
pie_colltype["weight"] = pie_colltype["n_points"]

# 2) Круговая №2: Литологии — вес = число сырых точек ПОСЛЕ коллекторного фильтра
pie_lith = (
    res_df.dropna(subset=[lith_col])
          .groupby(lith_col, as_index=False)
          .size()
          .rename(columns={"size": "weight"})
          .sort_values("weight", ascending=False)
)

# Хелперы
def allowed_reservoirs_by_raw_counts(df_in: pd.DataFrame, value_col: str) -> pd.Index:
    if value_col not in df_in.columns:
        return pd.Index([])
    counts = (
        df_in.dropna(subset=[reservoir_col, value_col])
          .groupby(reservoir_col)[value_col]
          .count()
    )
    return counts[counts >= MIN_PER_CATEGORY].index

def make_mean_with_count(df_in: pd.DataFrame, value_col: str, value_name: str):
    if value_col not in df_in.columns:
        return pd.DataFrame(columns=[reservoir_col, well_col, value_name, f"{value_name}_n"])
    agg = (
        df_in.dropna(subset=[reservoir_col, well_col, value_col])
        .groupby([reservoir_col, well_col], as_index=False)
        .agg(**{value_name: (value_col, "mean"), f"{value_name}_n": (value_col, "count")})
    )
    return agg

def prepare_box_df(plot_df: pd.DataFrame, group_col: str, y_col: str, label_col_name: str):
    if plot_df.empty:
        return plot_df
    n_cat = plot_df.groupby(group_col)[y_col].count()
    plot_df = plot_df.copy()
    plot_df["N_cat"] = plot_df[group_col].map(n_cat)
    label_map = {res: f"{res} (n={int(n_cat.get(res, 0))})" for res in n_cat.index}
    plot_df[label_col_name] = plot_df[group_col].map(label_map).astype(str)
    return plot_df

def build_weighted_source(df_in: pd.DataFrame, count_col: str, max_rep: int) -> pd.DataFrame:
    w = df_in[count_col].fillna(1).astype(float)
    max_w = w.max()
    if not np.isfinite(max_w) or max_w <= 0:
        reps = pd.Series(1, index=df_in.index)
    else:
        scale = max_rep / max_w
        reps = np.maximum(1, np.floor(w * scale).astype(int))
    return df_in.loc[df_in.index.repeat(reps)].copy()

# Режим построения
single_well_mode = bool(selected_wells) and len(selected_wells) == 1

# Пористость
phi_allowed = allowed_reservoirs_by_raw_counts(res_df, poro_col)
if single_well_mode:
    phi_plot = res_df.dropna(subset=[reservoir_col, poro_col]).copy()
    phi_plot = phi_plot[phi_plot[reservoir_col].isin(phi_allowed)]
    phi_plot["phi_value"] = phi_plot[poro_col]
    phi_plot = prepare_box_df(phi_plot, reservoir_col, "phi_value", "res_label_phi")
    phi_y = "phi_value"
else:
    mean_phi = make_mean_with_count(res_df, poro_col, "phi_mean")
    phi_plot = mean_phi[mean_phi[reservoir_col].isin(phi_allowed)].copy()
    if weight_mode == "weighted":
        phi_plot = build_weighted_source(phi_plot, "phi_mean_n", max_rep)
    phi_plot = prepare_box_df(phi_plot, reservoir_col, "phi_mean", "res_label_phi")
    phi_y = "phi_mean"

# Вероятность трещин
dis_allowed = allowed_reservoirs_by_raw_counts(res_df, "dis_frac_rfn")
if single_well_mode:
    dis_plot = res_df.dropna(subset=[reservoir_col, "dis_frac_rfn"]).copy()
    dis_plot = dis_plot[dis_plot[reservoir_col].isin(dis_allowed)]
    dis_plot["dis_value"] = dis_plot["dis_frac_rfn"]
    dis_plot = prepare_box_df(dis_plot, reservoir_col, "dis_value", "res_label_disfrac")
    dis_y = "dis_value"
else:
    mean_disfrac = make_mean_with_count(res_df, "dis_frac_rfn", "dis_frac_mean")
    dis_plot = mean_disfrac[mean_disfrac[reservoir_col].isin(dis_allowed)].copy()
    if weight_mode == "weighted":
        dis_plot = build_weighted_source(dis_plot, "dis_frac_mean_n", max_rep)
    dis_plot = prepare_box_df(dis_plot, reservoir_col, "dis_frac_mean", "res_label_disfrac")
    dis_y = "dis_frac_mean"

# Проницаемость (лог) — используем выбранную колонку perm_col
kpr_allowed = allowed_reservoirs_by_raw_counts(res_df, perm_col)
if single_well_mode:
    kpr_plot = res_df.dropna(subset=[reservoir_col, perm_col]).copy()
    kpr_plot = kpr_plot[kpr_plot[reservoir_col].isin(kpr_allowed)]
    kpr_plot = kpr_plot[kpr_plot[perm_col] > 0]
    kpr_plot["kpr_value"] = kpr_plot[perm_col]
    kpr_plot = prepare_box_df(kpr_plot, reservoir_col, "kpr_value", "res_label_kpr")
    kpr_y = "kpr_value"
else:
    mean_kpr = make_mean_with_count(res_df, perm_col, "kpr_mean")
    kpr_plot = mean_kpr[mean_kpr[reservoir_col].isin(kpr_allowed)].copy()
    kpr_plot = kpr_plot[kpr_plot["kpr_mean"] > 0]
    if weight_mode == "weighted":
        kpr_plot = build_weighted_source(kpr_plot, "kpr_mean_n", max_rep)
    kpr_plot = prepare_box_df(kpr_plot, reservoir_col, "kpr_mean", "res_label_kpr")
    kpr_y = "kpr_mean"

# Вероятность получения УВ
sat_allowed = allowed_reservoirs_by_raw_counts(res_df, "SAT_rf")
if single_well_mode:
    sat_plot = res_df.dropna(subset=[reservoir_col, "SAT_rf"]).copy()
    sat_plot = sat_plot[sat_plot[reservoir_col].isin(sat_allowed)]
    sat_plot["sat_value"] = sat_plot["SAT_rf"]
    sat_plot = prepare_box_df(sat_plot, reservoir_col, "sat_value", "res_label_sat")
    sat_y = "sat_value"
else:
    mean_sat = make_mean_with_count(res_df, "SAT_rf", "sat_mean")
    sat_plot = mean_sat[mean_sat[reservoir_col].isin(sat_allowed)].copy()
    if weight_mode == "weighted":
        sat_plot = build_weighted_source(sat_plot, "sat_mean_n", max_rep)
    sat_plot = prepare_box_df(sat_plot, reservoir_col, "sat_mean", "res_label_sat")
    sat_y = "sat_mean"

# Водоудерживающая способность
kvo_allowed = allowed_reservoirs_by_raw_counts(res_df, "kvo_rf")
if single_well_mode:
    kvo_plot = res_df.dropna(subset=[reservoir_col, "kvo_rf"]).copy()
    kvo_plot = kvo_plot[kvo_plot[reservoir_col].isin(kvo_allowed)]
    kvo_plot["kvo_value"] = kvo_plot["kvo_rf"]
    kvo_plot = prepare_box_df(kvo_plot, reservoir_col, "kvo_value", "res_label_kvo")
    kvo_y = "kvo_value"
else:
    mean_kvo = make_mean_with_count(res_df, "kvo_rf", "kvo_mean")
    kvo_plot = mean_kvo[mean_kvo[reservoir_col].isin(kvo_allowed)].copy()
    if weight_mode == "weighted":
        kvo_plot = build_weighted_source(kvo_plot, "kvo_mean_n", max_rep)
    kvo_plot = prepare_box_df(kvo_plot, reservoir_col, "kvo_mean", "res_label_kvo")
    kvo_y = "kvo_mean"

# Ряд 1: две круговые — COLL_TYPE (без коллекторного фильтра) и Литологии (с коллекторным фильтром)
c0, c1 = st.columns(2)
with c0:
    st.subheader("Распределение типов коллекторов (COLL_TYPE)")
    if len(pie_colltype) == 0:
        st.info("Нет данных для COLL_TYPE.")
    else:
        fig0 = px.pie(pie_colltype, names="COLL_TYPE", values="weight", hole=0.25)
        fig0.update_traces(
            textposition="inside",
            texttemplate="%{label}<br>h=%{customdata[0]:.1f} м",
            customdata=np.stack([pie_colltype["thickness_m"]], axis=-1)
        )
        st.plotly_chart(fig0, use_container_width=True)

with c1:
    st.subheader("Распределение литологий в коллекторах")
    if len(pie_lith) == 0:
        st.info("Нет данных для литологий.")
    else:
        fig1 = px.pie(pie_lith, names=lith_col, values="weight", hole=0.25)
        fig1.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig1, use_container_width=True)

# Ряд 2: пористость + трещиноватость
col2, col3 = st.columns(2)
with col2:
    st.subheader(f"Боксплот: пористость ({'сырые' if single_well_mode else ('взвеш.' if weight_mode=='weighted' else 'средние')})")
    if phi_plot.empty:
        st.info("Нет данных для пористости.")
    else:
        fig2 = px.box(phi_plot, x="res_label_phi", y=phi_y, points="outliers",
                      hover_data=[reservoir_col, well_col, "N_cat"])
        fig2.update_layout(xaxis_title="Коллектор/пласт (n по точкам)", yaxis_title=f"Пористость ({phi_unit})")
        st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader(f"Боксплот: вероятность трещин ({'сырые' if single_well_mode else ('взвеш.' if weight_mode=='weighted' else 'средние')})")
    if dis_plot.empty:
        st.info("Нет данных для dis_frac_rfn.")
    else:
        fig3 = px.box(dis_plot, x="res_label_disfrac", y=dis_y, points="outliers",
                      hover_data=[reservoir_col, well_col, "N_cat"])
        fig3.update_layout(xaxis_title="Коллектор/пласт (n по точкам)", yaxis_title="Вероятность трещин (доля)")
        st.plotly_chart(fig3, use_container_width=True)

# Ряд 3: проницаемость + вероятность УВ
col4, col5 = st.columns(2)
with col4:
    st.subheader(f"Боксплот: проницаемость ({perm_col}) (лог, {'сырые' if single_well_mode else ('взвеш.' if weight_mode=='weighted' else 'средние')})")
    if kpr_plot.empty:
        st.info(f"Нет данных для {perm_col}.")
    else:
        fig4 = px.box(kpr_plot, x="res_label_kpr", y=kpr_y, points="outliers",
                      hover_data=[reservoir_col, well_col, "N_cat"])
        fig4.update_layout(xaxis_title="Коллектор/пласт (n по точкам)", yaxis_title=f"Проницаемость ({perm_col})")
        fig4.update_yaxes(type="log")
        st.plotly_chart(fig4, use_container_width=True)

with col5:
    st.subheader(f"Боксплот: вероятность УВ SAT_rf ({'сырые' if single_well_mode else ('взвеш.' if weight_mode=='weighted' else 'средние')})")
    if sat_plot.empty:
        st.info("Нет данных для SAT_rf.")
    else:
        fig5 = px.box(sat_plot, x="res_label_sat", y=sat_y, points="outliers",
                      hover_data=[reservoir_col, well_col, "N_cat"])
        fig5.update_layout(xaxis_title="Коллектор/пласт (n по точкам)", yaxis_title=f"Вероятность УВ ({sat_unit})")
        st.plotly_chart(fig5, use_container_width=True)

# Ряд 4: водоудерживающая способность + кросс‑плот PHI vs PERM (лог Y)
col6_left, col6_right = st.columns(2)
with col6_left:
    st.subheader(f"Боксплот: водоудерж. способность kvo_rf ({'сырые' if single_well_mode else ('взвеш.' if weight_mode=='weighted' else 'средние')})")
    if kvo_plot.empty:
        st.info("Нет данных для kvo_rf.")
    else:
        fig6 = px.box(
            kvo_plot, x="res_label_kvo", y=kvo_y, points="outliers",
            hover_data=[reservoir_col, well_col, "N_cat"]
        )
        fig6.update_layout(xaxis_title="Коллектор/пласт (n по точкам)", yaxis_title=f"kvo_rf ({kvo_unit})")
        st.plotly_chart(fig6, use_container_width=True)

with col6_right:
    st.subheader(f"Кросс-плот: пористость vs проницаемость ({perm_col}), лог Y, раскраска по литологии")
    cross_df = res_df.dropna(subset=[poro_col, perm_col, lith_col]).copy()
    cross_df = cross_df[cross_df[perm_col] > 0]
    if cross_df.empty:
        st.info("Нет данных для построения кросс-плота.")
    else:
        fig_cross = px.scatter(
            cross_df,
            x=poro_col,
            y=perm_col,
            color=lith_col,
            hover_data=[well_col, reservoir_col] if all(c in cross_df.columns for c in [well_col, reservoir_col]) else None,
            opacity=0.7
        )
        fig_cross.update_yaxes(type="log")
        fig_cross.update_layout(
            xaxis_title=f"Пористость ({phi_unit})",
            yaxis_title=f"Проницаемость ({perm_col})",
            legend_title="Литология"
        )
        st.plotly_chart(fig_cross, use_container_width=True)

# Объединённая таблица (средние по (пласт × скважина)) — только при нескольких скважинах
if not single_well_mode:
    combined = (mean_phi[[reservoir_col, well_col, "phi_mean"]]
                .merge(mean_disfrac[[reservoir_col, well_col, "dis_frac_mean"]], on=[reservoir_col, well_col], how="outer")
                .merge(mean_kpr[[reservoir_col, well_col, "kpr_mean"]], on=[reservoir_col, well_col], how="outer")
                .merge(mean_sat[[reservoir_col, well_col, "sat_mean"]], on=[reservoir_col, well_col], how="outer")
                .merge(mean_kvo[[reservoir_col, well_col, "kvo_mean"]], on=[reservoir_col, well_col], how="outer"))
    with st.expander("Показать объединённую агрегированную таблицу (по всем метрикам)"):
        st.dataframe(combined.sort_values([reservoir_col, well_col]).reset_index(drop=True))
else:
    st.caption("Выбрана одна скважина: боксплоты по сырым данным; агрегированную таблицу не показываю.")