import io
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Плавная карта: RBF-интерполяция + интерактивные подписи
from scipy.interpolate import Rbf, griddata
from scipy.spatial import Delaunay
import plotly.graph_objects as go

st.set_page_config(page_title="Иерархическая кластеризация: сходство group_number", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stat.xlsx")


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def get_numeric_and_categorical_columns(df: pd.DataFrame, id_col: str) -> Tuple[List[str], List[str]]:
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str], standardize: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    # Совместимость с разными версиями scikit-learn (sparse_output появилось в 1.2)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    # ColumnTransformer: откат, если verbose_feature_names_out не поддерживается
    try:
        return ColumnTransformer(
            transformers=[("num", numeric_pipeline, numeric_cols),
                          ("cat", categorical_pipeline, categorical_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    except TypeError:
        return ColumnTransformer(
            transformers=[("num", numeric_pipeline, numeric_cols),
                          ("cat", categorical_pipeline, categorical_cols)],
            remainder="drop",
        )


def ensure_unique_labels(labels: List[str]) -> List[str]:
    seen, out = {}, []
    for lbl in labels:
        seen[lbl] = seen.get(lbl, 0) + 1
        out.append(lbl if seen[lbl] == 1 else f"{lbl}#{seen[lbl]}")
    return out


@st.cache_data(show_spinner=False)
def compute_features(df: pd.DataFrame, id_col: str, selected_numeric: List[str],
                     selected_categorical: List[str], standardize: bool) -> Tuple[np.ndarray, List[str]]:
    preprocessor = make_preprocessor(selected_numeric, selected_categorical, standardize)
    X = preprocessor.fit_transform(df[selected_numeric + selected_categorical])
    labels = ensure_unique_labels(df[id_col].astype(str).tolist())
    return X, labels


def compute_top_k_similar(X: np.ndarray, labels: List[str], target_label: str, metric: str, k: int):
    idx = {l: i for i, l in enumerate(labels)}
    if target_label not in idx:
        raise ValueError("Выбранный идентификатор не найден.")
    ti = idx[target_label]
    d = pairwise_distances(X, X[ti].reshape(1, -1), metric=metric).ravel()
    order = [i for i in np.argsort(d) if i != ti][:k]
    return [labels[i] for i in order], d[order]


def plot_dendrogram_subset(X_subset: np.ndarray, labels_subset: List[str],
                           linkage_method: str, metric: str,
                           target_display_label: str):
    metric_for_linkage = "euclidean" if linkage_method == "ward" else metric
    Z = linkage(X_subset, method=linkage_method, metric=metric_for_linkage)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    dendrogram(Z, labels=labels_subset, leaf_rotation=90, leaf_font_size=12, ax=ax)
    ax.set_title("Дендрограмма (целевой + 10 похожих)")
    ax.set_ylabel("Дистанция")
    for tick in ax.get_xmajorticklabels():
        if tick.get_text() == target_display_label:
            tick.set_color("crimson")
            tick.set_fontweight("bold")
        else:
            tick.set_color("black")
    fig.tight_layout()
    return fig


def plot_dendrogram_full(X: np.ndarray, labels: List[str], linkage_method: str,
                         metric: str, max_leaves: Optional[int] = None):
    metric_for_linkage = "euclidean" if linkage_method == "ward" else metric
    Z = linkage(X, method=linkage_method, metric=metric_for_linkage)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    if max_leaves and max_leaves < len(labels):
        dendrogram(Z, truncate_mode="lastp", p=max_leaves, leaf_font_size=9, ax=ax)
        ax.set_title(f"Дендрограмма (усеченная, последние {max_leaves} кластеров)")
    else:
        dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=12, ax=ax)
        ax.set_title("Дендрограмма (все объекты)")
    ax.set_ylabel("Дистанция")
    fig.tight_layout()
    return fig


def format_display_label(row: pd.Series, id_value: str) -> str:
    # Подпись: group_number | well | Q
    parts = [id_value]
    if "well" in row.index and pd.notna(row["well"]) and str(row["well"]).strip() != "":
        parts.append(str(row["well"]))
    if "Q" in row.index and pd.notna(row["Q"]):
        try:
            q = float(row["Q"])
            parts.append(f"Q={q:.3g}")
        except Exception:
            parts.append(f"Q={row['Q']}")
    return " | ".join(parts)


def rbf_interpolate_safe(x: np.ndarray, y: np.ndarray, v: np.ndarray, GX: np.ndarray, GY: np.ndarray) -> np.ndarray:
    # Удаляем дубликаты координат, усредняя значения
    pts = np.column_stack([x, y])
    uniq, idx = np.unique(pts, axis=0, return_inverse=True)
    v_mean = np.zeros(len(uniq))
    counts = np.zeros(len(uniq))
    np.add.at(v_mean, idx, v)
    np.add.at(counts, idx, 1)
    v_mean /= np.maximum(counts, 1)

    x_u, y_u, v_u = uniq[:, 0], uniq[:, 1], v_mean

    span_x = x_u.max() - x_u.min()
    span_y = y_u.max() - y_u.min()
    span = max(span_x, span_y) if max(span_x, span_y) > 0 else 1.0
    eps = 0.1 * span

    for func, smooth in [("multiquadric", 1e-6), ("linear", 1e-6), ("thin_plate", 1e-6)]:
        try:
            rbf = Rbf(x_u, y_u, v_u, function=func, epsilon=eps, smooth=smooth)
            return rbf(GX, GY)
        except Exception:
            continue

    Zi = griddata(uniq, v_u, (GX, GY), method="linear")
    if np.isnan(Zi).all():
        Zi = griddata(uniq, v_u, (GX, GY), method="nearest")
    return Zi


st.title("Иерархическая кластеризация: сходство по group_number")
st.caption("Данные читаются из stat.xlsx в этой же папке.")

# Загрузка с учетом mtime для авто-перечтения
try:
    mtime = os.path.getmtime(DATA_PATH)
    df = load_data_from_path(DATA_PATH, mtime)
except Exception as e:
    st.error(f"Ошибка чтения '{DATA_PATH}': {e}")
    st.stop()

if df.empty:
    st.error("Файл пустой.")
    st.stop()

# Таблица со строками, где TEST заполнен
st.subheader("Строки с наличием TEST")

if "TEST" in df.columns:
    mask = df["TEST"].notna() & (df["TEST"].astype(str).str.strip() != "")
    cols = [c for c in ["group_number", "well", "Q", 'fluid type',  'top', 'bottom',
                        'coll_type', 'h' , 'BF', "Литология по ГИС", "TEST", "TYPE"] if c in df.columns]
    test_tbl = df.loc[mask, cols]
    
    if test_tbl.empty:
        st.info("Нет строк с заполненным TEST.")
    else:
        # Фильтры
        with st.expander("Фильтры", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Фильтр по fluid type
                if "fluid type" in test_tbl.columns:
                    test_values = sorted(test_tbl["fluid type"].dropna().unique())
                    selected_tests = st.multiselect("fluid type", options=test_values, default=test_values)
                else:
                    selected_tests = []
                
                # Фильтр по TYPE
                if "TYPE" in test_tbl.columns:
                    type_values = sorted(test_tbl["TYPE"].dropna().unique())
                    selected_types = st.multiselect("TYPE", options=type_values, default=type_values)
                else:
                    selected_types = []
            
            with col2:
                # Фильтр по coll_type
                if "coll_type" in test_tbl.columns:
                    coll_values = sorted(test_tbl["coll_type"].dropna().unique())
                    selected_wells = st.multiselect("coll_type", options=coll_values, default=coll_values)
                else:
                    selected_wells = []
                
                # Фильтр по Литология по качеству
                if "BF" in test_tbl.columns:
                    bf_values = sorted(test_tbl["BF"].dropna().unique())
                    selected_bf = st.multiselect("BF", options=bf_values, default=bf_values)
                else:
                    selected_bf = []
                
                # Фильтр по Литология по ГИС
                if "Литология по ГИС" in test_tbl.columns:
                    lith_gis_values = sorted(test_tbl["Литология по ГИС"].dropna().unique())
                    selected_lith_gis = st.multiselect("Литология по ГИС", options=lith_gis_values, default=lith_gis_values)
                else:
                    selected_lith_gis = []
            

        
        # Применяем фильтры
        filtered_tbl = test_tbl.copy()
        
        if selected_tests and "fluid type" in filtered_tbl.columns:
            filtered_tbl = filtered_tbl[filtered_tbl["fluid type"].isin(selected_tests)]
        
        if selected_types and "TYPE" in filtered_tbl.columns:
            filtered_tbl = filtered_tbl[filtered_tbl["TYPE"].isin(selected_types)]
        
        if selected_wells and "coll_type" in filtered_tbl.columns:
            filtered_tbl = filtered_tbl[filtered_tbl["coll_type"].isin(selected_wells)]
            
        if selected_bf and "BF" in filtered_tbl.columns:
                filtered_tbl = filtered_tbl[filtered_tbl["BF"].isin(selected_bf)]
        
        if selected_lith_gis and "Литология по ГИС" in filtered_tbl.columns:
            filtered_tbl = filtered_tbl[filtered_tbl["Литология по ГИС"].isin(selected_lith_gis)]
        

        # Показываем результат
        st.write(f"Найдено строк: {len(filtered_tbl)} из {len(test_tbl)}")
        st.dataframe(filtered_tbl, use_container_width=True, hide_index=True)
        
        # Кнопка экспорта отфильтрованных данных
        if not filtered_tbl.empty:
            csv_buf = io.StringIO()
            filtered_tbl.to_csv(csv_buf, index=False, encoding="utf-8")
            st.download_button(
                "Скачать отфильтрованные данные (CSV)",
                csv_buf.getvalue().encode("utf-8"),
                "filtered_test_data.csv",
                "text/csv"
            )
else:
    st.info("Колонка 'TEST' отсутствует в данных.")

# Идентификатор
if "group_number" not in df.columns:
    st.error("Не найдена колонка 'group_number'.")
    st.stop()
id_col = "group_number"

# Признаки
num_all, cat_all = get_numeric_and_categorical_columns(df, id_col)
if not num_all and not cat_all:
    st.error("Нет признаков, кроме идентификатора.")
    st.stop()

preferred_numeric = [
    "GK_NORM", "NK", "DTP_NORM", "log_BK_NORM", "frac_rf",
    "dis_frac_rfn", "por_rf", "kvo_rf", "SAT_rf",
    "log10Kpr_tim", "log10Kpr_rf",
]

left, mid, right = st.columns(3)
with left:
    default_nums = [c for c in preferred_numeric if c in num_all] or num_all
    sel_num = st.multiselect("Числовые признаки", options=num_all, default=default_nums)
with mid:
    sel_cat = st.multiselect("Категориальные признаки (one-hot)", options=cat_all, default=[])
with right:
    standardize = st.checkbox("Стандартизация числовых", value=True)
    metric = st.selectbox("Метрика", options=["euclidean", "cityblock", "cosine"], index=0)
    linkage_method = st.selectbox("Линковка", options=["ward", "average", "complete", "single"], index=0)
    if linkage_method == "ward" and metric != "euclidean":
        st.info("Ward требует евклидову метрику; для дендрограммы будет использована euclidean.")

if len(sel_num) + len(sel_cat) == 0:
    st.warning("Выберите хотя бы один признак.")
    st.stop()

# Очистка по идентификатору
df = df.dropna(subset=[id_col]).reset_index(drop=True)

# Векторизация
X, labels = compute_features(df, id_col, sel_num, sel_cat, standardize)

st.subheader("Сходство для выбранного идентификатора")
target_label = st.selectbox("Значение идентификатора (group_number)", options=labels)
idx_by_label = {l: i for i, l in enumerate(labels)}
target_idx = idx_by_label[target_label]

# TEST и TYPE выбранного
meta_parts = []
if "TEST" in df.columns:
    meta_parts.append(f"TEST: **{df.loc[target_idx, 'TEST']}**")
if "TYPE" in df.columns:
    meta_parts.append(f"TYPE: **{df.loc[target_idx, 'TYPE']}**")
if meta_parts:
    st.markdown(" | ".join(meta_parts))

# Таблица значений признаков выбранного
selected_cols_for_table = sel_num + sel_cat
target_row = df.iloc[target_idx]
feat_df = (
    target_row[selected_cols_for_table]
    .to_frame(name="value")
    .rename_axis("feature")
    .reset_index()
)
st.write("Значения выбранных признаков:")
st.dataframe(feat_df, use_container_width=True, hide_index=True)

# Топ-10 похожих
neighbor_labels, neighbor_dists = compute_top_k_similar(X, labels, target_label, metric, 10)
neighbor_indices = [idx_by_label[l] for l in neighbor_labels]

# Таблица похожих (+ well/top/bottom/TEST/TYPE)
extra_cols = [c for c in ["well", "top", "bottom", 'BF', "TEST", "TYPE"] if c in df.columns]
neighbors_extra = df.loc[neighbor_indices, extra_cols] if extra_cols else pd.DataFrame(index=neighbor_indices)
neighbors_df = pd.DataFrame({id_col: neighbor_labels, "distance": neighbor_dists})
if not neighbors_extra.empty:
    neighbors_df = pd.concat([neighbors_df.reset_index(drop=True), neighbors_extra.reset_index(drop=True)], axis=1)
if "well" in neighbors_df.columns:
    neighbors_df = neighbors_df[[id_col, "well"] + [c for c in neighbors_df.columns if c not in [id_col, "well"]]]
neighbors_df = neighbors_df.sort_values("distance", ascending=True, ignore_index=True)
st.write("10 наиболее похожих:")
st.dataframe(neighbors_df, use_container_width=True, hide_index=True)

buf = io.StringIO()
neighbors_df.to_csv(buf, index=False, encoding="utf-8")
st.download_button("Скачать топ-10 (CSV)", buf.getvalue().encode("utf-8"), "top10_neighbors.csv", "text/csv")

# Подписи: group_number | well | Q
def make_display_labels(indices: List[int]) -> List[str]:
    out = []
    for i in indices:
        orig_id = str(df.iloc[i][id_col])
        out.append(format_display_label(df.iloc[i], orig_id))
    return out


subset_indices = [target_idx] + neighbor_indices
subset_display_labels = make_display_labels(subset_indices)
target_display_label = subset_display_labels[0]
X_subset = X[subset_indices, :]

# Дендрограмма и кроссплот
col_left, col_right = st.columns(2)

with col_left:
    fig_subset = plot_dendrogram_subset(
        X_subset=X_subset,
        labels_subset=subset_display_labels,
        linkage_method=linkage_method,
        metric=metric,
        target_display_label=target_display_label,
    )
    st.pyplot(fig_subset, use_container_width=True)
    png_buf = io.BytesIO()
    fig_subset.savefig(png_buf, format="png", bbox_inches="tight", dpi=200)
    st.download_button("Скачать дендрограмму (PNG)", png_buf.getvalue(), "dendrogram_top10.png", "image/png")

with col_right:
    has_x = "por_rf" in df.columns
    has_y = "perm_timur_ad" in df.columns
    if has_x and has_y:
        subset_df = df.iloc[subset_indices].copy()
        if "Литология по ГИС" in subset_df.columns:
            subset_df["lith"] = subset_df["Литология по ГИС"].astype(str)
        else:
            subset_df["lith"] = "неизвестно"
        subset_df = subset_df.rename(columns={"por_rf": "x", "perm_timur_ad": "y"})
        subset_df = subset_df.loc[subset_df["x"].notna() & subset_df["y"].notna() & (subset_df["y"] > 0)]
        if subset_df.empty:
            st.warning("Нет валидных значений для логарифмической оси Y (perm_timur_ad > 0).")
        else:
            lith_values = subset_df["lith"]
            unique_lith = lith_values.unique()
            cmap = plt.get_cmap("tab20", max(1, len(unique_lith)))
            color_map = {cat: cmap(i) for i, cat in enumerate(unique_lith)}
            colors = lith_values.map(color_map)

            fig_sc, ax_sc = plt.subplots(figsize=(5, 4), dpi=150)
            ax_sc.scatter(subset_df["x"], subset_df["y"], c=colors, s=60, edgecolors="k", alpha=0.9)
            ax_sc.set_yscale("log")

            display_labels_scatter = make_display_labels(subset_df.index.tolist())
            target_df_index = subset_indices[0]
            for (x, y, lbl, idx) in zip(subset_df["x"], subset_df["y"], display_labels_scatter, subset_df.index):
                is_target = (idx == target_df_index)
                ax_sc.text(x, y, lbl,
                           fontsize=9 if is_target else 8,
                           ha="left", va="bottom",
                           color="crimson" if is_target else "black",
                           fontweight="bold" if is_target else "normal")
            ax_sc.set_xlabel("Кп (por_rf)")
            ax_sc.set_ylabel("Кпр (perm_timur_ad), лог шкала")
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=cat,
                           markerfacecolor=color_map[cat], markeredgecolor="k", markersize=8)
                for cat in unique_lith
            ]
            ax_sc.legend(handles=handles, title="Литология по ГИС", fontsize=8)
            fig_sc.tight_layout()
            st.pyplot(fig_sc, use_container_width=True)
    else:
        st.warning("Нет колонок 'por_rf' и/или 'perm_timur_ad' для кроссплота.")

# PCA и карта
st.subheader("PCA-биплот (целевой + 10 похожих)")
col_pca, col_map = st.columns(2)

# PCA
try:
    if len(sel_num) < 2:
        raise ValueError("Для PCA нужно минимум 2 числовых признака.")
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler() if standardize else None

    X_num_full = num_imputer.fit_transform(df[sel_num])
    if scaler:
        X_num_full = scaler.fit_transform(X_num_full)
    X_num_subset = X_num_full[subset_indices, :]

    if np.allclose(X_num_subset.std(axis=0), 0):
        with col_pca:
            st.warning("Недостаточная вариативность для PCA.")
    else:
        pca = PCA(n_components=2, random_state=0)
        scores = pca.fit_transform(X_num_subset)
        loadings = pca.components_.T
        evr = pca.explained_variance_ratio_

        with col_pca:
            fig_pca, ax_pca = plt.subplots(figsize=(5,6), dpi=150)
            ax_pca.scatter(scores[1:, 0], scores[1:, 1],  c="gray", s=60, edgecolors="k", alpha=0.9, label="Похожие")
            ax_pca.scatter(scores[0, 0], scores[0, 1], c="crimson", s=100, edgecolors="k", marker="*", label="Целевой")
            for i, (x, y) in enumerate(scores):
                ax_pca.text(x, y, subset_display_labels[i],
                            fontsize=9 if i == 0 else 8, ha="left", va="bottom",
                            color="crimson" if i == 0 else "black",
                            fontweight="bold" if i == 0 else "normal")
            arrow_scale = np.max(np.linalg.norm(scores, axis=1)) *1.3
            for j, feature in enumerate(sel_num):
                vx, vy = loadings[j, 0] * arrow_scale, loadings[j, 1] * arrow_scale
                ax_pca.arrow(0, 0, vx, vy, color="tab:blue",
                             width=0.0006, head_width=0.012, head_length=0.02,
                             length_includes_head=True, alpha=0.9)
                ax_pca.text(vx, vy, feature, fontsize=8, color="tab:blue", ha="center", va="center")
            ax_pca.axhline(0, color="lightgray", linewidth=1)
            ax_pca.axvline(0, color="lightgray", linewidth=1)
            ax_pca.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
            ax_pca.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
            ax_pca.legend(fontsize=8)
            fig_pca.tight_layout()
            st.pyplot(fig_pca, use_container_width=True)
except Exception as e:
    with col_pca:
        st.warning(f"PCA не построен: {e}")

# Карта (интерполяция по всем скважинам; отображаем только целевой+похожие)
with col_map:
    #st.caption("Расположение на площади")
    prop_col = st.session_state.get("interp_prop", "por_rf")

    has_coords = all(c in df.columns for c in ["Surface X", "Surface Y"])
    if has_coords and prop_col in df.columns:
        all_valid = df[["Surface X", "Surface Y", prop_col]].dropna()
        if all_valid.empty:
            st.warning("Нет валидных данных для интерполяции.")
        else:
            x_all = all_valid["Surface X"].to_numpy(float)
            y_all = all_valid["Surface Y"].to_numpy(float)
            v_all = all_valid[prop_col].to_numpy(float)

            xmin, xmax = x_all.min(), x_all.max()
            ymin, ymax = y_all.min(), y_all.max()
            gx = np.linspace(xmin, xmax, 400)
            gy = np.linspace(ymin, ymax, 400)
            GX, GY = np.meshgrid(gx, gy)

            Zi = rbf_interpolate_safe(x_all, y_all, v_all, GX, GY)

            # Маска по выпуклой оболочке (не экстраполируем далеко за точки)
            try:
                tri = Delaunay(np.c_[x_all, y_all])
                inside = tri.find_simplex(np.c_[GX.ravel(), GY.ravel()]) >= 0
                Zi = np.where(inside.reshape(GX.shape), Zi, np.nan)
            except Exception:
                pass

            subset_map_df = df.iloc[subset_indices].copy()
            subset_map_df = subset_map_df[subset_map_df["Surface X"].notna() & subset_map_df["Surface Y"].notna()]
            pts_sel = subset_map_df[["Surface X", "Surface Y"]].to_numpy(float)
            labels_map = make_display_labels(subset_map_df.index.tolist())
            target_df_index = subset_indices[0]
            is_target = (subset_map_df.index.values == target_df_index)


            fig_map = go.Figure()
            fig_map.add_trace(go.Heatmap(z=Zi, x=gx, y=gy,
                                         colorscale="Viridis", showscale=False, zsmooth="best"))
            if np.any(~is_target):
                fig_map.add_trace(go.Scatter(
                    x=pts_sel[~is_target, 0], y=pts_sel[~is_target, 1],
                    mode="markers",
                    marker=dict(color="black", size=8, line=dict(color="white", width=0.5)),
                    hovertext=[labels_map[i] for i in np.where(~is_target)[0]],
                    hoverinfo="text", showlegend=False))
            if np.any(is_target):
                fig_map.add_trace(go.Scatter(
                    x=pts_sel[is_target, 0], y=pts_sel[is_target, 1],
                    mode="markers",
                    marker=dict(color="crimson", size=12, symbol="star", line=dict(color="black", width=0.5)),
                    hovertext=[labels_map[i] for i in np.where(is_target)[0]],
                    hoverinfo="text", showlegend=False))
            fig_map.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Surface X", yaxis_title="Surface Y",
                xaxis=dict(constrain="domain"),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Нет колонок 'Surface X' и/или 'Surface Y' для карты.")

    # селектор свойства под картой
    prop_options = []
    for c in sel_num + ["por_rf"]:
        if c in df.columns and c not in prop_options:
            prop_options.append(c)
    st.selectbox(
        "Свойство для интерполяции",
        options=prop_options,
        index=(prop_options.index("por_rf") if "por_rf" in prop_options else 0),
        key="interp_prop",
    )

with st.expander("Полная дендрограмма"):
    max_leaves = st.number_input("Усечь до N последних кластеров (0 — не усекать)", min_value=0,
                                 max_value=max(0, len(labels)), value=min(0, len(labels)))
    fig_full = plot_dendrogram_full(X, labels, linkage_method, metric, None if max_leaves == 0 else int(max_leaves))
    st.pyplot(fig_full, use_container_width=True)