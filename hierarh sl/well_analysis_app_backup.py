import streamlit as st

st.set_page_config(page_title="Анализ скважин: LAS + Испытания", layout="wide")

import io
import os
import tempfile
import base64
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import zipfile

import numpy as np
import pandas as pd
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

# Для работы с LAS файлами
try:
    import lasio
    LASIO_AVAILABLE = True
except ImportError:
    LASIO_AVAILABLE = False
    st.warning("lasio не установлен. Установите: pip install lasio")

# Импорт модуля агломеративного бустинга
try:
    from agglomerative_boosting import create_boosting_interface
    BOOSTING_AVAILABLE = True
except ImportError:
    BOOSTING_AVAILABLE = False

def generate_clustering_report(df, sel_num, sel_cat, standardize, metric, linkage_method, id_col):
    """Генерация HTML отчёта по кластеризации"""
    
    # Создаём HTML шаблон
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Отчёт по кластеризации коллекторов</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2E86AB; }}
            h2 {{ color: #A23B72; }}
            h3 {{ color: #F18F01; }}
            .info-box {{ background-color: #f0f8ff; padding: 10px; border-left: 4px solid #2E86AB; margin: 10px 0; }}
            .warning-box {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
            .success-box {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #2E86AB; }}
            .center {{ text-align: center; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 Отчёт по кластеризации коллекторов</h1>
        
        <div class="info-box">
            <strong>📅 Дата генерации:</strong> {date}<br>
            <strong>🔍 Критерии фильтрации:</strong> COLL == 1 и TEST не пустой<br>
            <strong>📊 Количество коллекторов:</strong> {count}
        </div>
        
        <h2>⚙️ Параметры анализа</h2>
        <table>
            <tr><th>Параметр</th><th>Значение</th></tr>
            <tr><td>Числовые признаки</td><td>{numeric_features}</td></tr>
            <tr><td>Категориальные признаки</td><td>{categorical_features}</td></tr>
            <tr><td>Стандартизация</td><td>{standardize}</td></tr>
            <tr><td>Метрика расстояния</td><td>{metric}</td></tr>
            <tr><td>Метод связывания</td><td>{linkage_method}</td></tr>
        </table>
        
        <h2>📋 Данные для анализа</h2>
        {data_table}
        
        <h2>📈 Результаты кластеризации</h2>
        {clustering_results}
        
        <h2>📊 Статистика</h2>
        {statistics}
        
        <div class="success-box">
            <strong>✅ Отчёт успешно сгенерирован!</strong><br>
            Всего проанализировано {count} коллекторов с COLL=1 и не пустым TEST.
        </div>
    </body>
    </html>
    """
    
    try:
        # Подготавливаем данные
        df_clean = df.dropna(subset=[id_col]).reset_index(drop=True)
        
        if len(df_clean) == 0:
            return html_template.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                count=0,
                numeric_features=", ".join(sel_num) if sel_num else "Нет",
                categorical_features=", ".join(sel_cat) if sel_cat else "Нет",
                standardize="Да" if standardize else "Нет",
                metric=metric,
                linkage_method=linkage_method,
                data_table="<p>Нет данных для анализа</p>",
                clustering_results="<p>Недостаточно данных для кластеризации</p>",
                statistics="<p>Статистика недоступна</p>"
            )
        
        # Создаём таблицу данных
        display_cols = ['group_number', 'well', 'top', 'bottom', 'h', 'COLL', 'TEST']
        available_display_cols = [col for col in display_cols if col in df_clean.columns]
        data_table_html = df_clean[available_display_cols].to_html(index=False, classes="table")
        
        # Вычисляем статистику
        # Безопасно вычисляем статистику по толщине
        h_mean = f"{df_clean['h'].mean():.2f}" if 'h' in df_clean.columns else 'N/A'
        h_min = f"{df_clean['h'].min():.2f}" if 'h' in df_clean.columns else 'N/A'
        h_max = f"{df_clean['h'].max():.2f}" if 'h' in df_clean.columns else 'N/A'
        
        stats_html = f"""
        <table>
            <tr><th>Метрика</th><th>Значение</th></tr>
            <tr><td>Общее количество коллекторов</td><td class="metric">{len(df_clean)}</td></tr>
            <tr><td>Количество скважин</td><td class="metric">{df_clean['well'].nunique() if 'well' in df_clean.columns else 'N/A'}</td></tr>
            <tr><td>Средняя толщина, м</td><td class="metric">{h_mean}</td></tr>
            <tr><td>Мин. толщина, м</td><td class="metric">{h_min}</td></tr>
            <tr><td>Макс. толщина, м</td><td class="metric">{h_max}</td></tr>
        </table>
        """
        
        # Кластеризация (если достаточно данных)
        clustering_results = ""
        if len(sel_num) + len(sel_cat) > 0 and len(df_clean) > 1:
            try:
                # Векторизация
                X, labels = compute_features(df_clean, id_col, sel_num, sel_cat, standardize)
                
                if len(X) > 1:
                    # Создаём дендрограмму
                    metric_for_linkage = "euclidean" if linkage_method == "ward" else metric
                    Z = linkage(X, method=linkage_method, metric=metric_for_linkage)
                    
                    # Сохраняем дендрограмму в base64
                    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
                    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=10, ax=ax)
                    ax.set_title("Дендрограмма коллекторов")
                    ax.set_ylabel("Дистанция")
                    fig.tight_layout()
                    
                    # Конвертируем в base64
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    plt.close(fig)
                    
                    clustering_results = f"""
                    <h3>Дендрограмма</h3>
                    <img src="data:image/png;base64,{image_base64}" alt="Дендрограмма">
                    
                    <h3>Информация о кластеризации</h3>
                    <table>
                        <tr><th>Параметр</th><th>Значение</th></tr>
                        <tr><td>Количество объектов</td><td class="metric">{len(X)}</td></tr>
                        <tr><td>Размерность признаков</td><td class="metric">{X.shape[1]}</td></tr>
                        <tr><td>Метод связывания</td><td>{linkage_method}</td></tr>
                        <tr><td>Метрика расстояния</td><td>{metric}</td></tr>
                    </table>
                    """
                else:
                    clustering_results = "<p>Недостаточно данных для создания дендрограммы</p>"
                    
            except Exception as e:
                clustering_results = f"<div class='warning-box'>Ошибка при кластеризации: {str(e)}</div>"
        else:
            clustering_results = "<p>Не выбраны признаки для кластеризации</p>"
        
        # Формируем финальный HTML
        return html_template.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            count=len(df_clean),
            numeric_features=", ".join(sel_num) if sel_num else "Нет",
            categorical_features=", ".join(sel_cat) if sel_cat else "Нет",
            standardize="Да" if standardize else "Нет",
            metric=metric,
            linkage_method=linkage_method,
            data_table=data_table_html,
            clustering_results=clustering_results,
            statistics=stats_html
        )
        
    except Exception as e:
        error_html = f"""
        <div class="warning-box">
            <strong>❌ Ошибка при генерации отчёта:</strong><br>
            {str(e)}
        </div>
        """
        return html_template.format(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            count=0,
            numeric_features="Ошибка",
            categorical_features="Ошибка",
            standardize="Ошибка",
            metric="Ошибка",
            linkage_method="Ошибка",
            data_table="<p>Ошибка загрузки данных</p>",
            clustering_results=error_html,
            statistics="<p>Статистика недоступна</p>"
        )
    # st.warning("Модуль агломеративного бустинга недоступен")

# Функции для работы с LAS файлами
def read_las_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """Читает LAS файл и возвращает DataFrame"""
    if not LASIO_AVAILABLE:
        raise ImportError("lasio не установлен")
    
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Читаем LAS файл
        las = lasio.read(tmp_file_path)
        
        # Преобразуем в DataFrame
        df = las.df()
        df.reset_index(inplace=True)
        
        # Добавляем информацию о скважине
        well_name = getattr(las.well, 'WELL', {}).get('VALUE', filename.split('.')[0])
        df['well'] = str(well_name)
        
        return df
    finally:
        # Удаляем временный файл
        os.unlink(tmp_file_path)

def process_las_files(uploaded_files) -> pd.DataFrame:
    """Обрабатывает загруженные LAS файлы"""
    all_data = []
    
    for uploaded_file in uploaded_files:
        try:
            file_content = uploaded_file.read()
            df = read_las_file(file_content, uploaded_file.name)
            all_data.append(df)
            st.success(f"✅ {uploaded_file.name}: {len(df)} точек")
        except Exception as e:
            st.error(f"❌ Ошибка при обработке {uploaded_file.name}: {str(e)}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        st.success(f"📊 Всего загружено: {len(combined_df)} точек из {len(all_data)} файлов")
        return combined_df
    else:
        return pd.DataFrame()

def thick_to_dots(las: pd.DataFrame, core: pd.DataFrame, list_las_to_core: List[str]) -> pd.DataFrame:
    """
    Объединяет данные LAS и таблицы испытаний по интервалам
    
    Args:
        las: DataFrame с LAS данными (должен содержать колонки 'well', 'DEPTH')
        core: DataFrame с данными испытаний (должен содержать колонки 'well', 'top', 'bottom')
        list_las_to_core: список колонок из core для переноса в las
    
    Returns:
        Объединенный DataFrame
    """
    # Преобразуем названия скважин в строковый тип
    las = las.copy()
    core = core.copy()
    las['well'] = las['well'].astype(str)
    core['well'] = core['well'].astype(str)
    
    # Фильтруем core только по скважинам, которые есть в las
    core_filtered = core[core["well"].isin(las["well"].unique())].copy()
    
    if core_filtered.empty:
        st.warning("Нет общих скважин между LAS данными и таблицей испытаний")
        return las
    
    # Создаем список для результатов
    result_list = []
    
    # Проходим по скважинам
    for well_name, core_part in core_filtered.groupby("well"):
        las_part = las[las["well"] == well_name].copy()
        
        # Мержим данные через интервал
        for _, row in core_part.iterrows():
            mask = (las_part["DEPTH"] >= row["top"]) & (las_part["DEPTH"] < row["bottom"])
            for col in list_las_to_core:
                if col in row:
                    las_part.loc[mask, col] = row[col]
        
        result_list.append(las_part)
    
    return pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()

def calculate_group_number(df: pd.DataFrame, coll_column: str = 'coll') -> pd.DataFrame:
    """
    Рассчитывает group_number на основе изменений в колонках coll и well
    """
    df = df.copy()
    
    # Проверяем наличие необходимых колонок
    if coll_column not in df.columns:
        st.warning(f"⚠️ Колонка '{coll_column}' не найдена. group_number не будет рассчитан.")
        return df
    
    if 'well' not in df.columns:
        st.warning("⚠️ Колонка 'well' не найдена. group_number не будет рассчитан.")
        return df
    
    # Преобразуем в строковый тип для корректного сравнения
    df['well'] = df['well'].astype(str)
    df[coll_column] = df[coll_column].astype(str)
    
    # Сортируем по скважине и глубине для корректного расчета
    if 'DEPTH' in df.columns:
        df = df.sort_values(['well', 'DEPTH']).reset_index(drop=True)
    else:
        df = df.sort_values(['well']).reset_index(drop=True)
    
    # Определяем новые группы: изменение coll или well
    new_group = (df[coll_column] != df[coll_column].shift(1)) | (df['well'] != df['well'].shift(1))
    
    # Кумулятивно нумеруем группы по всему DataFrame
    df['group_number'] = new_group.cumsum()
    
    st.success(f"✅ group_number рассчитан: {df['group_number'].nunique()} уникальных групп")
    
    return df

def aggregate_to_collectors(df: pd.DataFrame, group_by_column: str = None) -> pd.DataFrame:
    """
    Преобразует поточечную таблицу в таблицу индексов коллекторов
    
    Args:
        df: DataFrame с поточечными данными
        group_by_column: колонка для группировки (аналог 'coll')
    """
    # Преобразуем названия скважин в строковый тип
    df = df.copy()
    if 'well' in df.columns:
        df['well'] = df['well'].astype(str)
    
    # Рассчитываем group_number если его нет
    if 'group_number' not in df.columns:
        st.info("🔄 Рассчитываем group_number...")
        
        # Определяем колонку для группировки
        if group_by_column and group_by_column in df.columns:
            coll_col = group_by_column
        elif (st.session_state.selected_coll_column and 
              st.session_state.selected_coll_column in df.columns):
            coll_col = st.session_state.selected_coll_column
        else:
            # Fallback: определяем колонку автоматически
            if 'coll' in df.columns:
                coll_col = 'coll'
            elif 'COLL' in df.columns:
                coll_col = 'COLL'
            elif 'коллектор' in df.columns:
                coll_col = 'коллектор'
            else:
                # Ищем первую подходящую колонку
                available_cols = [col for col in df.columns if col not in ['well', 'DEPTH', 'group_number']]
                coll_col = available_cols[0] if available_cols else 'coll'
        
        df = calculate_group_number(df, coll_col)
    
    # Фильтрация h < 0.4
    if 'h' in df.columns:
        initial_count = len(df)
        df = df[df['h'] >= 0.4]
        filtered_count = len(df)
        if initial_count != filtered_count:
            st.info(f"🔍 Отфильтровано {initial_count - filtered_count} точек с h < 0.4. Осталось {filtered_count} точек.")
    
    # Автоматически определяем все LAS кривые из данных
    # Исключаем служебные колонки (но включаем COLL)
    service_columns = ['well', 'DEPTH', 'group_number', 'top', 'bottom', 'h', 'coll', 'коллектор']
    
    # Все числовые колонки, которые не являются служебными - это и есть LAS кривые (включая COLL)
    available_las_curves = [c for c in df.columns 
                           if pd.api.types.is_numeric_dtype(df[c]) and c not in service_columns]
    
    # Используем выбранные пользователем признаки или автоматически определяем
    if hasattr(st.session_state, 'selected_las_curves') and st.session_state.selected_las_curves:
        selected_las_curves = st.session_state.selected_las_curves
    else:
        selected_las_curves = available_las_curves
    
    # Все выбранные LAS кривые для усреднения
    all_potential_mean_columns = selected_las_curves
    
    # Фильтруем только числовые колонки для усреднения
    mean_columns = []
    for col in all_potential_mean_columns:
        if col in df.columns:
            # Проверяем, что колонка числовая
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_columns.append(col)
            else:
                # Пытаемся преобразовать в числовой тип
                try:
                    # Создаем копию колонки для преобразования
                    converted_col = pd.to_numeric(df[col], errors='coerce')
                    # Проверяем, что есть хотя бы одно числовое значение
                    if not converted_col.isna().all():
                        # Заменяем оригинальную колонку на преобразованную
                        df[col] = converted_col
                        mean_columns.append(col)
                except Exception as e:
                    st.warning(f"⚠️ Не удалось преобразовать колонку '{col}' в числовой тип: {str(e)}")
                    pass  # Пропускаем колонку, если не удается преобразовать
    
    # Показываем информацию о найденных LAS кривых
    if available_las_curves:
        st.info(f"🔍 Найдено {len(available_las_curves)} LAS кривых для агрегации")
        if len(available_las_curves) <= 10:
            st.info(f"📊 Кривые: {', '.join(available_las_curves)}")
        else:
            st.info(f"📊 Кривые: {', '.join(available_las_curves[:10])}... (и еще {len(available_las_curves)-10})")
        
        if mean_columns:
            st.success(f"✅ {len(mean_columns)} кривых выбрано для усреднения")
        else:
            st.warning("⚠️ Не найдено числовых LAS кривых для усреднения")
    else:
        st.warning("⚠️ LAS кривые не найдены в данных")
    
    # Показываем отладочную информацию
    if st.checkbox("🔍 Показать отладочную информацию", help="Показать детали обработки колонок"):
        st.write("**Доступные колонки в данных:**")
        for col in df.columns:
            dtype = df[col].dtype
            st.write(f"- {col}: {dtype}")
        
        st.write("**Колонки для агрегации:**")
        st.write(f"- Усреднение (mean): {mean_columns}")
        st.write(f"- Суммирование (sum): {count_columns}")
        st.write(f"- Модальное значение: {mode_columns}")
        st.write(f"- Min/Max: {min_max_columns}")
    
    # Колонки для суммирования (обычно счетчики) - только числовые
    potential_count_columns = ['COLL_poro_type', 'COLL_frac_type', 'COLL_mix_type', 'COLL']
    count_columns = [c for c in potential_count_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    # Колонки для модального значения (категориальные)
    if hasattr(st.session_state, 'selected_categorical') and st.session_state.selected_categorical:
        selected_categorical = st.session_state.selected_categorical
    else:
        potential_mode_columns = ['well', 'TYPE', 'Кластеры ГИС', 'Литология по ГИС', 'TEST', 'BF', 'fluid type', 'coll_type', 'coll', 'COLL']
        selected_categorical = [c for c in potential_mode_columns if c in df.columns]
    
    mode_columns = [c for c in selected_categorical if c in df.columns]
    
    # Колонки для min/max (обычно глубины) - только числовые
    potential_min_max_columns = ['DEPTH']
    min_max_columns = [c for c in potential_min_max_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    def most_frequent(s):
        s = s.dropna()
        return s.value_counts().idxmax() if not s.empty else np.nan
    
    # Создаем словарь агрегации
    agg_dict = {}
    
    # Добавляем колонки для усреднения
    for c in mean_columns:
        if c in df.columns:
            agg_dict[c] = 'mean'
    
    # Добавляем колонки для суммирования
    for c in count_columns:
        if c in df.columns:
            agg_dict[c] = 'sum'
    
    # Добавляем колонки для модального значения
    for c in mode_columns:
        if c in df.columns:
            agg_dict[c] = most_frequent
    
    # Добавляем колонки для min/max
    for c in min_max_columns:
        if c in df.columns:
            agg_dict[c] = ['min', 'max']
    
    # Группируем по group_number
    if 'group_number' not in df.columns:
        st.error("Колонка 'group_number' не найдена в данных")
        return df
    
    # Преобразуем group_number в строковый тип
    df['group_number'] = df['group_number'].astype(str)
    
    # Дополнительная проверка: убеждаемся, что все колонки в agg_dict действительно числовые
    final_agg_dict = {}
    for col, agg_func in agg_dict.items():
        if col in df.columns:
            if agg_func == 'mean':
                # Дополнительная проверка для mean
                if pd.api.types.is_numeric_dtype(df[col]):
                    final_agg_dict[col] = agg_func
                else:
                    st.warning(f"⚠️ Пропускаем колонку '{col}' для усреднения - не числовая")
            elif agg_func == 'sum':
                # Дополнительная проверка для sum
                if pd.api.types.is_numeric_dtype(df[col]):
                    final_agg_dict[col] = agg_func
                else:
                    st.warning(f"⚠️ Пропускаем колонку '{col}' для суммирования - не числовая")
            elif agg_func == ['min', 'max']:
                # Дополнительная проверка для min/max
                if pd.api.types.is_numeric_dtype(df[col]):
                    final_agg_dict[col] = agg_func
                else:
                    st.warning(f"⚠️ Пропускаем колонку '{col}' для min/max - не числовая")
            else:
                # Для most_frequent и других функций
                final_agg_dict[col] = agg_func
    
    # Проверяем, что есть колонки для агрегации
    if not final_agg_dict:
        st.error("❌ Нет подходящих колонок для агрегации")
        return df
    
    try:
        result = df.groupby('group_number', dropna=False).agg(final_agg_dict).reset_index()
    except Exception as e:
        st.error(f"❌ Ошибка при агрегации: {str(e)}")
        st.error("Попробуйте проверить типы данных в исходных файлах")
        return df
    
    # Плоские колонки с заменой DEPTH_min/DEPTH_max
    new_columns = []
    for col in result.columns:
        if isinstance(col, tuple):
            if col[0] == 'DEPTH' and col[1] == 'min':
                new_columns.append('top')
            elif col[0] == 'DEPTH' and col[1] == 'max':
                new_columns.append('bottom')
            else:
                new_columns.append(col[0])
        else:
            new_columns.append(col)
    
    result.columns = new_columns
    
    # Добавляем толщину
    if 'top' in result.columns and 'bottom' in result.columns:
        result['h'] = result['bottom'] - result['top']
    
    # Фильтруем коллекторы
    if 'coll_type' in result.columns and 'h' in result.columns:
        result = result[(result['h'] > 0.3) & (result['coll_type'] != 'неколлектор')]
    
    return result

# Функции кластеризации (из оригинального приложения)
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
    
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    
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

def format_display_label(row: pd.Series, id_value: str) -> str:
    parts = [str(id_value)]
    if "well" in row.index and pd.notna(row["well"]) and str(row["well"]).strip() != "":
        parts.append(str(row["well"]))
    if "Q" in row.index and pd.notna(row["Q"]):
        try:
            q = float(row["Q"])
            parts.append(f"Q={q:.3g}")
        except Exception:
            parts.append(f"Q={row['Q']}")
    return " | ".join(parts)

def make_display_labels(df: pd.DataFrame, indices: List[int], id_col: str) -> List[str]:
    out = []
    for i in indices:
        orig_id = str(df.iloc[i][id_col])
        out.append(format_display_label(df.iloc[i], orig_id))
    return out

# Основной интерфейс
st.title("🔬 Анализ скважин: LAS + Испытания")
st.caption("Загрузите LAS файлы и таблицу испытаний для анализа")

# Инициализация session state
if 'las_data' not in st.session_state:
    st.session_state.las_data = pd.DataFrame()
if 'core_data' not in st.session_state:
    st.session_state.core_data = pd.DataFrame()
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = pd.DataFrame()
if 'aggregated_data' not in st.session_state:
    st.session_state.aggregated_data = pd.DataFrame()
if 'selected_coll_column' not in st.session_state:
    st.session_state.selected_coll_column = None

# Создаем табы
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Загрузка данных", "🔗 Объединение", "🚀 Бустинг", "📊 Агрегация", "📄 Отчёты"])

with tab1:
    st.header("Загрузка данных")
    
    # Кнопка полной очистки всех данных
    if st.button("🗑️ Очистить все данные", help="Удалить все загруженные данные и начать заново"):
        st.session_state.las_data = pd.DataFrame()
        st.session_state.core_data = pd.DataFrame()
        st.session_state.merged_data = pd.DataFrame()
        st.session_state.aggregated_data = pd.DataFrame()
        st.session_state.selected_coll_column = None
        st.success("✅ Все данные очищены")
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. LAS файлы")
        if not LASIO_AVAILABLE:
            st.error("⚠️ lasio не установлен. Установите: pip install lasio")
        else:
            uploaded_las_files = st.file_uploader(
                "Выберите LAS файлы",
                type=['las'],
                accept_multiple_files=True,
                help="Загрузите один или несколько LAS файлов"
            )
            
            if uploaded_las_files:
                if st.button("📥 Загрузить LAS файлы"):
                    with st.spinner("Обработка LAS файлов..."):
                        st.session_state.las_data = process_las_files(uploaded_las_files)
                
                if not st.session_state.las_data.empty:
                    st.success(f"✅ LAS данные загружены: {len(st.session_state.las_data)} точек")
                    st.dataframe(st.session_state.las_data.head(), use_container_width=True)
                    
                    # Кнопка очистки LAS данных
                    if st.button("🗑️ Очистить LAS данные", help="Удалить все загруженные LAS данные"):
                        st.session_state.las_data = pd.DataFrame()
                        st.session_state.merged_data = pd.DataFrame()
                        st.session_state.aggregated_data = pd.DataFrame()
                        st.success("✅ LAS данные очищены")
                        st.rerun()
    
    with col2:
        st.subheader("2. Таблица испытаний")
        uploaded_core_file = st.file_uploader(
            "Выберите файл с испытаниями",
            type=['xlsx', 'xls', 'csv'],
            help="Excel или CSV файл с данными испытаний"
        )
        
        if uploaded_core_file:
            if st.button("📥 Загрузить таблицу испытаний"):
                with st.spinner("Обработка таблицы испытаний..."):
                    try:
                        if uploaded_core_file.name.endswith(('.xlsx', '.xls')):
                            st.session_state.core_data = pd.read_excel(uploaded_core_file, engine="openpyxl")
                        else:
                            st.session_state.core_data = pd.read_csv(uploaded_core_file)
                        
                        # Преобразуем названия скважин в строковый тип
                        if 'well' in st.session_state.core_data.columns:
                            st.session_state.core_data['well'] = st.session_state.core_data['well'].astype(str)
                        
                        st.success(f"✅ Таблица испытаний загружена: {len(st.session_state.core_data)} записей")
                    except Exception as e:
                        st.error(f"❌ Ошибка при загрузке: {str(e)}")
            
            if not st.session_state.core_data.empty:
                st.success(f"✅ Данные испытаний загружены: {len(st.session_state.core_data)} записей")
                st.dataframe(st.session_state.core_data.head(), use_container_width=True)
                
                # Кнопка очистки данных испытаний
                if st.button("🗑️ Очистить данные испытаний", help="Удалить загруженные данные испытаний"):
                    st.session_state.core_data = pd.DataFrame()
                    st.session_state.merged_data = pd.DataFrame()
                    st.session_state.aggregated_data = pd.DataFrame()
                    st.success("✅ Данные испытаний очищены")
                    st.rerun()

with tab2:
    st.header("Объединение данных")
    
    if st.session_state.las_data.empty or st.session_state.core_data.empty:
        st.warning("⚠️ Сначала загрузите LAS файлы и таблицу испытаний")
    else:
        st.subheader("Настройки объединения")
        
        # Выбор колонок для переноса
        available_columns = [col for col in st.session_state.core_data.columns 
                           if col not in ['well', 'top', 'bottom']]
        
        selected_columns = st.multiselect(
            "Выберите колонки для переноса из таблицы испытаний в LAS данные:",
            options=available_columns,
            default=available_columns[:5] if len(available_columns) > 5 else available_columns
        )
        
        if st.button("🔗 Объединить данные"):
            with st.spinner("Объединение данных..."):
                try:
                    st.session_state.merged_data = thick_to_dots(
                        st.session_state.las_data,
                        st.session_state.core_data,
                        selected_columns
                    )
                    st.success(f"✅ Данные объединены: {len(st.session_state.merged_data)} точек")
                except Exception as e:
                    st.error(f"❌ Ошибка при объединении: {str(e)}")
        
        if not st.session_state.merged_data.empty:
            st.subheader("Объединенные данные")
            st.dataframe(st.session_state.merged_data.head(), use_container_width=True)
            
            # Расчет group_number
            st.subheader("Расчет group_number")
            st.caption("group_number рассчитывается на основе изменений в колонках 'coll' и 'well'")
            
            # Проверяем наличие необходимых колонок
            missing_cols = []
            if 'well' not in st.session_state.merged_data.columns:
                missing_cols.append('well')
            
            if missing_cols:
                st.warning(f"⚠️ Для расчета group_number нужны колонки: {', '.join(missing_cols)}")
            else:
                # Выбор колонки для группировки
                available_coll_columns = [col for col in st.session_state.merged_data.columns 
                                        if col not in ['well', 'DEPTH', 'group_number']]
                
                if 'coll' in available_coll_columns:
                    default_coll_col = 'coll'
                else:
                    default_coll_col = available_coll_columns[0] if available_coll_columns else None
                
                if default_coll_col:
                    coll_column = st.selectbox(
                        "Выберите колонку для группировки (аналог 'coll'):",
                        options=available_coll_columns,
                        index=available_coll_columns.index(default_coll_col) if default_coll_col in available_coll_columns else 0
                    )
                    # Сохраняем выбранную колонку в session_state
                    st.session_state.selected_coll_column = coll_column
                    
                    if 'group_number' not in st.session_state.merged_data.columns:
                        if st.button("🔄 Рассчитать group_number"):
                            with st.spinner("Расчет group_number..."):
                                try:
                                    st.session_state.merged_data = calculate_group_number(st.session_state.merged_data, coll_column)
                                    st.success("✅ group_number рассчитан!")
                                except Exception as e:
                                    st.error(f"❌ Ошибка при расчете group_number: {str(e)}")
                    else:
                        st.success("✅ group_number уже рассчитан")
                        if st.button("🔄 Пересчитать group_number"):
                            with st.spinner("Пересчет group_number..."):
                                try:
                                    # Удаляем старый group_number
                                    st.session_state.merged_data = st.session_state.merged_data.drop('group_number', axis=1)
                                    st.session_state.merged_data = calculate_group_number(st.session_state.merged_data, coll_column)
                                    st.success("✅ group_number пересчитан!")
                                except Exception as e:
                                    st.error(f"❌ Ошибка при пересчете group_number: {str(e)}")
                else:
                    st.warning("⚠️ Нет доступных колонок для группировки")
            
            # Статистика по скважинам
            st.subheader("Статистика по скважинам")
            well_stats = st.session_state.merged_data.groupby('well').agg({
                'DEPTH': ['count', 'min', 'max'],
                **{col: 'count' for col in selected_columns if col in st.session_state.merged_data.columns}
            }).round(2)
            st.dataframe(well_stats, use_container_width=True)
            
            # Статистика по группам
            if 'group_number' in st.session_state.merged_data.columns:
                st.subheader("Статистика по группам")
                
                # Используем ту же колонку, которую выбрал пользователь для группировки
                if (st.session_state.selected_coll_column and 
                    st.session_state.selected_coll_column in st.session_state.merged_data.columns):
                    coll_col_for_stats = st.session_state.selected_coll_column
                else:
                    # Fallback: определяем колонку автоматически
                    available_coll_columns = [col for col in st.session_state.merged_data.columns 
                                            if col not in ['well', 'DEPTH', 'group_number']]
                    
                    if 'coll' in available_coll_columns:
                        coll_col_for_stats = 'coll'
                    elif 'COLL' in available_coll_columns:
                        coll_col_for_stats = 'COLL'
                    elif available_coll_columns:
                        coll_col_for_stats = available_coll_columns[0]
                    else:
                        coll_col_for_stats = None
                
                if coll_col_for_stats:
                    group_stats = st.session_state.merged_data.groupby('group_number').agg({
                        'well': 'first',
                        coll_col_for_stats: 'first',
                        'DEPTH': ['count', 'min', 'max']
                    }).round(2)
                    st.dataframe(group_stats.head(10), use_container_width=True)
                else:
                    # Если нет подходящей колонки, показываем только базовую статистику
                    group_stats = st.session_state.merged_data.groupby('group_number').agg({
                        'well': 'first',
                        'DEPTH': ['count', 'min', 'max']
                    }).round(2)
                    st.dataframe(group_stats.head(10), use_container_width=True)

with tab3:
    st.header("Агломеративный бустинг")
    
    if not BOOSTING_AVAILABLE:
        st.error("❌ Модуль агломеративного бустинга недоступен")
        st.info("Убедитесь, что файл agglomerative_boosting.py находится в той же папке")
    else:
        create_boosting_interface()

with tab4:
    st.header("Агрегация в коллекторы")
    
    if st.session_state.merged_data.empty:
        st.warning("⚠️ Сначала объедините данные")
    else:
        st.subheader("Настройки агрегации")
        
        # Выбор колонки для группировки
        available_group_columns = [col for col in st.session_state.merged_data.columns 
                                 if col not in ['well', 'DEPTH', 'group_number']]
        
        # Определяем колонку по умолчанию
        default_group_col = None
        for preferred_col in ['коллектор', 'coll', 'COLL', 'coll_type']:
            if preferred_col in available_group_columns:
                default_group_col = preferred_col
                break
        
        if not default_group_col and available_group_columns:
            default_group_col = available_group_columns[0]
        
        if default_group_col:
            group_column = st.selectbox(
                "Выберите признак для группировки (расчет group_number):",
                options=available_group_columns,
                index=available_group_columns.index(default_group_col) if default_group_col in available_group_columns else 0,
                help="По этому признаку будут группироваться точки в коллекторы"
            )
        else:
            st.warning("⚠️ Нет доступных колонок для группировки")
            group_column = None
        
        # Настройка признаков для агрегации
        st.subheader("Признаки для агрегации")
        
        # Получаем все числовые колонки
        numeric_columns = [col for col in st.session_state.merged_data.columns 
                          if pd.api.types.is_numeric_dtype(st.session_state.merged_data[col]) 
                          and col not in ['well', 'DEPTH', 'group_number', 'top', 'bottom', 'h']]
        
        # Автоматически определяем LAS кривые из данных
        # Исключаем служебные колонки (но включаем COLL)
        service_columns = ['well', 'DEPTH', 'group_number', 'top', 'bottom', 'h', 'coll', 'коллектор']
        
        # Все числовые колонки, которые не являются служебными (включая COLL)
        default_las_curves = [col for col in numeric_columns if col not in service_columns]
        
        # Категориальные колонки для модального значения
        categorical_columns = [col for col in st.session_state.merged_data.columns 
                              if not pd.api.types.is_numeric_dtype(st.session_state.merged_data[col]) 
                              and col not in ['well', 'group_number']]
        
        # Выбор признаков для агрегации
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**LAS кривые для усреднения:**")
            selected_las_curves = st.multiselect(
                "Выберите LAS кривые:",
                options=default_las_curves,
                default=default_las_curves,
                help="Эти кривые будут усреднены по коллекторам"
            )
        
        with col2:
            st.write("**Категориальные признаки:**")
            selected_categorical = st.multiselect(
                "Выберите категориальные признаки:",
                options=categorical_columns,
                default=[],  # По умолчанию ничего не выбрано
                help="Для этих признаков будет взято модальное значение"
            )
        
        # Показываем информацию о найденных кривых
        st.subheader("📊 Информация о найденных кривых")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("LAS кривые", len(default_las_curves))
            if default_las_curves:
                st.caption(f"Найдено: {', '.join(default_las_curves[:5])}{'...' if len(default_las_curves) > 5 else ''}")
        
        with col2:
            st.metric("Категориальные", len(categorical_columns))
            if categorical_columns:
                st.caption(f"Найдено: {', '.join(categorical_columns[:3])}{'...' if len(categorical_columns) > 3 else ''}")
        
        # Сохраняем выбранные признаки в session_state
        st.session_state.selected_las_curves = selected_las_curves
        st.session_state.selected_categorical = selected_categorical
        
        if st.button("📊 Создать таблицу коллекторов"):
            with st.spinner("Агрегация данных..."):
                try:
                    st.session_state.aggregated_data = aggregate_to_collectors(
                        st.session_state.merged_data, 
                        group_column
                    )
                    st.success(f"✅ Таблица коллекторов создана: {len(st.session_state.aggregated_data)} коллекторов")
                except Exception as e:
                    st.error(f"❌ Ошибка при агрегации: {str(e)}")
        
        if not st.session_state.aggregated_data.empty:
            st.subheader("Таблица коллекторов")
            
            # Фильтрация после агрегации
            st.subheader("🔍 Фильтрация коллекторов")
            
            # Показываем общую статистику
            total_collectors = len(st.session_state.aggregated_data)
            st.info(f"📊 Всего коллекторов после агрегации: {total_collectors}")
            
            # Фильтр по h < 0.4
            if 'h' in st.session_state.aggregated_data.columns:
                initial_count = len(st.session_state.aggregated_data)
                filtered_data = st.session_state.aggregated_data[st.session_state.aggregated_data['h'] >= 0.4]
                filtered_count = len(filtered_data)
                
                if initial_count != filtered_count:
                    st.info(f"🔍 Отфильтровано {initial_count - filtered_count} коллекторов с h < 0.4. Осталось {filtered_count} коллекторов.")
                    display_data = filtered_data
                else:
                    st.info("✅ Все коллекторы имеют h >= 0.4")
                    display_data = st.session_state.aggregated_data
            else:
                st.warning("⚠️ Колонка 'h' не найдена в таблице коллекторов")
                display_data = st.session_state.aggregated_data
            
            # Фильтр по значению группировочного признака
            if group_column and group_column in display_data.columns:
                st.write(f"**Фильтр по {group_column}:**")
                
                # Получаем уникальные значения
                unique_values = display_data[group_column].unique()
                unique_values = sorted([v for v in unique_values if pd.notna(v)])
                
                if unique_values:
                    # Выбор значений для фильтрации
                    selected_values = st.multiselect(
                        f"Выберите значения {group_column} для отображения:",
                        options=unique_values,
                        default=unique_values,
                        help=f"Показать только коллекторы с выбранными значениями {group_column}"
                    )
                    
                    if selected_values:
                        display_data = display_data[display_data[group_column].isin(selected_values)]
                        st.info(f"📊 Показано {len(display_data)} коллекторов с {group_column} в {selected_values}")
                        
                        # Показываем распределение по значениям
                        if len(selected_values) > 1:
                            value_counts = display_data[group_column].value_counts()
                            st.write("**Распределение по значениям:**")
                            for value, count in value_counts.items():
                                st.write(f"- {value}: {count} коллекторов")
                    else:
                        st.warning("⚠️ Не выбрано ни одного значения для отображения")
                        display_data = pd.DataFrame()  # Пустая таблица
                else:
                    st.warning(f"⚠️ В колонке {group_column} нет значений для фильтрации")
            else:
                st.info("ℹ️ Группировочный признак не найден, фильтрация недоступна")
            
            # Отображение отфильтрованной таблицы
            if not display_data.empty:
                st.dataframe(display_data, use_container_width=True)
                
                # Экспорт отфильтрованных данных
            csv_buf = io.StringIO()
            display_data.to_csv(csv_buf, index=False, encoding="utf-8")
            st.download_button(
                    "📥 Скачать отфильтрованную таблицу коллекторов (CSV)",
                csv_buf.getvalue().encode("utf-8"),
                    "filtered_collectors_table.csv",
                "text/csv"
                )
        else:
                st.warning("⚠️ Нет данных для отображения после фильтрации")
            
            # Экспорт полной таблицы (без фильтров)
        st.subheader("📥 Экспорт данных")
            csv_buf = io.StringIO()
            st.session_state.aggregated_data.to_csv(csv_buf, index=False, encoding="utf-8")
            st.download_button(
                "📥 Скачать полную таблицу коллекторов (CSV)",
                csv_buf.getvalue().encode("utf-8"),
                "full_collectors_table.csv",
                "text/csv",
                help="Скачать все коллекторы без фильтрации"
            )

with tab4:
    st.header("Кластеризация")
    
    if st.session_state.aggregated_data.empty:
        st.warning("⚠️ Сначала создайте таблицу коллекторов")
    else:
        df = st.session_state.aggregated_data.copy()
        
        # Проверяем наличие group_number
        if "group_number" not in df.columns:
            st.error("❌ Колонка 'group_number' не найдена в данных")
        else:
            id_col = "group_number"
            
            # Признаки
            num_all, cat_all = get_numeric_and_categorical_columns(df, id_col)
            if not num_all and not cat_all:
                st.error("❌ Нет признаков, кроме идентификатора")
            else:
                # Определяем LAS кривые автоматически
                # LAS кривые обычно имеют стандартные названия
                las_curves = [
                    "GR", "GK", "GK_NORM", "NK", "DTP", "DTP_NORM", "DT", "RHOB", "NPHI", 
                    "SP", "LLD", "LLS", "MSFL", "BK", "log_BK", "log_BK_NORM", "RT", "RXO",
                    "CALI", "CGR", "PEF", "DRHO", "LITH", "LITHO", "LITHOLOGY"
                ]
                
                # Исключаем служебные колонки из числовых признаков
                service_columns = ['top', 'bottom', 'h', 'group_number', 'well']
                filtered_num_all = [c for c in num_all if c not in service_columns]
                
                # Находим LAS кривые в отфильтрованных данных
                available_las_curves = [c for c in filtered_num_all if c in las_curves or 
                                      any(las_name in c.upper() for las_name in ["GR", "GK", "NK", "DTP", "DT", "RHOB", "NPHI", "SP", "LLD", "LLS", "MSFL", "BK", "RT", "RXO", "CALI", "CGR", "PEF", "DRHO"])]
                
                # Если есть LAS кривые, используем их по умолчанию, иначе используем предпочтительные
                if available_las_curves:
                    default_nums = available_las_curves
                    st.info(f"🔍 Найдены LAS кривые: {', '.join(available_las_curves)}")
                else:
                    preferred_numeric = [
                        "GK_NORM", "NK", "DTP_NORM", "log_BK_NORM", "frac_rf",
                        "dis_frac_rfn", "por_rf", "kvo_rf", "SAT_rf",
                        "log10Kpr_tim", "log10Kpr_rf",
                    ]
                    default_nums = [c for c in preferred_numeric if c in filtered_num_all] or filtered_num_all
                
                left, mid, right = st.columns(3)
                with left:
                    sel_num = st.multiselect("Числовые признаки", options=num_all, default=default_nums, key="boosting_numeric")
                with mid:
                    sel_cat = st.multiselect("Категориальные признаки", options=cat_all, default=[], key="boosting_categorical")
                with right:
                    standardize = st.checkbox("Стандартизация числовых", value=True, key="boosting_standardize")
                    metric = st.selectbox("Метрика", options=["euclidean", "cityblock", "cosine"], index=0, key="boosting_metric")
                    linkage_method = st.selectbox("Линковка", options=["ward", "average", "complete", "single"], index=0, key="boosting_linkage")
                
                if len(sel_num) + len(sel_cat) == 0:
                    st.warning("⚠️ Выберите хотя бы один признак")
                else:
                    # Очистка по идентификатору
                    df = df.dropna(subset=[id_col]).reset_index(drop=True)
                    
                    # Векторизация
                    X, labels = compute_features(df, id_col, sel_num, sel_cat, standardize)
                    
                    st.subheader("Сходство для выбранного идентификатора")
                    
                    # Улучшенный выбор целевого коллектора
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Выбор по скважине
                        if 'well' in df.columns:
                            available_wells = df['well'].unique()
                            selected_well = st.selectbox("Выберите скважину:", options=available_wells)
                            
                            # Фильтруем по выбранной скважине
                            well_filtered_df = df[df['well'] == selected_well]
                            well_labels = well_filtered_df[id_col].unique()
                        else:
                            well_labels = labels
                            selected_well = None
                    
                    with col2:
                        # Выбор по глубине (если есть колонки top/bottom)
                        if 'top' in df.columns and 'bottom' in df.columns:
                            if selected_well:
                                depth_options = well_filtered_df[['top', 'bottom']].dropna()
                            else:
                                depth_options = df[['top', 'bottom']].dropna()
                            
                            if not depth_options.empty:
                                min_depth = depth_options['top'].min()
                                max_depth = depth_options['bottom'].max()
                                
                                depth_range = st.slider(
                                    "Диапазон глубин:",
                                    min_value=float(min_depth),
                                    max_value=float(max_depth),
                                    value=(float(min_depth), float(max_depth)),
                                    step=0.1
                                )
                                
                                # Фильтруем по глубине
                                if selected_well:
                                    depth_filtered_df = well_filtered_df[
                                        (well_filtered_df['top'] >= depth_range[0]) & 
                                        (well_filtered_df['bottom'] <= depth_range[1])
                                    ]
                                else:
                                    depth_filtered_df = df[
                                        (df['top'] >= depth_range[0]) & 
                                        (df['bottom'] <= depth_range[1])
                                    ]
                                
                                depth_labels = depth_filtered_df[id_col].unique()
                            else:
                                depth_labels = well_labels
                        else:
                            depth_labels = well_labels
                    
                    # Финальный выбор идентификатора
                    if len(depth_labels) > 0:
                        target_label = st.selectbox("Значение идентификатора (group_number):", options=depth_labels)
                    idx_by_label = {l: i for i, l in enumerate(labels)}
                    target_idx = idx_by_label[target_label]
                    
                        # Показываем информацию о выбранном коллекторе
                        target_info = df[df[id_col] == target_label].iloc[0]
                        st.info(f"🎯 Выбран коллектор {target_label}: скважина {target_info.get('well', 'N/A')}, "
                               f"глубина {target_info.get('top', 'N/A')}-{target_info.get('bottom', 'N/A')} м")
                    else:
                        st.error("❌ Нет коллекторов, соответствующих выбранным критериям")
                        target_label = None
                        target_idx = None
                    
                    # Топ-10 похожих (только если выбран целевой коллектор)
                    if target_label is not None:
                    neighbor_labels, neighbor_dists = compute_top_k_similar(X, labels, target_label, metric, 10)
                    neighbor_indices = [idx_by_label[l] for l in neighbor_labels]
                    
                    # Таблица похожих
                    extra_cols = [c for c in ["well", "top", "bottom", 'BF', "TEST", "TYPE"] if c in df.columns]
                    neighbors_extra = df.loc[neighbor_indices, extra_cols] if extra_cols else pd.DataFrame(index=neighbor_indices)
                    neighbors_df = pd.DataFrame({id_col: neighbor_labels, "distance": neighbor_dists})
                    if not neighbors_extra.empty:
                        neighbors_df = pd.concat([neighbors_df.reset_index(drop=True), neighbors_extra.reset_index(drop=True)], axis=1)
                    if "well" in neighbors_df.columns:
                        neighbors_df = neighbors_df[[id_col, "well"] + [c for c in neighbors_df.columns if c not in [id_col, "well"]]]
                    neighbors_df = neighbors_df.sort_values("distance", ascending=True, ignore_index=True)
                    
                    st.write("10 наиболее похожих:")
                    st.dataframe(neighbors_df, use_container_width=True)
                    
                    # Экспорт топ-10
                    buf = io.StringIO()
                    neighbors_df.to_csv(buf, index=False, encoding="utf-8")
                    st.download_button("📥 Скачать топ-10 (CSV)", buf.getvalue().encode("utf-8"), "top10_neighbors.csv", "text/csv")
                        
                        # Дендрограмма и PCA для похожих коллекторов (по образцу best_well_hc.py)
                        st.subheader("📊 Визуализация схожести")
                        
                        # Подготавливаем данные для визуализации
                        subset_indices = [target_idx] + neighbor_indices
                        subset_labels = [target_label] + neighbor_labels
                        X_subset = X[subset_indices, :]
                        
                        # Создаем подписи в формате: group_number | well | Q (как в best_well_hc.py)
                        def format_display_label(row, id_value):
                            parts = [str(id_value)]
                            if "well" in row.index and pd.notna(row["well"]) and str(row["well"]).strip() != "":
                                parts.append(str(row["well"]))
                            if "Q" in row.index and pd.notna(row["Q"]):
                                try:
                                    q = float(row["Q"])
                                    parts.append(f"Q={q:.3g}")
                                except Exception:
                                    parts.append(f"Q={row['Q']}")
                            return " | ".join(parts)
                        
                        def make_display_labels(indices):
                            out = []
                            for i in indices:
                                orig_id = str(df.iloc[i][id_col])
                                out.append(format_display_label(df.iloc[i], orig_id))
                            return out
                        
                        subset_display_labels = make_display_labels(subset_indices)
                        target_display_label = subset_display_labels[0]
                        
                        # Дендрограмма и PCA в двух колонках
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            # Дендрограмма (по образцу best_well_hc.py)
                            try:
                                metric_for_linkage = "euclidean" if linkage_method == "ward" else metric
                                Z = linkage(X_subset, method=linkage_method, metric=metric_for_linkage)
                                
                                fig_dendro, ax_dendro = plt.subplots(figsize=(9, 7), dpi=150)
                                dendrogram(Z, labels=subset_display_labels, leaf_rotation=90, leaf_font_size=12, ax=ax_dendro)
                                ax_dendro.set_title("Дендрограмма (целевой + 10 похожих)")
                                ax_dendro.set_ylabel("Дистанция")
                                
                                # Выделяем целевой коллектор красным цветом
                                for tick in ax_dendro.get_xmajorticklabels():
                                    if tick.get_text() == target_display_label:
                                        tick.set_color("crimson")
                                        tick.set_fontweight("bold")
                                    else:
                                        tick.set_color("black")
                                
                                fig_dendro.tight_layout()
                                st.pyplot(fig_dendro, use_container_width=True)
                                
                                # Экспорт дендрограммы
                                png_buf = io.BytesIO()
                                fig_dendro.savefig(png_buf, format="png", bbox_inches="tight", dpi=200)
                                st.download_button("Скачать дендрограмму (PNG)", png_buf.getvalue(), "dendrogram_top10.png", "image/png")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Ошибка при создании дендрограммы: {str(e)}")
                    
                    with col_right:
                            # PCA биплот (по образцу best_well_hc.py)
                            try:
                                if len(sel_num) < 2:
                                    raise ValueError("Для PCA нужно минимум 2 числовых признака.")
                                
                                from sklearn.decomposition import PCA
                                from sklearn.impute import SimpleImputer
                                
                                # Подготавливаем числовые данные
                                num_imputer = SimpleImputer(strategy="median")
                                scaler = StandardScaler() if standardize else None
                                
                                X_num_full = num_imputer.fit_transform(df[sel_num])
                                if scaler:
                                    X_num_full = scaler.fit_transform(X_num_full)
                                X_num_subset = X_num_full[subset_indices, :]
                                
                                if np.allclose(X_num_subset.std(axis=0), 0):
                                    st.warning("Недостаточная вариативность для PCA.")
                                else:
                                    pca = PCA(n_components=2, random_state=0)
                                    scores = pca.fit_transform(X_num_subset)
                                    loadings = pca.components_.T
                                    evr = pca.explained_variance_ratio_
                                    
                                    fig_pca, ax_pca = plt.subplots(figsize=(5, 6), dpi=150)
                                    
                                    # Точки: похожие (серые) и целевой (красная звезда)
                                    ax_pca.scatter(scores[1:, 0], scores[1:, 1], c="gray", s=60, edgecolors="k", alpha=0.9, label="Похожие")
                                    ax_pca.scatter(scores[0, 0], scores[0, 1], c="crimson", s=100, edgecolors="k", marker="*", label="Целевой")
                                    
                                    # Подписи точек
                                    for i, (x, y) in enumerate(scores):
                                        ax_pca.text(x, y, subset_display_labels[i],
                                                  fontsize=9 if i == 0 else 8, ha="left", va="bottom",
                                                  color="crimson" if i == 0 else "black",
                                                  fontweight="bold" if i == 0 else "normal")
                                    
                                    # Стрелки нагрузок (если есть числовые признаки)
                                    if len(sel_num) > 0:
                                        arrow_scale = np.max(np.linalg.norm(scores, axis=1)) * 1.3
                                        for j, feature in enumerate(sel_num):
                                            vx, vy = loadings[j, 0] * arrow_scale, loadings[j, 1] * arrow_scale
                                            ax_pca.arrow(0, 0, vx, vy, color="tab:blue",
                                                       width=0.0006, head_width=0.012, head_length=0.02,
                                                       length_includes_head=True, alpha=0.9)
                                            ax_pca.text(vx, vy, feature, fontsize=8, color="tab:blue", ha="center", va="center")
                                    
                                    # Сетка и оси
                                    ax_pca.axhline(0, color="lightgray", linewidth=1)
                                    ax_pca.axvline(0, color="lightgray", linewidth=1)
                                    ax_pca.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
                                    ax_pca.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
                                    ax_pca.legend(fontsize=8)
                                    ax_pca.set_title("PCA биплот (целевой + 10 похожих)")
                                    
                                    fig_pca.tight_layout()
                                    st.pyplot(fig_pca, use_container_width=True)
                                    
                                    # Информация о PCA
                                    st.info(f"📊 PCA объясняет {pca.explained_variance_ratio_.sum()*100:.1f}% общей дисперсии")
                                    
                            except Exception as e:
                                st.warning(f"⚠️ Ошибка при создании PCA: {str(e)}")
                    st.warning("⚠️ Выберите целевой коллектор для анализа схожести")

with tab5:
    st.header("📄 Генерация отчётов")
    
    if st.session_state.aggregated_data.empty:
        st.warning("⚠️ Сначала создайте таблицу коллекторов")
    else:
        df = st.session_state.aggregated_data.copy()
        
        if "group_number" not in df.columns:
            st.error("❌ Колонка 'group_number' не найдена")
        else:
            # Настройки для отчёта
            st.subheader("Настройки анализа")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Выбор числовых признаков
                numeric_columns = [col for col in df.columns 
                                 if pd.api.types.is_numeric_dtype(df[col]) 
                                 and col not in ['group_number', 'top', 'bottom', 'h']]
                
                sel_num = st.multiselect(
                    "Числовые признаки",
                    options=numeric_columns,
                    default=numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns,
                    help="Выберите числовые признаки для анализа",
                    key="report_numeric"
                )
            
            with col2:
                # Выбор категориальных признаков
                categorical_columns = [col for col in df.columns 
                                     if not pd.api.types.is_numeric_dtype(df[col]) 
                                     and col not in ['group_number', 'well']]
                
                sel_cat = st.multiselect(
                    "Категориальные признаки",
                    options=categorical_columns,
                    default=[],
                    help="Категориальные признаки будут преобразованы в one-hot encoding",
                    key="report_categorical"
                )
            
            with col3:
                standardize = st.checkbox("Стандартизация числовых", value=True, key="report_standardize")
                metric = st.selectbox("Метрика расстояния", 
                                    options=["euclidean", "cityblock", "cosine"], 
                                    index=0, key="report_metric")
                linkage_method = st.selectbox("Метод связывания", 
                                            options=["ward", "average", "complete", "single"], 
                                            index=0, key="report_linkage")
                
                if linkage_method == "ward" and metric != "euclidean":
                    st.info("ℹ️ Ward требует евклидову метрику")
            
            if len(sel_num) + len(sel_cat) == 0:
                st.warning("⚠️ Выберите хотя бы один признак")
            else:
                # Фильтрация данных для отчёта: COLL == 1 и TEST != nan
                st.subheader("📋 Фильтрация данных для отчёта")
                
                # Проверяем наличие нужных колонок
                has_coll = 'COLL' in df.columns
                has_test = 'TEST' in df.columns
                
                if not has_coll:
                    st.error("❌ Колонка 'COLL' не найдена")
                elif not has_test:
                    st.error("❌ Колонка 'TEST' не найдена")
                else:
                    # Применяем фильтры
                    initial_count = len(df)
                    
                    # COLL == 1
                    coll_filtered = df[df['COLL'] == 1]
                    coll_count = len(coll_filtered)
                    
                    # TEST is not NaN
                    test_filtered = coll_filtered[~coll_filtered['TEST'].isna()]
                    final_count = len(test_filtered)
                    
                    st.info(f"📊 Исходно коллекторов: {initial_count}")
                    st.info(f"🔍 После фильтра COLL == 1: {coll_count}")
                    st.info(f"✅ После фильтра TEST не пустой: {final_count}")
                    
                    if final_count == 0:
                        st.warning("⚠️ Нет коллекторов, соответствующих критериям отчёта")
            else:
                        # Показываем отфильтрованные данные
                        st.write("**Отфильтрованные данные для анализа:**")
                        display_cols = ['group_number', 'well', 'top', 'bottom', 'h', 'COLL', 'TEST']
                        available_display_cols = [col for col in display_cols if col in test_filtered.columns]
                        st.dataframe(test_filtered[available_display_cols], use_container_width=True)
                        
                        # Кнопка генерации отчёта
                        if st.button("📄 Сгенерировать отчёт", type="primary"):
                            with st.spinner("Генерация отчёта..."):
                                try:
                                    # Создаём отчёт
                                    report_html = generate_clustering_report(
                                        test_filtered, sel_num, sel_cat, standardize, 
                                        metric, linkage_method, 'group_number'
                                    )
                                    
                                    # Показываем превью отчёта
                                    st.subheader("📄 Превью отчёта")
                                    st.components.v1.html(report_html, height=600, scrolling=True)
                                    
                                    # Кнопка скачивания
                                    st.download_button(
                                        "📥 Скачать отчёт (HTML)",
                                        report_html,
                                        "clustering_report.html",
                                        "text/html",
                                        help="Отчёт в формате HTML с графиками и таблицами"
                                    )
                                    
                                    st.success("✅ Отчёт успешно сгенерирован!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Ошибка при генерации отчёта: {str(e)}")
                                    st.exception(e)
