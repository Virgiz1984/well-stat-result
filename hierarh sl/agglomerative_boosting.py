import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt

# Импорт CatBoost
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    st.warning("⚠️ CatBoost не установлен. Установите: pip install catboost")

def create_boosting_interface():
    st.subheader('🚀 CatBoost для доинтерпретации коллекторов')
    st.caption('CatBoost для прямого поиска целевой переменной (например, COLL) без кластеризации')
    
    if 'merged_data' not in st.session_state or st.session_state.merged_data.empty:
        st.warning('⚠️ Сначала объедините данные в разделе "Объединение"')
        return
    
    df = st.session_state.merged_data.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Настройки')
        task_type = st.selectbox('Тип задачи', ['regression', 'classification'])
        n_estimators = 50  # Фиксированное значение
    
    with col2:
        st.subheader('Признаки')
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['group_number', 'top', 'bottom', 'h', 'DEPTH', 'depth', 'COLL']]
        
        if not numeric_columns:
            st.error('❌ Нет числовых признаков для обучения')
            return
        
        selected_features = st.multiselect('Выберите признаки', numeric_columns, default=numeric_columns[:5])
        
        # Показываем исключенные колонки
        excluded_columns = [col for col in df.columns if col in ['group_number', 'top', 'bottom', 'h', 'DEPTH', 'depth', 'COLL']]
        if excluded_columns:
            st.caption(f'🚫 Исключены из признаков: {", ".join(excluded_columns)}')
        
        # Определяем целевую переменную - приоритет COLL
        potential_targets = [col for col in numeric_columns if col not in selected_features]
        categorical_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and col not in ['group_number', 'well']]
        
        # Ищем колонку коллектора
        collector_columns = [col for col in categorical_columns if any(keyword in col.lower() for keyword in ['coll', 'коллектор', 'collector'])]
        
        # Создаем список опций для целевой переменной
        target_options = []
        default_index = 0
        
        # Добавляем COLL если есть
        if 'COLL' in df.columns:
            target_options.append('COLL')
            default_index = 0
        
        # Добавляем другие коллекторные колонки
        for col in collector_columns:
            if col != 'COLL' and col not in target_options:
                target_options.append(col)
        
        # Добавляем числовые переменные
        for col in potential_targets:
            if col not in target_options:
                target_options.append(col)
        
        if target_options:
            target_column = st.selectbox('Целевая переменная', target_options, index=default_index)
            if target_column == 'COLL':
                st.info("🎯 Выбрана колонка COLL (по умолчанию)")
            else:
                st.info(f"🎯 Выбрана целевая переменная: {target_column}")
        else:
            st.error('❌ Нет доступных целевых переменных')
            return
        
        # Проверяем наличие COLL и создаем COLL_advance
        if 'COLL' in df.columns:
            st.info("📊 Найдена колонка COLL. Будет создана колонка COLL_advance с предсказаниями для nan значений.")
            
            # Показываем статистику по COLL
            coll_stats = df['COLL'].value_counts(dropna=False)
            st.write("**Статистика по COLL:**")
            st.write(coll_stats)
            
            # Показываем количество nan значений
            nan_count = df['COLL'].isna().sum()
            total_count = len(df)
            st.write(f"**NaN значений в COLL:** {nan_count} из {total_count} ({nan_count/total_count*100:.1f}%)")
        else:
            st.warning("⚠️ Колонка COLL не найдена в данных")
        
    
    if st.button('🎯 Обучить модель'):
        if not selected_features:
            st.error('❌ Выберите признаки')
            return
        
        if not CATBOOST_AVAILABLE:
            st.error('❌ CatBoost не установлен. Установите: pip install catboost')
            return
        
        with st.spinner('Обучение CatBoost модели...'):
            try:
                # Специальная логика для COLL
                if target_column == 'COLL' and 'COLL' in df.columns:
                    # Обучаемся только на не-nan значениях COLL
                    train_mask = df['COLL'].notna()
                    train_data = df[train_mask].copy()
                    
                    if len(train_data) == 0:
                        st.error('❌ Нет данных для обучения - все значения COLL равны NaN')
                        return
                    
                    st.info(f'📊 Обучаемся на {len(train_data)} точках с известными значениями COLL')
                    
                    # Подготовка данных для обучения
                    X_train = train_data[selected_features].fillna(train_data[selected_features].median())
                    y_train = train_data['COLL']
                    
                    # Проверка на уникальность значений целевой переменной
                    unique_values = y_train.nunique()
                    if unique_values < 2:
                        st.warning(f'⚠️ COLL содержит только одно значение: {y_train.iloc[0]}. Используем методы поиска аналогий.')
                        
                        # Используем методы поиска аналогий для одного значения
                        similarity_method = st.selectbox(
                            'Выберите метод поиска аналогий:',
                            ['Косинусное сходство', 'Евклидово расстояние', 'Манхэттенское расстояние', 'Корреляция Пирсона', 'One-Class SVM', 'Изоляционный лес']
                        )
                        
                        # Показываем рекомендации по выбору метода
                        st.info("""
                        **💡 Рекомендации по выбору метода:**
                        - **Косинусное сходство**: Лучше для высокоразмерных данных, не зависит от масштаба
                        - **Евклидово расстояние**: Классический метод, хорошо для нормализованных данных
                        - **Манхэттенское расстояние**: Устойчив к выбросам, лучше для категориальных признаков
                        - **Корреляция Пирсона**: Показывает линейную зависимость между признаками
                        - **One-Class SVM**: Находит границы "нормальных" объектов, хорошо для аномалий
                        - **Изоляционный лес**: Эффективен для многомерных данных, быстро работает
                        """)
                        
                        # Подготовка всех данных для предсказания
                        X_all = df[selected_features].fillna(df[selected_features].median())
                        y_all = df['COLL']
                        
                        # Нормализация данных
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_all_scaled = scaler.transform(X_all)
                        
                        # Вычисляем схожесть для каждой точки с известными значениями COLL
                        nan_mask = df['COLL'].isna()
                        predictions = np.zeros(len(X_all_scaled))
                        
                        if similarity_method == 'Косинусное сходство':
                            from sklearn.metrics.pairwise import cosine_similarity
                            similarities = cosine_similarity(X_all_scaled[nan_mask], X_train_scaled)
                            # Берем максимальную схожесть
                            max_similarities = np.max(similarities, axis=1)
                            predictions[nan_mask] = max_similarities
                            
                        elif similarity_method == 'Евклидово расстояние':
                            from sklearn.metrics.pairwise import euclidean_distances
                            distances = euclidean_distances(X_all_scaled[nan_mask], X_train_scaled)
                            # Берем минимальное расстояние (максимальная схожесть)
                            min_distances = np.min(distances, axis=1)
                            predictions[nan_mask] = 1 / (1 + min_distances)  # Преобразуем в схожесть
                            
                        elif similarity_method == 'Манхэттенское расстояние':
                            from sklearn.metrics.pairwise import manhattan_distances
                            distances = manhattan_distances(X_all_scaled[nan_mask], X_train_scaled)
                            min_distances = np.min(distances, axis=1)
                            predictions[nan_mask] = 1 / (1 + min_distances)
                            
                        elif similarity_method == 'Корреляция Пирсона':
                            from scipy.stats import pearsonr
                            correlations = []
                            for i in range(len(X_all_scaled[nan_mask])):
                                max_corr = 0
                                for j in range(len(X_train_scaled)):
                                    corr, _ = pearsonr(X_all_scaled[nan_mask][i], X_train_scaled[j])
                                    if not np.isnan(corr):
                                        max_corr = max(max_corr, abs(corr))
                                correlations.append(max_corr)
                            predictions[nan_mask] = correlations
                            
                        elif similarity_method == 'One-Class SVM':
                            from sklearn.svm import OneClassSVM
                            # Обучаем One-Class SVM на известных значениях COLL
                            ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
                            ocsvm.fit(X_train_scaled)
                            
                            # Получаем оценки схожести (расстояние до границы решения)
                            decision_scores = ocsvm.decision_function(X_all_scaled[nan_mask])
                            # Преобразуем в схожесть (чем больше, тем лучше)
                            predictions[nan_mask] = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-8)
                            
                        elif similarity_method == 'Изоляционный лес':
                            from sklearn.ensemble import IsolationForest
                            # Обучаем Isolation Forest на известных значениях COLL
                            iso_forest = IsolationForest(contamination=0.1, random_state=42)
                            iso_forest.fit(X_train_scaled)
                            
                            # Получаем оценки аномальности (чем меньше, тем более аномальная точка)
                            anomaly_scores = iso_forest.decision_function(X_all_scaled[nan_mask])
                            # Преобразуем в схожесть (чем больше, тем лучше)
                            predictions[nan_mask] = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
                        
                        # Создаем колонку Cov со схожестью
                        df['Cov'] = 0.0  # Инициализируем нулями
                        df.loc[nan_mask, 'Cov'] = predictions[nan_mask]
                        
                        # Создаем COLL_advance по правилу: 1 если COLL=1 или Cov>0.98, иначе 0
                        coll_advance = np.zeros(len(df))
                        
                        # Случай 1: COLL = 1
                        coll_1_mask = df['COLL'] == 1
                        coll_advance[coll_1_mask] = 1
                        
                        # Случай 2: Cov > 0.98
                        cov_high_mask = df['Cov'] > 0.98
                        coll_advance[cov_high_mask] = 1
                        
                        df['COLL_advance'] = coll_advance
                        
                        # Показываем результаты
                        predicted_count = nan_mask.sum()
                        cov_high_count = (df['Cov'] > 0.98).sum()
                        coll_advance_1_count = (df['COLL_advance'] == 1).sum()
                        
                        st.success(f'🎯 Созданы колонки Cov и COLL_advance!')
                        st.info(f'📊 Использован метод: {similarity_method}')
                        st.info(f'📈 Статистика:')
                        st.info(f'   • Предсказано {predicted_count} значений схожести (Cov)')
                        st.info(f'   • {cov_high_count} точек с Cov > 0.98')
                        st.info(f'   • {coll_advance_1_count} точек с COLL_advance = 1')
                        
                        # Показываем статистику схожести
                        st.subheader('📊 Статистика схожести')
                        
                        # Создаем DataFrame с доступными колонками
                        similarity_data = {'Cov': df.loc[nan_mask, 'Cov']}
                        
                        # Добавляем колонки если они существуют
                        if 'group_number' in df.columns:
                            similarity_data['Группа'] = df.loc[nan_mask, 'group_number']
                        if 'well' in df.columns:
                            similarity_data['Скважина'] = df.loc[nan_mask, 'well']
                        if 'DEPTH' in df.columns:
                            similarity_data['Глубина'] = df.loc[nan_mask, 'DEPTH']
                        
                        similarity_stats = pd.DataFrame(similarity_data)
                        st.dataframe(similarity_stats.describe())
                        
                        # Показываем топ-10 самых похожих точек
                        st.subheader('🔍 Топ-10 самых похожих точек')
                        top_similar = similarity_stats.nlargest(10, 'Cov')
                        st.dataframe(top_similar)
                        
                        # Визуализация схожести
                        st.subheader('📈 Визуализация схожести')
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Гистограмма схожести
                        ax1.hist(df.loc[nan_mask, 'Cov'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                        ax1.axvline(x=0.98, color='red', linestyle='--', label='Порог 0.98')
                        ax1.set_xlabel('Cov (схожесть)')
                        ax1.set_ylabel('Количество точек')
                        ax1.set_title(f'Распределение Cov ({similarity_method})')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Box plot по группам или скважинам
                        if 'Группа' in similarity_stats.columns and len(similarity_stats['Группа'].unique()) > 1:
                            similarity_stats.boxplot(column='Cov', by='Группа', ax=ax2)
                            ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7)
                            ax2.set_title('Cov по группам')
                            ax2.set_xlabel('Группа')
                            ax2.set_ylabel('Cov')
                        elif 'Скважина' in similarity_stats.columns and len(similarity_stats['Скважина'].unique()) > 1:
                            similarity_stats.boxplot(column='Cov', by='Скважина', ax=ax2)
                            ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7)
                            ax2.set_title('Cov по скважинам')
                            ax2.set_xlabel('Скважина')
                            ax2.set_ylabel('Cov')
                        else:
                            # Показываем распределение по глубине если есть
                            if 'Глубина' in similarity_stats.columns:
                                ax2.scatter(similarity_stats['Глубина'], similarity_stats['Cov'], alpha=0.6)
                                ax2.axhline(y=0.98, color='red', linestyle='--', alpha=0.7, label='Порог 0.98')
                                ax2.set_xlabel('Глубина')
                                ax2.set_ylabel('Cov')
                                ax2.set_title('Cov vs Глубина')
                                ax2.legend()
                                ax2.grid(True, alpha=0.3)
                            else:
                                ax2.text(0.5, 0.5, 'Недостаточно данных\nдля группировки', 
                                       ha='center', va='center', transform=ax2.transAxes)
                                ax2.set_title('Дополнительная визуализация')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Показываем сравнение
                        st.subheader('📊 Сравнение COLL и COLL_advance')
                        
                        # Создаем таблицу сравнения с доступными колонками
                        comparison_columns = ['COLL', 'Cov', 'COLL_advance']
                        if 'group_number' in df.columns:
                            comparison_columns.insert(0, 'group_number')
                        if 'well' in df.columns:
                            comparison_columns.insert(1, 'well')
                        if 'DEPTH' in df.columns:
                            comparison_columns.insert(-3, 'DEPTH')
                        
                        comparison_df = df[comparison_columns].copy()
                        st.dataframe(comparison_df.head(20))
                        
                        # Показываем статистику по COLL_advance
                        st.subheader('📊 Статистика COLL_advance')
                        coll_advance_stats = df['COLL_advance'].value_counts()
                        st.write("**Распределение COLL_advance:**")
                        st.write(coll_advance_stats)
                        
                        # Показываем точки с Cov > 0.98
                        high_cov_points = df[df['Cov'] > 0.98]
                        if len(high_cov_points) > 0:
                            st.subheader('🎯 Точки с высокой схожестью (Cov > 0.98)')
                            high_cov_display = high_cov_points[comparison_columns].copy()
                            st.dataframe(high_cov_display)
                        
                        # Сохраняем результаты
                        st.session_state.merged_data = df
                        st.success('💾 Результаты сохранены в session_state.merged_data')
                        return
                    
                    st.info(f'📊 Целевая переменная "{target_column}" содержит {unique_values} уникальных значений: {list(y_train.unique())}')
                    
                    # Подготовка всех данных для предсказания
                    X_all = df[selected_features].fillna(df[selected_features].median())
                    y_all = df['COLL']
                    
                else:
                    # Обычная логика для других переменных
                    X_train = df[selected_features].fillna(df[selected_features].median())
                    y_train = df[target_column]
                    X_all = X_train
                    y_all = y_train
                    train_mask = pd.Series([True] * len(df), index=df.index)
                    
                    # Проверка на уникальность значений целевой переменной
                    unique_values = y_train.nunique()
                    if unique_values < 2:
                        st.error(f'❌ Целевая переменная "{target_column}" содержит только одно уникальное значение: {y_train.iloc[0]}. Нужно минимум 2 разных значения для обучения.')
                        st.info('💡 Попробуйте выбрать другую целевую переменную или проверьте данные.')
                        return
                    
                    st.info(f'📊 Целевая переменная "{target_column}" содержит {unique_values} уникальных значений: {list(y_train.unique())}')
                
                # Для категориальных переменных преобразуем в числовые
                if not pd.api.types.is_numeric_dtype(y_train):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train.astype(str))
                    y_original = y_train.copy()
                else:
                    y_train_encoded = y_train.fillna(y_train.median())
                    y_original = y_train.copy()
                
                # Простое обучение CatBoost без кластеризации
                if task_type == 'regression':
                    model = CatBoostRegressor(
                        iterations=n_estimators,
                        learning_rate=0.1,
                        depth=6,
                        random_seed=42,
                        verbose=False
                    )
                else:
                    model = CatBoostClassifier(
                        iterations=n_estimators,
                        learning_rate=0.1,
                        depth=6,
                        random_seed=42,
                        verbose=False
                    )
                
                # Обучаем модель на всех данных
                model.fit(X_train, y_train_encoded)
                st.success(f'✅ CatBoost модель обучена!')
                
                # Предсказания для всех данных
                predictions = model.predict(X_all)
                
                # Создаем COLL_advance
                if target_column == 'COLL' and 'COLL' in df.columns:
                    # COLL_advance = COLL где не nan, иначе предсказание
                    coll_advance = df['COLL'].copy()
                    nan_mask = df['COLL'].isna()
                    coll_advance[nan_mask] = predictions[nan_mask]
                    
                    # Добавляем в результаты
                    df['COLL_advance'] = coll_advance
                    
                    # Показываем статистику
                    predicted_count = nan_mask.sum()
                    st.success(f'🎯 Создана колонка COLL_advance! Предсказано {predicted_count} значений для NaN позиций.')
                    
                    # Показываем сравнение
                    st.subheader('📊 Сравнение COLL и COLL_advance')
                    comparison_df = df[['group_number', 'well', 'COLL', 'COLL_advance']].copy()
                    if 'DEPTH' in df.columns:
                        comparison_df['DEPTH'] = df['DEPTH']
                    st.dataframe(comparison_df.head(20))
                
                # Для классификации преобразуем предсказания обратно в оригинальные метки
                if not pd.api.types.is_numeric_dtype(y_original):
                    predictions_labels = le.inverse_transform(predictions.astype(int))
                else:
                    predictions_labels = predictions
                
                # Метрики только на обучающих данных
                if task_type == 'regression':
                    train_predictions = predictions[train_mask] if target_column == 'COLL' else predictions
                    r2 = r2_score(y_train_encoded, train_predictions)
                    st.success(f'✅ CatBoost модель обучена! R² = {r2:.3f} (на обучающих данных)')
                else:
                    train_predictions = predictions[train_mask] if target_column == 'COLL' else predictions
                    acc = accuracy_score(y_train_encoded, train_predictions)
                    st.success(f'✅ CatBoost модель обучена! Accuracy = {acc:.3f} (на обучающих данных)')
                
                # График
                fig, ax = plt.subplots(figsize=(8, 6))
                if task_type == 'regression':
                    # Показываем только обучающие данные на графике
                    train_predictions = predictions[train_mask] if target_column == 'COLL' else predictions
                    ax.scatter(y_train_encoded, train_predictions, alpha=0.6, color='blue', label='Обучающие данные')
                    ax.plot([y_train_encoded.min(), y_train_encoded.max()], [y_train_encoded.min(), y_train_encoded.max()], 'r--', label='Идеальная линия')
                    ax.set_xlabel('Факт')
                    ax.set_ylabel('Предсказание')
                    ax.set_title('CatBoost: Предсказания vs Факт (обучающие данные)')
                    ax.legend()
                else:
                    # Для классификации показываем распределение предсказаний
                    prediction_counts = pd.Series(predictions_labels).value_counts().sort_index()
                    ax.bar(range(len(prediction_counts)), prediction_counts.values, color='skyblue', alpha=0.7)
                    ax.set_xlabel('Предсказанные классы')
                    ax.set_ylabel('Количество объектов')
                    ax.set_title('CatBoost: Распределение предсказаний')
                    ax.set_xticks(range(len(prediction_counts)))
                    ax.set_xticklabels(prediction_counts.index, rotation=45)
                
                st.pyplot(fig)
                
                # Результаты
                results_df = df.copy()
                results_df['prediction'] = predictions_labels
                results_df['prediction_numeric'] = predictions
                
                # Показываем результаты
                display_columns = [target_column, 'prediction']
                if 'group_number' in results_df.columns:
                    display_columns.insert(0, 'group_number')
                if 'well' in results_df.columns:
                    display_columns.insert(1, 'well')
                if 'DEPTH' in results_df.columns:
                    display_columns.insert(-2, 'DEPTH')
                if 'Cov' in results_df.columns:
                    display_columns.append('Cov')
                if 'COLL_advance' in results_df.columns:
                    display_columns.append('COLL_advance')
                
                st.dataframe(results_df[display_columns].head(20))
                
                # Статистика по предсказаниям
                st.subheader('📊 Статистика по предсказаниям')
                prediction_stats = results_df.groupby('prediction').agg({
                    target_column: 'count',
                    **{col: 'mean' for col in selected_features if col in results_df.columns}
                }).round(3)
                st.dataframe(prediction_stats)
                
                # Сохраняем результаты в session_state
                st.session_state.merged_data = results_df
                st.success('💾 Результаты сохранены в session_state.merged_data')
                
            except Exception as e:
                st.error(f'❌ Ошибка: {str(e)}')
