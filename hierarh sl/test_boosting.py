"""
Тестовый скрипт для проверки работы агломеративного бустинга
"""

import pandas as pd
import numpy as np
from agglomerative_boosting import AgglomerativeBoosting
import matplotlib.pyplot as plt

def create_test_data():
    """Создание тестовых данных для проверки работы бустинга"""
    
    # Создаем синтетические данные коллекторов
    np.random.seed(42)
    n_samples = 200
    
    # Создаем 3 типа коллекторов с разными характеристиками
    data = []
    
    # Тип 1: Высокопористые коллекторы
    for i in range(n_samples // 3):
        data.append({
            'group_number': i + 1,
            'well': f'Well_{i % 5 + 1}',
            'GR': np.random.normal(60, 10),  # Низкая радиоактивность
            'RHOB': np.random.normal(2.2, 0.1),  # Низкая плотность
            'NPHI': np.random.normal(0.25, 0.05),  # Высокая пористость нейтронов
            'por_rf': np.random.normal(0.22, 0.04),  # Высокая пористость
            'kvo_rf': np.random.normal(0.18, 0.03),  # Высокая нефтенасыщенность
            'Q': np.random.normal(50, 15),  # Высокий дебит
            'coll_type': 'высокопористый'
        })
    
    # Тип 2: Среднепористые коллекторы
    for i in range(n_samples // 3, 2 * n_samples // 3):
        data.append({
            'group_number': i + 1,
            'well': f'Well_{i % 5 + 1}',
            'GR': np.random.normal(80, 15),  # Средняя радиоактивность
            'RHOB': np.random.normal(2.4, 0.15),  # Средняя плотность
            'NPHI': np.random.normal(0.15, 0.04),  # Средняя пористость нейтронов
            'por_rf': np.random.normal(0.12, 0.03),  # Средняя пористость
            'kvo_rf': np.random.normal(0.08, 0.02),  # Средняя нефтенасыщенность
            'Q': np.random.normal(20, 8),  # Средний дебит
            'coll_type': 'среднепористый'
        })
    
    # Тип 3: Низкопористые коллекторы
    for i in range(2 * n_samples // 3, n_samples):
        data.append({
            'group_number': i + 1,
            'well': f'Well_{i % 5 + 1}',
            'GR': np.random.normal(120, 20),  # Высокая радиоактивность
            'RHOB': np.random.normal(2.6, 0.2),  # Высокая плотность
            'NPHI': np.random.normal(0.05, 0.02),  # Низкая пористость нейтронов
            'por_rf': np.random.normal(0.05, 0.02),  # Низкая пористость
            'kvo_rf': np.random.normal(0.02, 0.01),  # Низкая нефтенасыщенность
            'Q': np.random.normal(5, 3),  # Низкий дебит
            'coll_type': 'низкопористый'
        })
    
    return pd.DataFrame(data)

def test_regression():
    """Тест регрессионной модели"""
    print("🧪 Тестирование регрессионной модели...")
    
    # Создаем тестовые данные
    df = create_test_data()
    
    # Настройки модели
    model = AgglomerativeBoosting(
        n_clusters=3,
        linkage_method='ward',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        task_type='regression'
    )
    
    # Признаки для обучения
    features = ['GR', 'RHOB', 'NPHI', 'por_rf', 'kvo_rf']
    target = 'Q'
    
    # Обучение
    model.fit(df, features, target)
    print("✅ Модель обучена")
    
    # Предсказание
    predictions = model.predict(df, features)
    print("✅ Предсказания получены")
    
    # Кросс-валидация
    cv_results = model.cross_validate(df, features, target, cv_folds=3)
    print(f"📊 R² Score: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # Визуализация
    fig = model.plot_dendrogram(df, features, figsize=(10, 6))
    plt.title("Дендрограмма для регрессионной модели")
    plt.show()
    
    return model, df

def test_classification():
    """Тест классификационной модели"""
    print("\n🧪 Тестирование классификационной модели...")
    
    # Создаем тестовые данные
    df = create_test_data()
    
    # Настройки модели
    model = AgglomerativeBoosting(
        n_clusters=3,
        linkage_method='ward',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        task_type='classification'
    )
    
    # Признаки для обучения
    features = ['GR', 'RHOB', 'NPHI', 'por_rf', 'kvo_rf']
    target = 'coll_type'
    
    # Обучение
    model.fit(df, features, target)
    print("✅ Модель обучена")
    
    # Предсказание
    predictions = model.predict(df, features)
    print("✅ Предсказания получены")
    
    # Кросс-валидация
    cv_results = model.cross_validate(df, features, target, cv_folds=3)
    print(f"📊 Accuracy: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # Анализ кластеров
    fig = model.plot_cluster_analysis(df, features, target, figsize=(12, 8))
    plt.title("Анализ кластеров для классификационной модели")
    plt.show()
    
    return model, df

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование агломеративного бустинга")
    print("=" * 50)
    
    try:
        # Тест регрессии
        reg_model, reg_df = test_regression()
        
        # Тест классификации
        cls_model, cls_df = test_classification()
        
        print("\n✅ Все тесты прошли успешно!")
        print("\n📋 Сводка результатов:")
        print("- Регрессионная модель: обучена и протестирована")
        print("- Классификационная модель: обучена и протестирована")
        print("- Дендрограммы: построены")
        print("- Анализ кластеров: выполнен")
        print("- Кросс-валидация: проведена")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


