"""
Скрипт для анализа использования дискового пространства
"""

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

def get_folder_size(path):
    """Получить размер папки в байтах"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    continue
    except (OSError, IOError):
        pass
    return total_size

def format_size(size_bytes):
    """Форматировать размер в читаемый вид"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def analyze_disk_c():
    """Анализ диска C:"""
    print("🔍 Анализ диска C:")
    print("=" * 50)
    
    # Основные папки для анализа
    folders_to_check = [
        "C:\\Users",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
        "C:\\Windows",
        "C:\\ProgramData",
        "C:\\Temp",
        "C:\\tmp"
    ]
    
    folder_sizes = []
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            print(f"📁 Анализирую {folder}...")
            try:
                size = get_folder_size(folder)
                folder_sizes.append({
                    'Папка': folder,
                    'Размер (байты)': size,
                    'Размер (читаемый)': format_size(size)
                })
                print(f"   Размер: {format_size(size)}")
            except Exception as e:
                print(f"   Ошибка: {e}")
                folder_sizes.append({
                    'Папка': folder,
                    'Размер (байты)': 0,
                    'Размер (читаемый)': "Ошибка доступа"
                })
        else:
            print(f"📁 {folder} - не существует")
    
    # Сортируем по размеру
    folder_sizes.sort(key=lambda x: x['Размер (байты)'], reverse=True)
    
    print("\n📊 ТОП папок по размеру:")
    print("-" * 80)
    for i, folder in enumerate(folder_sizes[:10], 1):
        print(f"{i:2d}. {folder['Папка']:<30} {folder['Размер (читаемый)']:>15}")
    
    return folder_sizes

def analyze_user_folders():
    """Анализ папок пользователя"""
    print("\n👤 Анализ папок пользователя:")
    print("=" * 50)
    
    user_profile = os.path.expanduser("~")
    user_folders = [
        "AppData",
        "Documents", 
        "Downloads",
        "Desktop",
        "Pictures",
        "Videos",
        "Music"
    ]
    
    user_sizes = []
    
    for folder in user_folders:
        folder_path = os.path.join(user_profile, folder)
        if os.path.exists(folder_path):
            print(f"📁 Анализирую {folder}...")
            try:
                size = get_folder_size(folder_path)
                user_sizes.append({
                    'Папка': folder,
                    'Размер (байты)': size,
                    'Размер (читаемый)': format_size(size)
                })
                print(f"   Размер: {format_size(size)}")
            except Exception as e:
                print(f"   Ошибка: {e}")
    
    # Сортируем по размеру
    user_sizes.sort(key=lambda x: x['Размер (байты)'], reverse=True)
    
    print("\n📊 ТОП папок пользователя:")
    print("-" * 50)
    for i, folder in enumerate(user_sizes, 1):
        print(f"{i:2d}. {folder['Папка']:<20} {folder['Размер (читаемый)']:>15}")
    
    return user_sizes

def analyze_appdata():
    """Детальный анализ AppData"""
    print("\n🔧 Детальный анализ AppData:")
    print("=" * 50)
    
    appdata_path = os.path.join(os.path.expanduser("~"), "AppData")
    appdata_folders = ["Local", "Roaming", "LocalLow"]
    
    appdata_sizes = []
    
    for folder in appdata_folders:
        folder_path = os.path.join(appdata_path, folder)
        if os.path.exists(folder_path):
            print(f"📁 Анализирую AppData\\{folder}...")
            try:
                size = get_folder_size(folder_path)
                appdata_sizes.append({
                    'Папка': f"AppData\\{folder}",
                    'Размер (байты)': size,
                    'Размер (читаемый)': format_size(size)
                })
                print(f"   Размер: {format_size(size)}")
            except Exception as e:
                print(f"   Ошибка: {e}")
    
    # Сортируем по размеру
    appdata_sizes.sort(key=lambda x: x['Размер (байты)'], reverse=True)
    
    print("\n📊 AppData по размеру:")
    print("-" * 50)
    for i, folder in enumerate(appdata_sizes, 1):
        print(f"{i:2d}. {folder['Папка']:<20} {folder['Размер (читаемый)']:>15}")
    
    return appdata_sizes

def find_large_files(root_path, min_size_mb=100):
    """Поиск больших файлов"""
    print(f"\n🔍 Поиск файлов больше {min_size_mb} MB:")
    print("=" * 50)
    
    large_files = []
    min_size_bytes = min_size_mb * 1024 * 1024
    
    try:
        for root, dirs, files in os.walk(root_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        if file_size >= min_size_bytes:
                            large_files.append({
                                'Файл': file_path,
                                'Размер (байты)': file_size,
                                'Размер (читаемый)': format_size(file_size)
                            })
                except (OSError, IOError):
                    continue
    except (OSError, IOError):
        pass
    
    # Сортируем по размеру
    large_files.sort(key=lambda x: x['Размер (байты)'], reverse=True)
    
    print(f"Найдено {len(large_files)} файлов больше {min_size_mb} MB:")
    print("-" * 80)
    for i, file_info in enumerate(large_files[:20], 1):  # Показываем топ-20
        print(f"{i:2d}. {file_info['Размер (читаемый)']:>15} - {file_info['Файл']}")
    
    return large_files

def get_disk_usage():
    """Получить информацию об использовании диска"""
    print("\n💾 Информация о дисках:")
    print("=" * 50)
    
    try:
        # Используем PowerShell для получения информации о дисках
        result = subprocess.run([
            'powershell', '-Command', 
            'Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round($_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round($_.FreeSpace/1GB,2)}}, @{Name="PercentFree";Expression={[math]::Round(($_.FreeSpace/$_.Size)*100,2)}} | Format-Table -AutoSize'
        ], capture_output=True, text=True, encoding='utf-8')
        
        print(result.stdout)
    except Exception as e:
        print(f"Ошибка получения информации о дисках: {e}")

def main():
    """Основная функция"""
    print("🚀 Анализ использования дискового пространства")
    print("=" * 60)
    
    # Информация о дисках
    get_disk_usage()
    
    # Анализ основных папок
    folder_sizes = analyze_disk_c()
    
    # Анализ папок пользователя
    user_sizes = analyze_user_folders()
    
    # Детальный анализ AppData
    appdata_sizes = analyze_appdata()
    
    # Поиск больших файлов в пользовательской папке
    user_profile = os.path.expanduser("~")
    large_files = find_large_files(user_profile, min_size_mb=50)
    
    # Рекомендации по очистке
    print("\n💡 РЕКОМЕНДАЦИИ ПО ОЧИСТКЕ:")
    print("=" * 50)
    
    total_user_size = sum(folder['Размер (байты)'] for folder in user_sizes)
    total_appdata_size = sum(folder['Размер (байты)'] for folder in appdata_sizes)
    
    print(f"1. 📁 Папки пользователя занимают: {format_size(total_user_size)}")
    print(f"2. 🔧 AppData занимает: {format_size(total_appdata_size)}")
    print(f"3. 🔍 Найдено {len(large_files)} больших файлов (>50MB)")
    
    print("\n🎯 Приоритетные действия:")
    print("• Очистите временные файлы (Temp, %TEMP%)")
    print("• Проверьте папку Downloads на старые файлы")
    print("• Очистите кэш браузеров")
    print("• Удалите неиспользуемые программы")
    print("• Очистите корзину")
    
    # Сохранение отчета
    try:
        df = pd.DataFrame(folder_sizes)
        df.to_csv('disk_analysis_report.csv', index=False, encoding='utf-8')
        print(f"\n📄 Отчет сохранен в disk_analysis_report.csv")
    except Exception as e:
        print(f"Ошибка сохранения отчета: {e}")

if __name__ == "__main__":
    main()


