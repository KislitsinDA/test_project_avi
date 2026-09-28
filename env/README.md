# Виртуальное окружение (Windows)

Все команды — в **PowerShell**, из корня проекта `avito`.

## 1. Перейти в папку

```powershell
cd c:\Users\kisli\PycharmProjects\tbank\avito
```

## 2. Создать venv (папка `.venv` появится здесь)

```powershell
python -m venv .venv
```

## 3. Включить venv

```powershell
.\.venv\Scripts\Activate.ps1
```

Если ругается на политику выполнения:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Потом снова `Activate.ps1`.

В начале строки терминала должно быть `(.venv)`.

## 4. Обновить pip и поставить зависимости

**С CUDA (рекомендуется, если есть RTX 4050):**

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

**Только CPU:**

```powershell
python -m pip install -U pip
pip install -r requirements-cpu.txt
```

## 5. Проверить GPU

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Нужно: `cuda True` и имя видеокарты.

## 6. Модель e5 и решение

```powershell
python download_model.py
python run_solution.py --compare --semantic
python run_solution.py --submit --semantic
```

Файл для платформы: `answer.csv` (не открывать в Excel).

## Выключить venv

```powershell
deactivate
```
