# Используем базовый образ PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Копирование requirements.txt в контейер
COPY requirements.txt /app/requirements.txt

# Установка зависимостей из requirements.txt
RUN pip3 install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Добавляем пути к библиотекам CUDA и cuDNN в LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Убедитесь, что доступна поддержка CUDA
RUN python -c "import torch; print(torch.cuda.is_available())"

# Копируйте файлы вашей модели и скрипты в контейнер
COPY voice_verification.py /app/voice_verification.py
COPY endpoint.py /app/endpoint.py
COPY data_preparation.py /app/data_preparation.py
COPY verificated_voice /app/verificated_voice

# Установите рабочую директорию в /app
WORKDIR /app

# Запустите команду, которая будет выполнять вашу модель на CUDA
CMD ["uvicorn", "endpoint:app", "--host", "0.0.0.0", "--port", "80"]
