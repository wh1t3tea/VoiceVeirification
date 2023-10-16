FROM pytorch/pytorch:latest

# Установите все зависимости вашего проекта
RUN pip install numpy torch torchvision speechbrain torchaudio librosa pydub soundfile fastapi python-multipart uvicorn

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