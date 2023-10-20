FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY requirements.txt /app/requirements.txt

RUN pip3 install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r /app/requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN python -c "import torch; print(torch.cuda.is_available())"

COPY voice_verification.py /app/voice_verification.py
COPY endpoint.py /app/endpoint.py
COPY data_preparation.py /app/data_preparation.py
COPY verificated_voice /app/verificated_voice

WORKDIR /app

CMD ["uvicorn", "endpoint:app", "--host", "localhost", "--port", "80"]
