FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ~/.insightface/models
RUN wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true -O ~/.insightface/models/inswapper_128.onnx

CMD ["python", "main.py"]