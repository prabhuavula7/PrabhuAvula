FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch torchvision streamlit scikit-learn

CMD ["python", "train.py"]
