FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY tira/predict_tira.py /app/predict_tira.py
COPY tira/run.sh /run.sh

RUN chmod +x /run.sh

ENTRYPOINT ["/run.sh"]