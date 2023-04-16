FROM python:3.10.0

#RUN pip install --upgrade pip


WORKDIR /app
COPY requirements.txt requirements.txt


RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade -r ./requirements.txt

COPY /models /app/models
COPY SupportScript.py /app
COPY predictionAPI.py /app

CMD ["uvicorn", "predictionAPI:app", "--host", "0.0.0.0", "--port", "80"]
