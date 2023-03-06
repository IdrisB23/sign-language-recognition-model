# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get -y update && apt-get -y install git ffmpeg

WORKDIR /signlang_recog_model


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY src /signlang_recog_model/src
COPY data/dataset /signlang_recog_model/data/dataset

CMD ["python", "src/flask_app.py"]