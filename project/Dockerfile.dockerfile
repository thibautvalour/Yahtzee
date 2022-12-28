FROM python:3.8.6-slim-buster

LABEL maintainer ""

WORKDIR /usr/src/app

COPY requirements.txt /
RUN pip install --upgrade pip
RUN install -r /requirements.txt

COPY /app /usr/src/app/

RUN useradd -m appUser
USER appUser

CMD gunicorn --bind 127.0.0.1:8050 app:server
