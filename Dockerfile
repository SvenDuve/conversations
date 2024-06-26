FROM python:3.11-slim

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

COPY ./context ./context

COPY ./vs ./vs

RUN poetry install --no-interaction --no-ansi

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger

EXPOSE 3000

CMD exec uvicorn app.server:app --host 0.0.0.0 --port 3000
