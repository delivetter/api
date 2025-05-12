# build stage
FROM python:3.12-slim

# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# copy files
COPY . /project

# install dependencies and project
WORKDIR /project
RUN pdm install --prod --no-lock --no-editable

EXPOSE 8080

# Creates a non-root user with an explicit UID and adds permission to access the /project folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /project && chown -R appuser /opt
USER appuser

ENTRYPOINT ["pdm", "run", "server"]
