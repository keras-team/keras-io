
FROM python:3.6

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./
WORKDIR scripts
RUN python autogen.py make
