FROM python:3.11

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install keras --upgrade

COPY ./ ./
WORKDIR scripts
RUN python autogen.py make

CMD ["python", "-u", "autogen.py", "serve"]
