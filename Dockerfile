FROM python:3.9

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install keras==3.0.5

COPY ./ ./
WORKDIR scripts
RUN python autogen.py make

CMD ["python", "-u", "autogen.py", "serve"]
