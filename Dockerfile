FROM python:3.7

ARG TF_VERSION=2.2.0
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install tensorflow==${TF_VERSION}

COPY ./ ./
WORKDIR scripts
RUN python autogen.py make

CMD ["python", "-u", "autogen.py", "serve"]
