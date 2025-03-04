FROM node:23.8-bullseye AS node-builder

COPY ./ ./

FROM python:3.9 AS final

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install keras --upgrade

COPY ./ ./
COPY --from=node-builder /bundle ./bundle

WORKDIR /scripts
RUN python autogen.py make

CMD ["python", "-u", "autogen.py", "serve"]
