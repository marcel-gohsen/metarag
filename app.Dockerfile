FROM registry.webis.de/code-research/conversational-search/rag-on-rag-demo/base:latest
LABEL authors="Marcel Gohsen"

EXPOSE 8000

COPY ./static/ /app/static
COPY ./templates/ /app/templates

ENTRYPOINT python3 src/application/serve.py