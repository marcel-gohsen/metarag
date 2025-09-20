import asyncio
import importlib.metadata
import json
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from config import CONFIG, INDICES, EMBEDDING_MODELS
from rag.llm import OpenAIModel, OpenAIModelVersion
from rag.pipeline import HybridPipeline

app = FastAPI(
    title=CONFIG["app"]["title"],
    version=importlib.metadata.version("rag-on-rag-demo")
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

pipeline = HybridPipeline(
    sparse_index=INDICES["elastic"](**CONFIG["index"]["elastic"]),
    dense_index=INDICES["weaviate"](EMBEDDING_MODELS["qwen3-embedding-4B"](), **CONFIG["index"]["weaviate"]),
    llm=OpenAIModel(OpenAIModelVersion.GPT_4_1)
)

class Utterance(BaseModel):
    role: str
    content: str

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={"app": {"version": importlib.metadata.version("rag-on-rag-demo"), **CONFIG["app"]}})

@app.post("/chat")
def chat(conversation: List[Utterance]):
    conversation = [c.model_dump() for c in conversation]

    async def streamer():
        for token in pipeline.chat(conversation):
            yield json.dumps({"text": token}) + "\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(streamer(), media_type='text/event-stream')


def main():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = log_format
    log_config["formatters"]["default"]["fmt"] = log_format

    logging.basicConfig(level=logging.INFO, format=log_format)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()