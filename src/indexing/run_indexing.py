import itertools
import json
import logging
import os
import re
from typing import Any, Dict, Generator

import click
import pandas as pd
from tqdm import tqdm

from config import EMBEDDING_MODELS, INDICES, CONFIG
from indexing.chunking import SemanticChunker
from indexing.search_index import DenseIndex


def load_raw(path: str) -> Generator[Dict[str, Any], None, None]:
    df_meta = pd.read_csv(CONFIG["data"]["meta_file"]).set_index("id")
    whitespace_pattern = re.compile(r"\s+")

    for file in os.listdir(path):
        if not file.endswith(".md"):
            continue

        with open(os.path.join(path, file), "r") as f:
            fulltext = f.read()

        _id = f"https://openalex.org/{file.split('.')[0]}"

        meta = df_meta.loc[[_id]]
        meta = meta.reset_index().to_dict(orient="index")[0]
        data = {
            "id": _id,
            "meta": {
                "title": whitespace_pattern.sub( " ", meta["title"]),
                "doi": meta["doi"],
                "type": meta["type"],
                "publication_year": meta["publication_year"],
                "language": meta["language"],
                "published": meta["primary_location.source.display_name"],
                "authors": [a for a in str(meta["authorships.author.display_name"]).split("|")],
                "topics": [t for t in str(meta["topics.display_name"]).split("|")],
                "subfields": [s for s in str(meta["topics.subfield.display_name"]).split("|")],
                "keywords": [k for k in str(meta["keywords.display_name"]).split("|")],
            },
            "fulltext": fulltext,
        }

        yield data

def count_raw(path: str) -> int:
    num = 0
    for file in os.listdir(path):
        if not file.endswith(".md"):
            continue

        num += 1

    return num

def load_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data


@click.command()
@click.option("--index-type", type=click.Choice(list(INDICES.keys())), default="weaviate")
@click.option("--emb-model", type=click.Choice(list(EMBEDDING_MODELS.keys())), default="qwen3-embedding-0.6B")
@click.option("--batch-size", type=int, default=128)
def main(emb_model: str, index_type: str, batch_size: int):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    search_index = None

    num_documents = 0
    num_chunks = 0

    try:
        logging.info("Loading emb model...")
        embedding_model = EMBEDDING_MODELS[emb_model]()

        logging.info("Connect to index...")
        search_index_class = INDICES[index_type]
        if issubclass(search_index_class, DenseIndex):
            search_index = search_index_class(emb_model, **CONFIG["index"][index_type])
        else:
            search_index = search_index_class(**CONFIG["index"][index_type])

        search_index.delete()

        chunker = SemanticChunker(chunk_length=256, overlap=0.3)

        logging.info("Start indexing...")
        for document in tqdm(load_raw(CONFIG["data"]["raw_dir"]), total=count_raw(CONFIG["data"]["raw_dir"])):
            chunk_id = 0
            num_documents += 1
            for document_chunks in chunker.chunk([document["fulltext"]]):
                for chunk_batch in itertools.batched(document_chunks, batch_size):
                    batch_docs = []
                    vectors = embedding_model.embed_documents(list(chunk_batch))

                    for chunk, vector in zip(chunk_batch, vectors):
                        num_chunks += 1
                        batch_docs.append({
                            "meta": {**document["meta"], "chunk_id": chunk_id},
                            "text": chunk,
                            "vector": vector.numpy()
                        })

                        chunk_id += 1

                    search_index.index(batch_docs)

        logging.info("Indexing complete.")

    finally:
        logging.info("Shutting down...")
        logging.info(f"Indexed {num_documents:,} documents.")
        logging.info(f"Indexed {num_chunks:,} chunks.")
        if search_index is not None:
            search_index.close()






if __name__ == '__main__':
    main()
