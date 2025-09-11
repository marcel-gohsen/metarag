import yaml

from indexing.embedding_model import SentenceTransformerEmbeddingModel
from indexing.search_index import WeaviateIndex

with open("config.yml", 'r') as ymlfile:
    CONFIG = yaml.safe_load(ymlfile)
    CONFIG = CONFIG["config"]

EMBEDDING_MODELS = {
    "stella_en_400M_v5": lambda: SentenceTransformerEmbeddingModel("NovaSearch/stella_en_400M_v5")
}

INDICES = {
    "weaviate": lambda index_name, emb_model: WeaviateIndex(index_name, emb_model, **CONFIG["index"]["weaviate"]),
}