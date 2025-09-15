import yaml

from indexing.embedding_model import SentenceTransformerEmbeddingModel
from indexing.search_index import WeaviateIndex, ElasticsearchIndex

with open("config.yml", 'r') as ymlfile:
    CONFIG = yaml.safe_load(ymlfile)
    CONFIG = CONFIG["config"]

EMBEDDING_MODELS = {
    "stella_en_400M_v5": lambda: SentenceTransformerEmbeddingModel("dunzhang/stella_en_400M_v5"),
    "stella_en_1.5B_v5": lambda: SentenceTransformerEmbeddingModel("NovaSearch/stella_en_1.5B_v5"),
    "qwen3-embedding-4B": lambda: SentenceTransformerEmbeddingModel("Qwen/Qwen3-Embedding-4B", query_prompt_name="query"),
    "qwen3-embedding-0.6B": lambda: SentenceTransformerEmbeddingModel("Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query")
}

INDICES = {
    "weaviate": WeaviateIndex,
    "elastic": ElasticsearchIndex,
}