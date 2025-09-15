import abc
import logging
import os
from typing import List, Dict, Any

import torch
import weaviate
import weaviate.classes as wvc
from elasticsearch import Elasticsearch, AuthenticationException
from weaviate.collections.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy
from weaviate.collections.classes.grpc import MetadataQuery

from indexing.embedding_model import EmbeddingModel


class SearchIndex(metaclass=abc.ABCMeta):
    def __init__(self, index_name: str, **index_kwargs) -> None:
        self.index_name = index_name
        self.index_kwargs = index_kwargs
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def index(self, documents: List[Dict[str, Any]]) -> None:
        pass

    @abc.abstractmethod
    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def delete(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class DenseIndex(SearchIndex, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, index_name: str, embedding_model: EmbeddingModel, **index_kwargs) -> None:
        super().__init__(index_name, **index_kwargs)
        self.embedding_model = embedding_model

    @abc.abstractmethod
    def search_vector(self, query_vectors: List[torch.Tensor]) -> List[Dict[str, Any]]:
        pass

class ElasticsearchIndex(SearchIndex):

    def __init__(self, **index_kwargs) -> None:
        super().__init__("rag-paper-index", **index_kwargs)

        api_key = os.getenv("ES_API_KEY")
        if api_key is None or api_key == "":
            self.logger.error("Elasticsearch API key (ES_API_KEY) not set!")
            return

        try:
            self.es_client = Elasticsearch(api_key=api_key,
                                           retry_on_timeout=True, max_retries=10, **self.index_kwargs)
        except AuthenticationException:
            self.logger.error("Authentication to Elasticsearch failed!")
            return

        self.mappings = {
            "mappings": {
                "properties": {
                    "meta": {
                        "properties": {
                            "title": {"type": "text"},
                            "doi": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "publication_year": {"type": "integer"},
                            "language": {"type": "keyword"},
                            "published": {"type": "text"},
                            "authors": {"type": "array"},
                            "topics": {"type": "array"},
                            "subfields": {"type": "array"},
                            "keywords": {"type": "array"},
                            "chunk_id": {"type": "integer"},
                        }
                    },
                    "text": {"type": "text"},
                    "vector": {"type": "dense_vector",
                               "dims": 1024},
                }
            }
        }

    def __len__(self) -> int:
        pass

    def index(self, documents: List[Dict[str, Any]]) -> None:
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
              index=self.index_name,
              mappings=self.mappings,
              settings={"number_of_shards": 5,
                        "number_of_replicas": 2,
                        "index": {
                            "similarity": {
                                "default": {
                                    "type": "BM25"
                                }
                            }
                        }})

        operations = []
        for doc in documents:
            operations.append({"index": {"_index": self.index_name, "_id": doc["id"]}})
            operations.append({"_source": doc})

        self.es_client.bulk(index=self.index_name, operations=operations, refresh=True)


    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        pass

    def delete(self):
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)

    def close(self):
        self.es_client.close()


class WeaviateIndex(DenseIndex):
    def __init__(self, embedding_model: EmbeddingModel, **index_kwargs) -> None:
        super().__init__("RAGPapers", embedding_model, **index_kwargs)
        self.client = weaviate.connect_to_custom(**self.index_kwargs)
        self.client.is_ready()
        self.collection = None

    def __len__(self) -> int:
        if self.collection is None:
            self.collection = self.client.collections.get(name=self.index_name)

        size = 0
        for _ in self.collection.iterator():
            size += 1

        return size

    def index(self, documents: List[Dict[str, Any]]) -> None:
        if not self.client.collections.exists(self.index_name):
             self.collection = self.client.collections.create(
                 self.index_name,
                 properties=[
                     Property(name="meta", data_type=DataType.OBJECT, nested_properties=[
                         Property(name="title", data_type=DataType.TEXT),
                         Property(name="doi", data_type=DataType.TEXT),
                         Property(name="type", data_type=DataType.TEXT),
                         Property(name="publication_year", data_type=DataType.NUMBER),
                         Property(name="language", data_type=DataType.TEXT),
                         Property(name="published", data_type=DataType.TEXT),
                         Property(name="authors", data_type=DataType.TEXT_ARRAY),
                         Property(name="topics", data_type=DataType.TEXT_ARRAY),
                         Property(name="subfields", data_type=DataType.TEXT_ARRAY),
                         Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                         Property(name="chunk_id", data_type=DataType.NUMBER),
                     ]),
                     Property(name="text", data_type=DataType.TEXT),
                 ],
                vector_config=wvc.config.Configure.Vectors.self_provided(
                    name="vector",
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE,
                        filter_strategy=VectorFilterStrategy.SWEEPING,
                    )
                ))

        if self.collection is None:
            self.collection = self.client.collections.get(name=self.index_name)


        objects = []
        for document in documents:
            objects.append(
                wvc.data.DataObject(
                    properties={"meta": document["meta"], "text": document["text"]},
                    vector=document["vector"]
                )
            )

        self.collection.data.insert_many(objects)


    def search_vector(self, query_vectors: List[torch.Tensor]) -> List[Dict[str, Any]]:
        if self.collection is None:
            self.collection = self.client.collections.get(name=self.index_name)

        response = self.collection.query.near_vector(
            near_vector=query_vectors[0].numpy(),
            limit=20,
            target_vector="vector",
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for res in response.objects:
            results.append({**res.properties, "distance": res.metadata.distance})
        return results

    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        query_vectors = self.embedding_model.embed_queries(queries)
        return self.search_vector(query_vectors)


    def close(self):
        self.client.close()

    def delete(self):
        if self.client.collections.exists(self.index_name):
            self.client.collections.delete(self.index_name)
