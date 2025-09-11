import abc
from typing import List, Dict, Any

import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

from indexing.embedding_model import EmbeddingModel


class SearchIndex(metaclass=abc.ABCMeta):
    def __init__(self, index_name: str, **index_kwargs) -> None:
        self.index_name = index_name
        self.index_kwargs = index_kwargs

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def index(self, documents: List[Dict[str, Any]]) -> None:
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


class WeaviateIndex(DenseIndex):
    def __init__(self, index_name: str, embedding_model: EmbeddingModel, **index_kwargs) -> None:
        super().__init__(index_name, embedding_model, **index_kwargs)
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

    def close(self):
        self.client.close()

    def delete(self):
        if self.client.collections.exists(self.index_name):
            self.client.collections.delete(self.index_name)

if __name__ == '__main__':
    search_index = WeaviateIndex("RAGPapers", None,
                                 http_host="weaviate.srv.webis.de", http_port=443, http_secure=True,
                                 grpc_host="weaviate-grpc.srv.webis.de", grpc_port=443, grpc_secure=True)
    print(len(search_index))
    search_index.close()
