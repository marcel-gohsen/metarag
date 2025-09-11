import abc
from typing import List, Generator

import semchunk


class Chunker(metaclass=abc.ABCMeta):
    def __init__(self, chunk_length, **kwargs):
        self.chunk_length = chunk_length
        self.kwargs = kwargs

    @abc.abstractmethod
    def chunk(self, texts: List[str]) -> Generator[List[str], None, None]:
        pass


class SemanticChunker(Chunker):
    def __init__(self, chunk_length, **kwargs):
        super().__init__(chunk_length, **kwargs)
        self.chunker = semchunk.chunkerify('isaacus/kanon-tokenizer', chunk_length)

    def chunk(self, texts: List[str]) -> Generator[List[str], None, None]:
        for chunks in self.chunker(texts, **self.kwargs):
            yield chunks

