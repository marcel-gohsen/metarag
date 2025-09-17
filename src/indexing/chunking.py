import abc
import re
from typing import List, Generator

import semchunk
from transformers import AutoTokenizer


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
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B')
        self.chunker = semchunk.chunkerify(self.tokenizer, chunk_length)
        self.alpha_num_pattern = r'[^a-zA-Z0-9\s\\{}.,!?]'

    def chunk(self, texts: List[str]) -> Generator[List[str], None, None]:
        for chunks in self.chunker(texts, **self.kwargs):
            filtered_chunks = []
            for c in chunks:
                if len(self.tokenizer(c)["input_ids"]) <= 50:
                    continue

                if len(re.findall(self.alpha_num_pattern, c)) / len(c) >= 0.1:
                    continue

                filtered_chunks.append(c)

            yield filtered_chunks

