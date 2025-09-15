import abc
from typing import List, Dict, Generator

from indexing.search_index import SearchIndex
from rag.llm import LLM
from rag.prompt import PromptBuilder
from rag.query import QueryFormulator


class Pipeline(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def chat(self, conversation: List[Dict[str, str]]) -> Dict[str, str]:
        pass

class SparsePipeline(Pipeline):
    def __init__(self,
                 query_formulator: QueryFormulator,
                 search_index: SearchIndex,
                 prompt_builder: PromptBuilder,
                 llm: LLM):
        self.query_formulator = query_formulator
        self.search_index = search_index
        self.prompt_builder = prompt_builder
        self.llm = llm

    def chat(self, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        for token in self.llm.generate_stream(conversation, max_new_tokens=256):
            yield token