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
                 llm: LLM,
                 gen_kwargs=None):

        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 512}

        self.query_formulator = query_formulator
        self.search_index = search_index
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.gen_kwargs = gen_kwargs

    def chat(self, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        queries = self.query_formulator.get_queries(conversation)
        search_results = self.search_index.search(queries)[:5]
        prompt = self.prompt_builder.build_prompt(conversation, search_results)

        for token in self.llm.generate_stream(prompt, **self.gen_kwargs):
            yield token
        yield "\n\n**References**:\n\n"

        dois = set()
        i = 1
        for result in search_results:
            if result["meta"]["doi"] in dois:
                continue

            author_string = ", ".join(result["meta"]["authors"])
            yield f"[{i}] {author_string}. [{result['meta']['title']}](https://doi.org/{result['meta']['doi']}). *{result['meta']['published']}*. {result['meta']['publication_year']}.\n\n"

            dois.add(result["meta"]["doi"])
            i += 1