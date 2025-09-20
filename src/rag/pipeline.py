import abc
import json
import logging
from typing import List, Dict, Generator

from indexing.search_index import SearchIndex, DenseIndex, ElasticsearchIndex
from rag.llm import LLM, HFModel, OpenAIModel
from rag.prompt import PromptBuilder, AnswerQuestionsBuilder, AnswerQuestionWithPassagesBuilder, \
    SummarizePaperPromptBuilder, FoundMultiplePromptBuilder, FoundNonePromptBuilder
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
            if isinstance(llm, HFModel):
                gen_kwargs = {"max_new_tokens": 512}
            elif isinstance(llm, OpenAIModel):
                gen_kwargs = {"max_completion_tokens": 512}

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


class DensePipeline(Pipeline):
    def __init__(self,
                 query_formulator: QueryFormulator,
                 search_index: DenseIndex,
                 prompt_builder: PromptBuilder,
                 llm: LLM,
                 gen_kwargs=None):

        if gen_kwargs is None:
            if isinstance(llm, HFModel):
                gen_kwargs = {"max_new_tokens": 512}
            elif isinstance(llm, OpenAIModel):
                gen_kwargs = {"max_completion_tokens": 512}

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

class HybridPipeline(Pipeline):
    def __init__(self, sparse_index: ElasticsearchIndex, dense_index: DenseIndex, llm: LLM, gen_kwargs=None):
        if gen_kwargs is None:
            gen_kwargs = {}
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.llm = llm
        self.gen_kwargs = gen_kwargs
        self.answer_question_builder = AnswerQuestionsBuilder()
        self.system_question_with_passages_builder = AnswerQuestionWithPassagesBuilder()
        self.summarize_paper_builder = SummarizePaperPromptBuilder()
        self.found_multiple_builder = FoundMultiplePromptBuilder()
        self.found_none_builder = FoundNonePromptBuilder()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "find_answers",
                    "description": "The user is asking questions about RAG. This tool provides excerpts from papers that try to answer the user's question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question the user is asking.",
                            }
                        },
                        "additionalProperties": False,
                        "required": ["question"],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_paper",
                    "description": "The user is asking details about a specific paper. This tool finds the content of that specific paper by meta information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The title of the paper the user is looking for.",
                            },
                            "authors": {
                                "type": "array",
                                "description": "The authors of the paper the user is looking for.",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "doi": {
                                "type": "string",
                                "description": "The DOI of the paper the user is looking for."
                            }
                        },
                        "additionalProperties": False,
                        "required": [],
                    }
                }
            }
        ]

        self.alpha = 1.0
        self.beta = 0.5

    def search_papers(self, data, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        search_results = self.dense_index.search([data["question"]])
        try:
            max_citations = max([r["meta"]["cited_by"] for r in search_results])
        except ValueError:
            max_citations = 0

        for result in search_results:
            result["score"] = self.alpha * result["score"] + self.beta * (
                        (result["meta"]["cited_by"] + 1) / (max_citations + 1))

        self.logger.info(f"Found {len(search_results)} chunks.")
        search_results = list(sorted(search_results, key=lambda r: r["score"], reverse=True))[:5]

        prompt = self.system_question_with_passages_builder.build_prompt(conversation, search_results)

        for token in self.llm.generate_stream(prompt, **self.gen_kwargs):
            yield token

        if len(search_results) == 0:
            return

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

    def find_paper_by_meta(self, data, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        query = {"query": {"bool": {}}}

        if "doi" in data:
            if "must" not in query["query"]["bool"]:
                query["query"]["bool"]["must"] = []
            query["query"]["bool"]["must"].append({"term": {"meta.doi": f'https://doi.org/{data["doi"]}'}})

        if "title" in data:
            if "must" not in query["query"]["bool"]:
                query["query"]["bool"]["must"] = []

            query["query"]["bool"]["must"].append(
                {"match": {"meta.title": {"query": data["title"], "operator": "and"}}}
            )
        if "authors" in data:
            if "must" not in query["query"]["bool"]:
                query["query"]["bool"]["must"] = []

            for author in data["authors"]:
                query["query"]["bool"]["must"].append(
                    {"match": {"meta.authors": {"query": author, "operator": "and"}}}
                )

        results = self.sparse_index.faceted_search(query)
        dois = set()
        for result in results:
            dois.add(result["meta"]["doi"])

        self.logger.info(f"Found {len(dois)} different paper.")
        if len(dois) == 0:
            prompt = self.found_none_builder.build_prompt(conversation, results)
        elif len(dois) > 1:
            prompt = self.found_multiple_builder.build_prompt(conversation, results)
        else:
            prompt = self.summarize_paper_builder.build_prompt(conversation, results)

        yield from self.llm.generate_stream(prompt, **self.gen_kwargs)



    def chat(self, conversation: List[Dict[str, str]]) -> Generator[str, None, None]:
        function_names = [f["function"]["name"] for f in self.tools]
        prompt = self.answer_question_builder.build_prompt(conversation, None)
        is_tool_call = False
        token = None
        for token in self.llm.generate_stream(prompt, self.tools, **self.gen_kwargs):
            if token.split("(")[0] in function_names:
                is_tool_call = True
                break

            yield token

        if not is_tool_call:
            return

        self.logger.info(f"Tool was called \"{token}\"")
        function_name = token.split("(")[0]
        arguments = "(".join(token.split("(")[1:])[:-1]
        data = json.loads(arguments)

        if function_name == "find_answers":
            yield from self.search_papers(data, conversation)
        elif function_name == "find_paper":
            yield from self.find_paper_by_meta(data, conversation)


