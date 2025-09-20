import abc
from typing import List, Dict, Any, Optional


class PromptBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        pass


class AnswerQuestionWithPassagesBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation. 
To help you answer questions of the users, you are provided a list of passages from scientific papers. 
Please add references in the text in a scientific format (e.g., [1], [2]).  
You may ignore these passages if they don't contain answers to the user's questions. 
If no passages are provided, please tell the user that you rely on your own knowledge to try to answer the question.
Do not provide a list of references in the end. 

# Passages 
"""
        self.reference_template = \
"""
[{id}]
{content}
"""

    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        system_prompt = self.system_prompt_template

        dois = set()
        i = 1
        for result in search_results:
            system_prompt += (self.reference_template.format(id=i, content=result["text"]))
            system_prompt += "\n\n"

            if not result["meta"]["doi"] in dois:
                dois.add(result["meta"]["doi"])
                i += 1

        return [{"role": "system", "content": system_prompt}, *conversation]


class AnswerQuestionsBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation. 
To help you answer questions of the users, you can call provided functions to obtain papers about retrieval-augmented generation.
Always search for papers if a question about retrival-augmented generation is asked.
"""

    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        system_prompt = self.system_prompt_template

        return [{"role": "system", "content": system_prompt}, *conversation]


class SummarizePaperPromptBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation. """

        self.user_turn_template = \
"""I wanna know \"{user_input}\".

Here is the content of the paper I ask about:
{content}"""

    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        system_prompt = self.system_prompt_template

        content = ""
        for result in sorted(search_results, key=lambda result: result["meta"]["chunk_id"]):
            content += result["text"]

        conversation[-1]["content"] = self.user_turn_template.format(user_input=conversation[-1]["content"], content=content)
        return [{"role": "system", "content": system_prompt}, *conversation]


class FoundMultiplePromptBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation.
You found {num_papers} papers that fit the description of the user. 
Don't answer the question and ask the user for clarification which paper he or she is referring to.
Below you find the found papers.

# Papers
"""

    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        system_prompt = self.system_prompt_template

        dois = set()
        for result in search_results:
            if result["meta"]["doi"] not in dois:
                author_string = ", ".join(result["meta"]["authors"])
                system_prompt += f"- {author_string}. [{result['meta']['title']}](https://doi.org/{result['meta']['doi']}). *{result['meta']['published']}*. {result['meta']['publication_year']}.\n"

            dois.add(result["meta"]["doi"])
        return [{"role": "system", "content": system_prompt.format(num_papers=len(dois))}, *conversation]


class FoundNonePromptBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation.
You found no papers that fit the users description in your database. 
Please ask the user to clarify which paper he or she is referring to."""

    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        system_prompt = self.system_prompt_template

        return [{"role": "system", "content": system_prompt}, *conversation]