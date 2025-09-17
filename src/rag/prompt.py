import abc
from typing import List, Dict, Any


class PromptBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build_prompt(self,
                     conversation: List[Dict[str, str]],
                     search_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        pass


class SystemPromptBuilder(PromptBuilder):
    def __init__(self):
        self.system_prompt_template = \
"""
You are a system called MetaRAG. Your task is to answer questions about retrieval-augmented generation. 
To help you answer questions of the users, you are provided a list of passages from scientific papers. 
Please add references in the text in a scientific format (e.g., [1], [2]).  
You may ignore these passages if they don't contain answers to the user's questions. 
Do not provide a list of references in the end. 

# References: 
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

            if not result["meta"]["doi"] in dois:
                dois.add(result["meta"]["doi"])
                i += 1

            system_prompt += (self.reference_template.format(id=i, content=result["text"]))
            system_prompt += "\n\n"

        return [{"role": "system", "content": system_prompt}, *conversation]