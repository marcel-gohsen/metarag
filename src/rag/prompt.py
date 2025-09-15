import abc
from typing import List, Dict


class PromptBuilder(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def build_prompt(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        pass