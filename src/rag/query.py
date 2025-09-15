import abc
from typing import List, Dict


class QueryFormulator(metaclass=abc.ABCMeta):
    """Extracts search queries from a conversational context."""

    @abc.abstractmethod
    def get_queries(self, conversation: List[Dict[str, str]]) -> List[str]:
        pass

class IdentityFormulator(QueryFormulator):
    """Treats last user message as user query."""

    def get_queries(self, conversation: List[Dict[str, str]]) -> List[str]:
        return [conversation[-1]["content"]]