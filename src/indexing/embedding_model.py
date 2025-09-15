import abc
from typing import List

import torch
from sentence_transformers import SentenceTransformer


class EmbeddingModel(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.emb_kwargs = kwargs

    @abc.abstractmethod
    def embed_queries(self, queries: List[str]) -> List[torch.Tensor]:
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[torch.Tensor]:
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_id, query_prompt_name="s2p_query",**kwargs):
        super().__init__(**kwargs)
        model_kwargs = {}

        cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        if ("A100" in cuda_device_name) or ("H100" in cuda_device_name):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(model_id, trust_remote_code=True,
                                         model_kwargs=model_kwargs)
        self.query_prompt_name = query_prompt_name

    def embed_queries(self, queries: List[str]) -> List[torch.Tensor]:
        embeddings = self.model.encode(
            queries,
            batch_size=len(queries),
            query_prompt_name=self.query_prompt_name,
            show_progress_bar=False,
            **self.emb_kwargs)
        embeddings = [torch.tensor(emb) for emb in embeddings]

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[torch.Tensor]:
        embeddings = self.model.encode(texts, batch_size=len(texts), show_progress_bar=False, **self.emb_kwargs)
        embeddings = [torch.tensor(emb) for emb in embeddings]

        return embeddings