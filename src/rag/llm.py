import abc
import logging
import os
from enum import Enum
from threading import Thread
from typing import List, Dict, Optional, Generator

import torch
from openai import OpenAI
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


class Precision(Enum):
    NF4 = "nf4"
    NF8 = "nf8"
    BF16 = "bf16"


class LLMVersion(Enum):
    # llama 3.1
    Llama_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    Llama_3_1_70B_INSTRUCT = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # llama 3.2
    Llama_3_2_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    Llama_3_2_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    # llama 3.3
    Llama_3_3_70B_INSTRUCT = "meta-llama/Llama-3.3-70B-Instruct"

    # gemma
    Gemma_2_2B_IT = "google/gemma-2-2b-it"
    Gemma_2_9B_IT = "google/gemma-2-9b-it"
    Gemma_2_27B_IT = "google/gemma-2-27b-it"

    Gemma_3_4B_IT = "google/gemma-3-4b-it"
    Gemma_3_27B_IT = "google/gemma-3-27b-it"

    # phi
    Phi_3_5_MINI_INSTRUCT= "microsoft/Phi-3.5-mini-instruct"
    Phi_3_5_MOE_INSTRUCT = "microsoft/Phi-3.5-MoE-instruct"
    Phi_3_MEDIUM_128K_INSTRUCT = "microsoft/Phi-3-medium-128k-instruct"
    PHI_4 = "microsoft/phi-4"
    PHI_4_MINI = "microsoft/Phi-4-mini-instruct"

    QWEN3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN3_4B = "Qwen/Qwen3-4B-Instruct-2507"

    SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"


class OpenAIModelVersion(Enum):
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_mini = "gpt-4.1-mini"
    GPT_4o_mini = "gpt-4o-mini"


class LLM(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> List[str]:
        pass

    @abc.abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> Generator[str, None, None]:
        pass

    @abc.abstractmethod
    def batch_generate(self, messages: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        pass


class HFModel(LLM, metaclass=abc.ABCMeta):
    """Base class for Hugging Face models."""
    def __init__(self, version: LLMVersion, quant_config: Optional[BitsAndBytesConfig] = None, **kwargs):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"Initialize model {version.value}")

        self.model_repo = version.value

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_repo,
            trust_remote_code=True,
            use_fast=True,
            add_eos_token=True,
            add_bos_token=True,
            padding_side="left"
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token

        cuda_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        if ("A100" in cuda_device_name) or ("H100" in cuda_device_name):
            self.logger.info("Using FlashAttention2")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_repo,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                **kwargs
            )
        else:
            self.logger.info("FlashAttention2 unavailable")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_repo,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                torch_dtype="auto",
                **kwargs
            )

        if not next(self.model.parameters()).is_cuda:
            try:
                self.model.to("cuda")
            except ValueError as e:
                self.logger.warning("Can't move model to cuda!")
                self.logger.warning(e)
                pass

        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)

    def tokenize_messages(self, messages: List[Dict[str, str]|List[Dict[str, str]]]):
        if isinstance(messages[0], list):
            return self.tokenizer.apply_chat_template(
                messages,
                return_tensors='pt',
                padding=True,
                pad_to_multiple_of=128,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=False).to("cuda")
        else:
            return self.tokenizer.apply_chat_template(
                messages, return_tensors='pt', add_generation_prompt=True, return_dict=True, enable_thinking=False).to("cuda")


    def generate(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> List[str]:
        inputs = self.tokenize_messages(messages)
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            **kwargs
        )

        out_ids = outputs.sequences[:,len(inputs.input_ids[0]):]
        out_texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return out_texts

    def batch_generate(self, messages: List[List[Dict[str, str]]],
                       **kwargs) -> List[str]:
        inputs = self.tokenize_messages(messages)
        gen_ids = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            cache_implementation="static",
            **kwargs
        )
        gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return outputs

    def generate_stream(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> Generator[str, None, None]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        inputs = self.tokenize_messages(messages)

        thread = Thread(target=lambda: self.model.generate(**inputs,
                            pad_token_id=self.tokenizer.eos_token_id,
                            streamer=streamer,
                            **kwargs))
        thread.start()

        for token in streamer:
            token = token.replace(self.tokenizer.eos_token, "")
            yield token



class HFModelQuantized(HFModel):
    def __init__(self, version: LLMVersion,
                 quantization: Precision = None,
                 ):
        bnb_config = None
        if quantization == Precision.NF4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == Precision.NF8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
        self.name = f"{version.value.split('/')[-1]}@{quantization.value}"
        super().__init__(version, bnb_config)

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return str(self.name)

class OpenAIModel(LLM):
    def __init__(self, model: OpenAIModelVersion):
        super().__init__()
        self.model_name = str(model.value)
        key = os.getenv("OPENAI_KEY")

        if key is None or key == "":
            raise EnvironmentError("OpenAI API key is missing.")

        logging.getLogger("httpcore.connection").setLevel(logging.INFO)
        logging.getLogger("httpcore.http11").setLevel(logging.INFO)
        self.client = OpenAI(api_key=key, project="proj_N9QKkLI9iESEPHCFdZudNHDj")

    def generate(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            modalities=["text"],
            **kwargs
        )

        outputs = [x.message.content for x in response.choices]

        return outputs

    def batch_generate(self, messages: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        raise NotImplemented()

    def generate_stream(self, messages: List[Dict[str, str]], tools=None, **kwargs) -> Generator[str, None, None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            modalities=["text"],
            stream=True,
            tools=tools,
            **kwargs
        )

        tool_call_string = ""
        arguments_string = ""
        for chunk in response:
            delta = chunk.choices[0].delta

            if delta.tool_calls is not None:
                if delta.tool_calls[0].function.name is not None:
                    tool_call_string += delta.tool_calls[0].function.name
                if delta.tool_calls[0].function.arguments is not None:
                    arguments_string += delta.tool_calls[0].function.arguments
            else:
                token = delta.content

                if token is not None:
                    yield chunk.choices[0].delta.content

        if tool_call_string != "":
            yield f"{tool_call_string}({arguments_string})"


