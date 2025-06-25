import asyncio
import json
import logging
import pickle
import string
import uuid
from enum import Enum
from pathlib import Path


class EXAMPLE_TAG(Enum):
    CONTROL = 0
    EXPERIMENT = 1


OPENAI_RATE_LIMIT = 10
PRICING_DOLLAR_PER_1M_TOKEN = {
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
}

UNIT_1M = 1_000_000

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


def is_first_char_punctuation(s):
    if s and s[0] in string.punctuation:
        return True
    return False


class LanguageModelStats(object):
    """Main class for recording language model usage"""

    def __init__(self, model):
        self.model = model

        _uuid = str(uuid.uuid4())
        self.key = f"{model}-{_uuid}"
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}
        self.total_call = 0
        self.total_cache_hit = 0

    def record(self, api_name, stats, prompt=None, completion=None):
        self.total_call += 1
        if stats is None:
            self.total_cache_hit += 1
            return
        if api_name not in self.completion_tokens:
            self.completion_tokens[api_name] = []
        if api_name not in self.prompt_tokens:
            self.prompt_tokens[api_name] = []
        if api_name not in self.prompt_cache:
            self.prompt_cache[api_name] = []

        completion_tokens = int(stats["completion_tokens"])
        self.completion_tokens[api_name].append(completion_tokens)
        prompt_tokens = int(stats["prompt_tokens"])
        self.prompt_tokens[api_name].append(prompt_tokens)
        logger.debug(
            f"calling {api_name}, input tokens {prompt_tokens}, "
            f"output tokens {completion_tokens}"
        )
        if prompt != None:
            self.prompt_cache[api_name].append(
                {"prompt": prompt, "completion": completion}
            )

    def get_total_tokens(self, breakdown=True):
        sum_completion_tokens, sum_prompt_tokens = 0, 0
        for _, v in self.prompt_tokens.items():
            sum_prompt_tokens += sum(v)
        for _, v in self.completion_tokens.items():
            sum_completion_tokens += sum(v)
        if breakdown:
            return sum_prompt_tokens, sum_completion_tokens
        return sum_prompt_tokens + sum_completion_tokens

    def reset(self):
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}

    def get_total_price(self):
        input_tokens, output_tokens = self.get_total_tokens()
        input_price = (input_tokens / UNIT_1M) * PRICING_DOLLAR_PER_1M_TOKEN[
            self.model
        ]["input"]
        output_price = (input_tokens / UNIT_1M) * PRICING_DOLLAR_PER_1M_TOKEN[
            self.model
        ]["output"]
        return input_price + output_price

    def print_report(self):
        logger.warning("=" * 20)
        logger.warning(
            f"Total calls: {self.total_call}, Total cache hits: {self.total_cache_hit}"
        )
        logger.warning(f"Total price: ${self.get_total_price()}")
        logger.warning("=" * 20)

    def get_report(self):
        return {
            "total_calls": self.total_call,
            "total_cache_hits": self.total_cache_hit,
            "total_price": self.get_total_price(),
        }


class LanguageModel(object):
    """Main class abstract async remote language model access"""

    def __init__(
        self, model, client, dump_dir=None, use_cache=True, cache_level="api", **kwargs
    ):
        self.model = model
        self.stats = LanguageModelStats(model)
        self.client = client
        # dump dir
        if dump_dir:
            cur_save_dir = Path(dump_dir) / "lm_cache"
            cur_save_dir.mkdir(parents=True, exist_ok=True)
            self.dump_dir = cur_save_dir
        self.temperature = kwargs.get("temperature", 1.0)
        self.cache_dir = None
        self.use_cache = use_cache
        self.cache_level = cache_level
        self.cache_in_mem = {}
        self.api_count = {}
        if self.use_cache:
            assert kwargs.get("master_data_dir", None), (
                "master_data_dir is required for cache"
            )
            self.cache_dir = Path(kwargs["master_data_dir"]) / "persist_lm_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # load cache from disk
            if kwargs.get("cache_tag", None):
                self.cache_file = (
                    self.cache_dir / f"{self.model}_{kwargs['cache_tag']}_cache.pkl"
                )
            else:
                self.cache_file = self.cache_dir / f"{self.model}_cache.pkl"
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as f:
                    self.cache_in_mem = pickle.load(f)

    def normalize(self, text):
        return text.strip()

    def _get_cache_key(self, prompt, api_count, api_name):
        if self.cache_level and self.cache_level == "prompt":
            return f"{prompt}"
        return f"{prompt}_____{api_count}_____{api_name}"

    async def chat_completion(self, client, prompt, api_name, system_prompt=None):
        # check if the prompt is cached
        api_count = self.api_count.get(api_name, 0)
        self.api_count[api_name] = api_count + 1  # increment api count
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, api_count, api_name)
            if cache_key in self.cache_in_mem:
                return (self.cache_in_mem[cache_key], None)
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        raw_completion = await client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
        )
        raw_completion = raw_completion.to_dict()
        completion = self.normalize(raw_completion["choices"][0]["message"]["content"])

        if self.use_cache:
            self.cache_in_mem[cache_key] = completion
        usage = raw_completion["usage"]
        return (completion, usage)

    async def chat_completions(self, api_names, prompts, batch_size=32, system_prompts=None):
        """handling batched async calls with internal batching mechanism, now with system prompts support"""
        # Ensure api_names is a list of appropriate length
        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)

        # Handle system prompts
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif isinstance(system_prompts, list):
            assert len(system_prompts) == len(prompts), "system_prompts must be same length as prompts"
        else:
            raise ValueError("system_prompts must be None, a string, or a list of strings")

        # Process in batches
        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_api_names = api_names[i : i + batch_size]
            batch_system_prompts = system_prompts[i : i + batch_size]

            # batched calls
            async_responses = [
                self.chat_completion(self.client, prompt, api_name, system_prompt)
                for prompt, api_name, system_prompt in zip(batch_prompts, batch_api_names, batch_system_prompts)
            ]
            raw_completions = await asyncio.gather(*async_responses)
            # post handling for current batch
            for j, (completion, usage) in enumerate(raw_completions):
                all_completions.append(completion)
                self.stats.record(
                    batch_api_names[j],
                    usage,
                    prompt=batch_prompts[j],
                    completion=completion,
                )

        return all_completions

    def dump(self):
        with open(self.dump_dir / "tmp_prompt_cache.json", "w") as outfile:
            json.dump(self.stats.prompt_cache, outfile, indent=4)

        with open(self.dump_dir / "cost.jsonl", "a") as f:
            f.write(json.dumps({"price": self.stats.get_total_price()}) + "\n")

    def get_cost(self):
        return self.stats.get_total_price()

    def save_cache(self):
        if self.use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache_in_mem, f, protocol=pickle.HIGHEST_PROTOCOL)

    async def close(self):
        """Close the underlying HTTP client"""
        await self.client.close()
