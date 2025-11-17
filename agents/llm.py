import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)
from dataclasses import dataclass
import re
from typing import Optional


SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

MEMORY_SYSTEMP_PROMPT = SYSTEM_PROMPT + (
    "In addition to the phrases you use to interact with the game, you can use memory tags to store information about the game or your thoughts for the future."
    "For example, if you output `get lamp <memory>... summary of what you have learned and plan for the future ...</memory>` you will be able to save this information for yourself (and the content of the memory will not be shown to the environment)."
    "Every time you use a memory tag, your memory scratchpad will be replaced with the content of the new memory."
    "The content of the memory will be shown to you at each step of the game."
)


class LLMAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "zero-shot",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            # For these models, we cannot set the seed.
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        action = response.text().strip()
        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=response.text()),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats

    def build_messages(self, observation):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


class LLMAgentWithMemory(LLMAgent):    
    @property
    def uid(self):
        return (
            f"LLMAgentWithMemory_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        res = super().params
        res['agent_type'] = 'zero-shot-with-memory'
        return res

    @staticmethod
    def extract_memory_from_action(action: str) -> tuple[str, Optional[str]]:
        """
        Returns [action without the memory tag, optional memory content]
        """

        matches = list(re.finditer(r"<memory>(.*?)</memory>", action, re.DOTALL))
        memory_content = matches[-1].group(1) if matches else None
        action = re.sub(r"<memory>.*?</memory>", "", action, re.DOTALL).strip()
        return action, memory_content

    def build_messages(self, observation):
        messages = [{"role": "system", "content": MEMORY_SYSTEMP_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        final_memory_scratchpad: Optional[str] = None

        for i, (obs, action, this_memory_scratchpad) in enumerate(self.history):
            # extract the memory scratchpad from the action and overwrite the action
            if this_memory_scratchpad is not None:
                final_memory_scratchpad = this_memory_scratchpad

            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            if i >= len(self.history) - limit:
                messages.append({"role": "user", "content": obs})
                messages.append({"role": "assistant", "content": action})

        if final_memory_scratchpad:
            messages.append({"role": "user", "content": f"The content of your memory scratchpad is:\n\n{final_memory_scratchpad}\n\nPlease use it to guide your actions. In order to update the memory scratchpad, you can use the <memory>...</memory> tag in your response to **REPLACE** the content of the memory scratchpad."})

        messages.append({"role": "user", "content": observation})

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
            "claude-3.7-sonnet",
        ]:
            # For these models, we cannot set the seed.
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        # extract the memory scratchpad from the response
        action, memory_scratchpad = self.extract_memory_from_action(response.text().strip())
        self.history.append((f"{obs}\n> ", f"{action}\n", memory_scratchpad))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "memory_scratchpad": memory_scratchpad,
            "response": response.text(),
            "nb_tokens_prompt": self.token_counter(messages=messages),
            "nb_tokens_response": self.token_counter(text=response.text()),
        }

        stats["nb_tokens"] = stats["nb_tokens_prompt"] + stats["nb_tokens_response"]

        return action, stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="zero-shot",
    desc=(
        "This agent uses a LLM to decide which action to take in a zero-shot manner."
    ),
    klass=LLMAgent,
    add_arguments=build_argparser,
)

register(
    name="zero-shot-with-memory",
    desc=(
        "This agent uses a LLM to decide which action to take in a zero-shot manner, but with memory."
    ),
    klass=LLMAgentWithMemory,
    add_arguments=build_argparser,
)
