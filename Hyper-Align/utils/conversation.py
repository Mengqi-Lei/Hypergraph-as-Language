import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str | None = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                ret += role + ": " + message + self.sep if message else role + ":"
            return ret

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                ret += role + ": " + message + seps[i % 2] if message else role + ":"
            return ret

        if self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if self.version == "qwen3" and role == self.roles[1]:
                        ret += role + "<think>\n\n</think>\n\n" + message + self.sep
                    else:
                        ret += role + message + self.sep
                else:
                    ret += role
                    if self.version == "qwen3":
                        ret += "<think>\n\n</think>\n\n"
            return ret

        if self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    ret += (self.sep + wrap_inst(message)) if i % 2 == 0 else (" " + message + " " + self.sep2)
            return ret.lstrip(self.sep)

        if self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (_, message) in enumerate(messages):
                if message:
                    ret += message + seps[i % 2]
            return ret

        raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_qwen3 = Conversation(
    system=(
        "<|im_start|>system\n"
        "You are a helpful language and hypergraph assistant. You are able to "
        "understand the hypergraph structure that the user provides, and assist "
        "the user with a variety of tasks using natural language."
    ),
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="qwen3",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)


default_conversation = conv_qwen3
conv_templates = {
    "default": conv_qwen3,
    "qwen3": conv_qwen3,
}
