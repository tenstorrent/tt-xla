# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code adapted from: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2
License: https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE
"""

import dataclasses
from enum import auto, Enum
from typing import List


class SeparatorStyle(Enum):
    TWO_NO_SYS = auto()


@dataclasses.dataclass
class Conversation:

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if self.sep_style == SeparatorStyle.TWO_NO_SYS:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        return ret

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


conv_mplug_owl2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO_NO_SYS,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_mplug_owl2
conv_templates = {
    "mplug_owl2": conv_mplug_owl2,
}
