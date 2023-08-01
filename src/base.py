from dataclasses import dataclass
from typing import Optional, List

SEPARATOR_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class Message:
    user: str
    text: Optional[str] = None

    def render(self):
        if self.user == "assistant":
            return {"role": "assistant", "content": self.text}

        if self.user == "system":
            return {"role": "system", "content": self.text}

        return {"role": "user", "content": self.text, "name": self.user}


@dataclass
class Conversation:
    messages: List[Message]

    def prepend(self, message: Message):
        self.messages.insert(0, message)
        return self

    def render(self):
        return [message.render() for message in self.messages]


@dataclass(frozen=True)
class Config:
    name: str
    instructions: str
    example_conversations: Optional[List[Conversation]]


@dataclass(frozen=True)
class Prompt:
    header: Message
    examples: List[Conversation]
    convo: Conversation

    def render(self):
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions#few-shot-learning-with-chat-completion
        return [
            self.header.render(),
            # For each conversation in our list of examples,
            # loop through each message in the conversation and
            # add the message dicts to the list we're going to return
            *[msg for conversation in self.examples for msg in conversation.render()],
            *self.convo.render(),
        ]
