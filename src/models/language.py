from enum import Enum


class Language(str, Enum):
    PT_BR = "pt_br"
    EN = "en"

    @property
    def waiting_text(self) -> str:
        return {
            Language.PT_BR: "aguardando sinal…",
            Language.EN: "waiting for sign…",
        }[self]

    @property
    def no_hand_text(self) -> str:
        return {
            Language.PT_BR: "mão não detectada",
            Language.EN: "no hand detected",
        }[self]
