from enum import Enum


class ZeroTags(Enum):
    FROZEN_EXPRESSIONS = "Frozen expressions are more than 75% of the entire SE"
    SH = "Too few words"
    IRR = "Too few content words"
    IRR_COH = "Extremely low cohesion"
    COPY_PASTE_PREV = "Copy & paste content words from previous"
    COPY_PASTE_TARGET = "Copy & paste content words from target"
    COPY_PASTE_BOTH = "Copy & paste content words from target and previous text"
