"""MolScaling pipeline package.

这个包把当前主线流程收拢到一个目录下：

clean record -> heavy atom graph -> FDDFS graph serialization ->
accelerated BPE -> encoded token stream -> packed Qwen3 pretraining.
"""

from .paths import PipelinePaths

__all__ = ["PipelinePaths"]
