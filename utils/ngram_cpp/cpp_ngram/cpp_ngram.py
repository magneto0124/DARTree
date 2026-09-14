# -*- coding: utf-8 -*-

import os

from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

_cpp_ngram = None


def load_cpp_ngram():
    """Build (once) and return the C++ n-gram extension module.

    The first call triggers a one-time JIT compilation through
    ``torch.utils.cpp_extension.load``; the result is cached on disk under
    ``~/.cache/torch_extensions`` and reused by later runs.  The build is
    lazy so that evaluations that never use an n-gram model pay no compile
    cost.  Requires torch plus a C++20 compiler with OpenMP support
    (Linux: gcc/g++ >= 10 with libgomp).
    """
    global _cpp_ngram
    if _cpp_ngram is None:
        _cpp_ngram = load(
            name="dartree_ngram_cpp",
            sources=[
                f"{_abs_path}/ngram_binding.cpp",
                f"{_abs_path}/trie_ngram.cpp",
                f"{_abs_path}/cpp_utils/buffered_file_reader.cpp",
            ],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-fopenmp",
                "-DNGRAM_BATCH_LOAD",
            ],
            # Pure C++ module (no CUDA): skip CUDA include probing, which
            # also keeps Ascend hosts (torch without CUDA) working.
            with_cuda=False,
        )
    return _cpp_ngram
