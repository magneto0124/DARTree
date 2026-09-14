"""C++ n-gram extension (DART TrieNgram), vendored under ``cpp_ngram/``.

Keeps DART's original layout (``cpp_ngram/``, ``cpp_utils/``, ``inc/``) so the
C++ files are byte-identical to upstream and their relative includes
(``../inc/*.h``) resolve exactly as in DART.
"""

from .cpp_ngram.cpp_ngram import load_cpp_ngram

__all__ = ["load_cpp_ngram"]
