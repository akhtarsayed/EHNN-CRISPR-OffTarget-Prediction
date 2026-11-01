"""Pure sequence-encoding helpers used by the EHNN pipeline.

These functions are intentionally dependency-free (no TensorFlow, pandas or
Biopython) so they can be unit-tested in CI without a GPU or a full
training environment. They are copied verbatim in behaviour from
``EHNN_Complete_Pipeline.py``.
"""

from itertools import product

NT_BITS = {"A": 0, "C": 1, "G": 2, "T": 3}
K = 3
KMER_VOCAB = {"".join(p): idx for idx, p in enumerate(product("ACGT", repeat=K))}
PAM_MAP = {p: i for i, p in enumerate(["TGG", "GAG", "GGG", "Other"])}


def encode_xor(off: str, on: str):
    """Return the XOR mismatch map between an off-target and on-target sequence."""
    return [NT_BITS.get(o, 0) ^ NT_BITS.get(i, 0) for o, i in zip(off.upper(), on.upper())]


def encode_kmer(seq: str):
    """Return the 3-mer vocabulary indices for a sequence."""
    vec = [KMER_VOCAB.get(seq[i : i + K].upper(), len(KMER_VOCAB)) for i in range(len(seq) - K + 1)]
    return vec if vec else [len(KMER_VOCAB)]


def encode_pam(pam: str):
    """Return the PAM category index, falling back to ``Other``."""
    return PAM_MAP.get(pam.upper(), PAM_MAP["Other"])
