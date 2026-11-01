import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehnn.encoding import KMER_VOCAB, encode_kmer, encode_pam, encode_xor


def test_encode_xor_identical_sequences_is_zero():
    assert encode_xor("ACGT", "ACGT") == [0, 0, 0, 0]


def test_encode_xor_single_mismatch():
    # A=0, C=1, so A xor C == 1
    assert encode_xor("A", "C") == [1]


def test_encode_kmer_length_and_vocab():
    result = encode_kmer("ACGT")
    assert len(result) == 2
    assert result[0] == KMER_VOCAB["ACG"]
    assert result[1] == KMER_VOCAB["CGT"]


def test_encode_kmer_short_sequence_falls_back():
    assert encode_kmer("AC") == [len(KMER_VOCAB)]


def test_encode_pam_known_and_unknown():
    assert encode_pam("TGG") == 0
    assert encode_pam("GAG") == 1
    assert encode_pam("GGG") == 2
    assert encode_pam("tgg") == 0
    assert encode_pam("AAA") == 3
