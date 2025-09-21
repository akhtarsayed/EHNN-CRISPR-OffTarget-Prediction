from ehnn.encoding import encode_xor, encode_kmer, encode_pam

def test_encode_xor():
    v = encode_xor("ACGT", "ACGA")
    assert len(v) == 4

def test_encode_kmer():
    v = encode_kmer("ACGT")
    assert len(v) >= 1

def test_encode_pam():
    assert isinstance(encode_pam("TGG"), int)
