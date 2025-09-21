import pandas as pd
from ehnn.features import calculate_features

def test_calculate_features():
    f = calculate_features("ACGTACGT")
    assert "GC_Content" in f and "AT_Content" in f and "PAM_Sequence" in f
    assert f["Length"] == 8
