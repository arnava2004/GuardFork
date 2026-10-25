from guardfork.serve.parity import compare, first_mismatch


def test_first_mismatch():
    assert first_mismatch([1, 2, 3], [1, 2, 3]) is None
    assert first_mismatch([1, 2, 3], [1, 9, 3]) == 1
    assert first_mismatch([1, 2], [1, 2, 3]) == 2


def test_compare():
    r = compare("x", [1, 2], [1, 2])
    assert r.equal and r.first_mismatch is None
