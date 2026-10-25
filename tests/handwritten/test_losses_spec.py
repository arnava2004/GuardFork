import math

import pytest

from guardfork.train.losses import dpo_loss


def test_dpo_zero_margin_is_log2():
    assert dpo_loss(0.0, 0.0, 0.0, 0.0) == pytest.approx(math.log(2))


def test_dpo_prefers_chosen():
    assert dpo_loss(-1.0, -5.0, -3.0, -3.0, beta=0.1) < math.log(2)


def test_length_normalized_differs():
    plain = dpo_loss(-10.0, -10.0, -12.0, -11.0)
    norm = dpo_loss(-10.0, -10.0, -12.0, -11.0, chosen_len=10, rejected_len=2,
                    length_normalize=True)
    assert plain != pytest.approx(norm)
