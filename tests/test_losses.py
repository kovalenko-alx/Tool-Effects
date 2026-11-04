import torch

from etool.train.losses import drop_tokens


def test_drop_tokens_removes_ids_but_never_empties_a_row():
    ids = torch.tensor([[5, 7, 9, 7], [7, 7, 1, 1]])
    mask = torch.tensor([[False, True, True, True], [True, True, False, False]])
    out = drop_tokens(mask, ids, torch.tensor([7]))
    assert out.tolist() == [[False, False, True, False], [True, True, False, False]]  # row 2 would be empty -> unchanged
    assert out.dtype == torch.bool
