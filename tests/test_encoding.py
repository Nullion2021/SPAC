import torch

from group_auth_dnn.protocol.encoding import encode_activation, hash_encode_activation


def test_activation_int8_deterministic() -> None:
    tensor = torch.randn(4, 8)
    first = encode_activation(tensor, mode="int8")
    second = encode_activation(tensor, mode="int8")
    assert first == second


def test_different_tensors_different_hash() -> None:
    t1 = torch.randn(4, 8)
    t2 = torch.randn(4, 8)
    assert hash_encode_activation(t1, mode="int8") != hash_encode_activation(t2, mode="int8")
