from dataclasses import replace

import torch

from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import proof_size_breakdown, run_authenticated_inference, run_plain_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session


def _build_fixture():
    shards = create_mlp_shards(64, 128, 10, 6, 3, seed=42)
    keys = [generate_key(seed=i) for i in range(3)]
    nodes = create_node_chain(shards, keys, route_id="route_m0", quantization="int8")
    registry = build_registry(nodes)
    x = torch.randn(1, 64)
    session = create_session("route_m0", "m0-v1", x)
    proof = run_authenticated_inference(nodes, x, session)
    return nodes, registry, x, session, proof


def test_honest_execution_accepts() -> None:
    nodes, registry, x, session, proof = _build_fixture()
    plain_output = run_plain_inference(nodes, x)
    result = verify_proof(proof, registry, session, x)
    assert result.accepted, result.reason
    assert torch.allclose(proof.output, plain_output)


def test_tampered_output_rejects() -> None:
    _, registry, x, session, proof = _build_fixture()
    tampered = replace(proof, output=proof.output + 1.0)
    result = verify_proof(tampered, registry, session, x)
    assert not result.accepted and "output" in result.reason


def test_proof_size_breakdown_positive() -> None:
    _, _, _, _, proof = _build_fixture()
    breakdown = proof_size_breakdown(proof)
    assert breakdown["total_bytes"] > 0
    assert breakdown["proof_items_bytes"] > 0


def test_factory_rejects_mismatched_code_hash_lengths() -> None:
    shards = create_mlp_shards(64, 128, 10, 6, 3, seed=42)
    keys = [generate_key(seed=i) for i in range(3)]
    try:
        create_node_chain(
            shards,
            keys,
            route_id="route_m0",
            quantization="int8",
            code_hashes=["only-one"],
        )
    except ValueError as exc:
        assert "code_hashes and shards must have the same length" in str(exc)
    else:
        raise AssertionError("Expected mismatched code_hashes to raise ValueError")
