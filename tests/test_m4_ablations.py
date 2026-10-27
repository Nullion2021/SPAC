import torch

from group_auth_dnn.attacks.activation_injection import run_activation_injection_attack
from group_auth_dnn.baselines.baseline_e import build_cra_registry, create_cra_node_chain, verify_cra_proof
from group_auth_dnn.baselines.no_activation_chain import verify_proof_wo_activation_chain
from group_auth_dnn.config import generate_key
from group_auth_dnn.models.mlp import create_mlp_shards
from group_auth_dnn.protocol.chain import run_authenticated_inference, verify_proof
from group_auth_dnn.protocol.node import build_registry, create_node_chain
from group_auth_dnn.protocol.session import create_session


def _build_fixture():
    shards = create_mlp_shards(64, 128, 10, 6, 3, seed=42)
    keys = [generate_key(seed=i) for i in range(3)]
    route_id = "route_m4_test"
    method_nodes = create_node_chain(shards, keys, route_id, quantization="int8")
    baseline_nodes = create_cra_node_chain(shards, keys, route_id)
    method_registry = build_registry(method_nodes)
    baseline_registry = build_cra_registry(baseline_nodes)
    return method_nodes, baseline_nodes, method_registry, baseline_registry, route_id


def test_activation_chain_ablation_accepts_attack_that_method_c_rejects() -> None:
    method_nodes, baseline_nodes, method_registry, baseline_registry, route_id = _build_fixture()
    x_input = torch.randn(1, 64)
    alternate_input = torch.randn(1, 64)
    session = create_session(route_id, "m4-mlp-v1", x_input)
    attack = run_activation_injection_attack(
        method_nodes,
        baseline_nodes,
        x_input,
        alternate_input,
        session,
        target_hop=2,
        strength=0.2,
    )

    baseline_verdict = verify_cra_proof(attack.baseline_bundle, baseline_registry, session)
    ablation_verdict = verify_proof_wo_activation_chain(attack.method_bundle, method_registry, session, x_input)
    method_verdict = verify_proof(attack.method_bundle, method_registry, session, x_input)

    assert baseline_verdict.accepted, baseline_verdict.reason
    assert ablation_verdict.accepted, ablation_verdict.reason
    assert not method_verdict.accepted and "activation chain mismatch" in method_verdict.reason
