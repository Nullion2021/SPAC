"""TCP transport helpers for the M5 real-chain execution path."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import socket
import socketserver
import threading
import time

import numpy as np
import torch

from group_auth_dnn.protocol.crypto import canonical_json_bytes
from group_auth_dnn.protocol.node import AuthenticatedNode, HopProof
from group_auth_dnn.protocol.session import SessionContext


def tensor_to_payload(tensor: torch.Tensor) -> dict[str, object]:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "data_b64": base64.b64encode(array.astype("<f4", copy=False).tobytes(order="C")).decode("ascii"),
    }


def tensor_from_payload(payload: dict[str, object], device: torch.device | str = "cpu") -> torch.Tensor:
    if payload.get("dtype") != "float32":
        raise ValueError(f"Unsupported tensor payload dtype: {payload.get('dtype')}")
    raw = base64.b64decode(str(payload["data_b64"]).encode("ascii"))
    shape = tuple(int(dim) for dim in payload["shape"])
    array = np.frombuffer(raw, dtype="<f4").reshape(shape)
    return torch.from_numpy(array.copy()).to(device)


def session_to_payload(session: SessionContext) -> dict[str, object]:
    return {
        "route_id": session.route_id,
        "model_version": session.model_version,
        "sid": session.sid,
        "expires_at": session.expires_at,
        "nonce_hex": session.nonce_hex,
        "input_hash": session.input_hash,
        "z0": session.z0,
    }


def session_from_payload(payload: dict[str, object]) -> SessionContext:
    return SessionContext(
        route_id=str(payload["route_id"]),
        model_version=str(payload["model_version"]),
        sid=str(payload["sid"]),
        expires_at=float(payload["expires_at"]),
        nonce_hex=str(payload["nonce_hex"]),
        input_hash=str(payload["input_hash"]),
        z0=str(payload["z0"]),
    )


@dataclass(frozen=True)
class RemoteCallStats:
    request_bytes: int
    response_bytes: int
    roundtrip_ms: float
    compute_ms: float


@dataclass(frozen=True)
class RemoteHopResult:
    activation: torch.Tensor
    stats: RemoteCallStats
    worker_label: str
    worker_device: str
    proof_item: HopProof | None = None
    z_next: str | None = None


@dataclass(frozen=True)
class RemoteWorkerSpec:
    host: str
    port: int
    worker_label: str
    worker_device: str


@dataclass(frozen=True)
class WorkerServerHandle:
    host: str
    port: int
    server: socketserver.ThreadingTCPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)


class RemoteNodeClient:
    def __init__(self, spec: RemoteWorkerSpec, *, timeout_s: float = 30.0) -> None:
        self.spec = spec
        self.timeout_s = timeout_s

    def ping(self) -> dict[str, object]:
        response, _, _, _ = self._send_request({"kind": "ping"})
        return response

    def plain_forward(self, activation: torch.Tensor) -> RemoteHopResult:
        response, request_bytes, response_bytes, roundtrip_ms = self._send_request(
            {
                "kind": "plain",
                "activation": tensor_to_payload(activation),
            }
        )
        return RemoteHopResult(
            activation=tensor_from_payload(response["activation"]),
            stats=RemoteCallStats(
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                roundtrip_ms=roundtrip_ms,
                compute_ms=float(response["compute_ms"]),
            ),
            worker_label=str(response["worker_label"]),
            worker_device=str(response["worker_device"]),
        )

    def auth_forward(
        self,
        activation: torch.Tensor,
        session: SessionContext,
        z_prev: str,
        *,
        input_digest_override: str | None = None,
    ) -> RemoteHopResult:
        payload: dict[str, object] = {
            "kind": "auth",
            "activation": tensor_to_payload(activation),
            "session": session_to_payload(session),
            "z_prev": z_prev,
        }
        if input_digest_override is not None:
            payload["input_digest_override"] = input_digest_override
        response, request_bytes, response_bytes, roundtrip_ms = self._send_request(payload)
        return RemoteHopResult(
            activation=tensor_from_payload(response["activation"]),
            stats=RemoteCallStats(
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                roundtrip_ms=roundtrip_ms,
                compute_ms=float(response["compute_ms"]),
            ),
            worker_label=str(response["worker_label"]),
            worker_device=str(response["worker_device"]),
            proof_item=HopProof(**response["proof_item"]),
            z_next=str(response["z_next"]),
        )

    def _send_request(self, payload: dict[str, object]) -> tuple[dict[str, object], int, int, float]:
        request_raw = canonical_json_bytes(payload) + b"\n"
        start = time.perf_counter_ns()
        with socket.create_connection((self.spec.host, self.spec.port), timeout=self.timeout_s) as sock:
            sock.sendall(request_raw)
            sock.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        roundtrip_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        response_raw = b"".join(chunks)
        response_bytes = len(response_raw)
        response_text = response_raw.decode("utf-8").rstrip("\n")
        response = json.loads(response_text)
        if not response.get("ok", False):
            raise RuntimeError(
                f"Remote worker {self.spec.worker_label}@{self.spec.host}:{self.spec.port} failed: "
                f"{response.get('error', 'unknown error')}"
            )
        return response, len(request_raw), response_bytes, roundtrip_ms


class RemoteShardWorker:
    def __init__(self, node: AuthenticatedNode, *, worker_label: str, device: torch.device | str) -> None:
        self.node = node
        self.worker_label = worker_label
        self.device = torch.device(device)
        self._device_init_lock = threading.Lock()
        self._device_ready = self._module_on_target_device()
        self._activation_dtype = torch.float16 if self.device.type == "npu" else torch.float32

    def _module_on_target_device(self) -> bool:
        for tensor in list(self.node.model_shard.parameters()) + list(self.node.model_shard.buffers()):
            return tensor.device == self.device
        return True

    def _ensure_device_ready(self) -> None:
        if self._device_ready:
            return
        with self._device_init_lock:
            if self._device_ready:
                return
            if self.device.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "set_compile_mode"):
                torch.npu.set_compile_mode(jit_compile=False)
            module = self.node.model_shard.to(self.device)
            if self.device.type == "npu":
                module = module.half()
            self.node.model_shard = module.eval()
            self._device_ready = True

    def handle(self, payload: dict[str, object]) -> dict[str, object]:
        kind = str(payload["kind"])
        if kind == "ping":
            return {
                "ok": True,
                "worker_label": self.worker_label,
                "worker_device": str(self.device),
                "route_id": self.node.route_id,
                "measurement": self.node.measurement,
            }

        self._ensure_device_ready()
        activation = tensor_from_payload(payload["activation"], self.device)
        if activation.dtype != self._activation_dtype:
            activation = activation.to(dtype=self._activation_dtype)
        start = time.perf_counter_ns()
        if kind == "plain":
            with torch.no_grad():
                activation_out = self.node.model_shard(activation)
            compute_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            return {
                "ok": True,
                "activation": tensor_to_payload(activation_out),
                "compute_ms": compute_ms,
                "worker_label": self.worker_label,
                "worker_device": str(self.device),
            }
        if kind == "auth":
            session = session_from_payload(payload["session"])
            activation_out, proof_item, z_next = self.node.forward(
                activation,
                session,
                str(payload["z_prev"]),
                input_digest_override=payload.get("input_digest_override"),
            )
            compute_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            return {
                "ok": True,
                "activation": tensor_to_payload(activation_out),
                "proof_item": proof_item.to_dict(),
                "z_next": z_next,
                "compute_ms": compute_ms,
                "worker_label": self.worker_label,
                "worker_device": str(self.device),
            }
        raise ValueError(f"Unsupported request kind: {kind}")


class _WorkerTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _WorkerRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline()
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
            response = self.server.runtime.handle(payload)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - exercised via client failure path
            response = {"ok": False, "error": str(exc)}
        self.wfile.write(canonical_json_bytes(response) + b"\n")


def start_worker_server(
    worker: RemoteShardWorker,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
) -> WorkerServerHandle:
    server = _WorkerTCPServer((host, port), _WorkerRequestHandler)
    server.runtime = worker  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address
    return WorkerServerHandle(host=str(bound_host), port=int(bound_port), server=server, thread=thread)
