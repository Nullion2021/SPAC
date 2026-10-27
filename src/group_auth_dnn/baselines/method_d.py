"""Method D：TEE 扩展对照（RSA 签名模拟 Quote）"""

import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


def generate_rsa_keypair(seed: int = None):
    """生成 RSA 密钥对（确定性）"""
    import random
    if seed is not None:
        # 使用 seed 生成确定性密钥（仅用于实验）
        import os
        random.seed(seed)
        # 用 seed 派生伪随机字节作为 rsa 生成的熵源
        rng = random.Random(seed)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
    else:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
    public_key = private_key.public_key()
    return private_key, public_key


def sign_quote(private_key, data: str) -> bytes:
    """TEE Quote 模拟：对数据签名"""
    return private_key.sign(
        data.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )


def verify_quote(public_key, data: str, signature: bytes) -> bool:
    """验证 TEE Quote 签名"""
    try:
        public_key.verify(
            signature,
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def compute_tee_binding(sid: str, route_id: str, index: int,
                        device_id: str, u_i: str, v_i: str,
                        z_prev: str) -> str:
    """构造 TEE 签名的绑定消息"""
    msg = f"{sid}|{route_id}|{index}|{device_id}|{u_i}|{v_i}|{z_prev}"
    return hashlib.sha256(msg.encode()).hexdigest()
