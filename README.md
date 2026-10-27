# SPAC: Session-Bound Path Authentication for Distributed Neural Inference

**[Paper (PDF)](docs/static/pdfs/spac_paper.pdf)** | **[Project Page](https://nullion2021.github.io/SPAC/)**

SPAC (Session-bound Path Authentication Chain) is a chain-based group authentication
mechanism for distributed neural inference. It binds device identities, ordered hop
records, registered shard claims, and cross-hop encoded-activation continuity into a
single verifiable proof, turning "the designated distributed inference actually ran
through my devices" into a checkable authentication statement. It defends against
proxy-model substitution and topology-tampering attacks at trust boundaries.

> Distributed neural inference executes ordered model shards across independently
> managed devices, creating a need to authenticate session participation and
> activation transfers. In a corruption game where at most k of N devices may be
> compromised (k < N), a reduction to MAC unforgeability bounds fresh false
> attribution of honest-device participation. No accepted forgery was observed in
> the tested proxy, topology, and digest-inconsistent injection workloads, while
> SPAC adds only 0.65–2.56 ms per session in the evaluated MLP and ResNet
> configurations.

## Results Overview

| Dimension | Result |
|-----------|--------|
| Proxy-model attacks | SPAC FAR = 0 in all 252 measured cells; output-only baseline accepts the full-model substitute at ASR = 1 |
| Topology tampering | 4 attack types × 3 chain depths, FAR = 0 |
| Activation injection | Zero SPAC acceptance events in every covered MLP / ResNet-18 / ViT configuration (95% one-sided upper bound 0.93% per cell) |
| Quantization stability | QMatch = 1.0, FRR = 0 under the shared numeric configuration |
| System overhead | +0.65–2.56 ms per session; proof metadata grows linearly with chain length |

## Repository Layout

```text
├── docs/                    # Project page (GitHub Pages, static site)
├── src/group_auth_dnn/      # Reusable package code
│   ├── protocol/            # Core protocol: encoding, measurement, session, chain, node
│   ├── models/              # Split model definitions: MLP, ResNet-18, ViT
│   └── ...
├── experiments/             # Retained experiment drivers (main results + ablations)
├── scripts/                 # Public entrypoints for reproduction
├── tests/                   # Lightweight release checks
├── data/                    # Local datasets (created by prepare script)
├── checkpoints/             # Local pretrained weights (created by prepare script)
└── results/                 # Generated CSV / JSON / figures
```

## Getting Started

Python 3.10+ is recommended.

```bash
git clone https://github.com/Nullion2021/SPAC.git
cd SPAC
pip install -r requirements.txt
```

Prepare local datasets and optional pretrained checkpoints:

```bash
python scripts/prepare_data.py
python scripts/prepare_checkpoints.py
```

Run the retained main experiments (proxy attacks, topology tampering,
quantization stability, system overhead):

```bash
python scripts/run_main_experiments.py
```

Run the retained key ablations (measurement binding, freshness, activation
chains, chain state, signature mode):

```bash
python scripts/run_ablations.py
```

Generate the paper-facing plots and release summary tables:

```bash
python scripts/generate_figures.py
```

Artifacts are written under `results/`. See
[`scripts/README.md`](scripts/README.md) for details on each entrypoint.

## Expected Inputs

- CIFAR-10 under `data/cifar10/` (some drivers can download it automatically)
- Optional pretrained checkpoints under `checkpoints/mlp/`, `checkpoints/resnet/`,
  `checkpoints/transformer/`, `checkpoints/vit/`

## Project Page

The academic project page is hosted from the [`docs/`](docs/) folder of this
repository at <https://nullion2021.github.io/SPAC/>, built with the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

## Citation

If you use this code, please cite:

```bibtex
@misc{xiang2026spac,
  title        = {SPAC: Session-Bound Path Authentication for Distributed Neural Inference},
  author       = {Xiang, Zhengzhe and Chen, Yinlin and Cheng, Guanjie and Taheri, Javid and Deng, Shuiguang},
  year         = {2026},
  howpublished = {Manuscript submitted to IEEE Transactions on Dependable and Secure Computing},
  url          = {https://nullion2021.github.io/SPAC/}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE) (see also [NOTICE](NOTICE)).
