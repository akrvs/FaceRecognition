# Visage

Real-time face recognition as a service. Visage turns a face image into a 512-dimensional ArcFace embedding, indexes enrolled identities in a FAISS vector store, and exposes enrollment, recognition, and verification over a typed HTTP API.

[![CI](https://github.com/akrvs/Visage/actions/workflows/ci.yml/badge.svg)](https://github.com/akrvs/Visage/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Problem

Most open-source face recognition demos hard-code a folder of images, compute one embedding per person, and run a linear scan against that list on every frame. That design does not survive contact with production: matching is O(n) in the number of identities, the gallery cannot be updated without a restart, there is no service boundary, and accuracy is never measured. Visage rebuilds the same idea as a deployable system with a measured recognition backbone, sublinear vector search, and a clean API.

## Approach

A face image flows through three stages behind stable interfaces:

1. **Detection and embedding** — InsightFace (`buffalo_l`, RetinaFace + ArcFace) detects faces and produces L2-normalized 512-d embeddings.
2. **Vector index** — embeddings are stored in a FAISS inner-product index. With normalized vectors, inner product equals cosine similarity, so nearest-neighbour search ranks identities by face similarity. A pure-NumPy index is provided as a dependency-free fallback and as the substrate for fast tests.
3. **Recognition** — a query embedding is matched against the gallery; a configurable cosine threshold separates a known identity from `Unknown`.

```
                 +-------------------+        +------------------+
 image bytes --> |  InsightFace      | -----> |  FAISS / NumPy   |
                 |  detect + ArcFace |  512-d |  vector index    |
                 +-------------------+        +------------------+
                          |                            |
                          v                            v
                 +-------------------------------------------------+
                 |  FaceRecognizer: enroll / recognize / verify    |
                 +-------------------------------------------------+
                          |                            |
                          v                            v
                 +-----------------+          +------------------+
                 |  FastAPI app    |          |  Typer CLI       |
                 +-----------------+          +------------------+
```

## Design decisions

- **Interfaces over implementations.** `Embedder` and `VectorIndex` are protocols. The heavy InsightFace/FAISS backends are optional extras and are imported lazily, so the core logic, the API, and the full test suite run without any deep-learning dependency installed.
- **Cosine similarity via normalized inner product.** Embeddings are normalized at the source, which lets FAISS `IndexFlatIP` and the NumPy backend share identical semantics and makes thresholds portable across both.
- **Separation of gallery and vectors.** The vector index holds numerical labels; a `Gallery` maps labels to identity names and persists alongside the index, so multiple embeddings can back a single identity.
- **Configuration through environment.** Pydantic settings (`VISAGE_*`) drive model choice, thresholds, and storage paths with no code changes.

## Quick start

```bash
pip install -e ".[recognition]"

visage serve
```

```bash
curl -F "name=ada" -F "file=@ada.jpg" http://localhost:8000/enroll
curl -F "file=@query.jpg" http://localhost:8000/recognize
curl -F "file_a=@a.jpg" -F "file_b=@b.jpg" http://localhost:8000/verify
```

Interactive API docs are served at `http://localhost:8000/docs`.

### Docker

```bash
docker compose up --build
```

The InsightFace weights are downloaded once into a mounted volume on first start.

## API

| Method | Path          | Description                                            |
| ------ | ------------- | ------------------------------------------------------ |
| GET    | `/health`     | Service status, model name, and indexed vector count   |
| POST   | `/enroll`     | Add an identity from an uploaded image                  |
| POST   | `/recognize`  | Detect and identify every face in an uploaded image    |
| POST   | `/verify`     | One-to-one similarity between two faces                 |
| GET    | `/identities` | Enrolled identities and embedding counts               |

## Evaluation

Verification quality is measured on the LFW (Labeled Faces in the Wild) benchmark using balanced positive and negative pairs. Cosine similarity is scored per pair; accuracy is reported at the best global threshold and ranking quality as ROC-AUC.

```bash
./scripts/download_lfw.sh data/lfw
visage evaluate data/lfw --pairs 2000
```

| Metric                | Value  |
| --------------------- | ------ |
| Pairs (balanced)      | 1000   |
| Pairs with two faces  | 997    |
| Verification accuracy | 98.6%  |
| ROC-AUC               | 0.9924 |
| Best cosine threshold | 0.17   |

Produced by `visage evaluate` on CPU with the `buffalo_l` model and seed 42, on a balanced subset of same/different identity pairs (three pairs were skipped because a face was not detectable in both images). The default service threshold is set higher (0.35) to favour precision in an open-set enrollment setting, where rejecting unknown faces matters more than on a closed verification benchmark.

## Project layout

```
src/visage/
  config.py              Pydantic settings
  models.py              Pydantic schemas and the DetectedFace value type
  imaging.py             Image decoding
  embeddings/            Embedder protocol + InsightFace implementation
  index/                 VectorIndex protocol + FAISS and NumPy backends
  service/               FaceRecognizer and identity gallery
  api/                   FastAPI app, routes, dependency injection
  evaluation/            LFW pair builder and verification metrics
  cli.py                 Typer command line
tests/                   Unit and API tests (no GPU or model weights required)
```

## Testing

```bash
pip install -e ".[dev]"
make test
make lint
```

The suite injects a deterministic fake embedder and the NumPy index, so recognition, persistence, the metrics, and every API route are tested without downloading model weights. CI runs lint, type checking, and tests on Python 3.10 through 3.12.

## Roadmap

- Approximate nearest-neighbour index (IVF/HNSW) for galleries beyond a million identities
- Liveness and anti-spoofing stage before enrollment
- Quality gating on detection score, pose, and blur at enrollment time
- Batched and streaming recognition endpoints for video sources
- Persistent metadata store (PostgreSQL/pgvector) as an alternative backend
- Prometheus metrics and structured request tracing

## License

MIT
