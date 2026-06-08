```
        ██╗   ██╗██╗███████╗ █████╗  ██████╗ ███████╗
        ██║   ██║██║██╔════╝██╔══██╗██╔════╝ ██╔════╝
        ██║   ██║██║███████╗███████║██║  ███╗█████╗
        ╚██╗ ██╔╝██║╚════██║██╔══██║██║   ██║██╔══╝
         ╚████╔╝ ██║███████║██║  ██║╚██████╔╝███████╗
          ╚═══╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
            f a c e   r e c o g n i t i o n   r i g
```

> A face walks past the lens. The rig lifts a 512-dimension signature, drops it into a vector index, and calls the match before the next frame lands. Enroll a target, then watch the box own every face it has seen before.

![status](https://img.shields.io/badge/status-ACTIVE-brightgreen)
![category](https://img.shields.io/badge/category-AI%20%2F%20Computer%20Vision-9cf)
![difficulty](https://img.shields.io/badge/difficulty-Hard-red)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

```
┌─[ MACHINE ]────────────────────────────────────────────────┐
│ codename    : Visage                                        │
│ category    : AI / Computer Vision                          │
│ difficulty  : Hard                                          │
│ stack       : InsightFace (ArcFace) · FAISS · FastAPI       │
│ interfaces  : REST API + Typer CLI                          │
│ flags       : user [enroll]   root [recognize @ scale]      │
│ status      : OWNED - 98.6% on LFW                          │
└─────────────────────────────────────────────────────────────┘
```

## [ Briefing ]

Visage turns a face image into a 512-dimensional ArcFace embedding, indexes enrolled identities in a FAISS vector store, and exposes enrollment, recognition, and one-to-one verification over a typed HTTP API and a CLI. It is the difference between a classroom demo and a service you could actually put behind a door.

## [ Recon ] - reading the target

Most open-source face recognition demos hard-code a folder of images, compute one embedding per person, and run a linear scan against that list on every frame. The design does not survive contact with production:

```
[!] matching is O(n) in the number of identities      -> melts under load
[!] the gallery cannot change without a restart        -> no live enrollment
[!] no service boundary                                 -> not callable, not deployable
[!] accuracy is never measured                          -> "trust me" is not a metric
```

Visage rebuilds the same idea as a deployable system with a measured recognition backbone, sublinear vector search, and a clean API.

## [ Attack Path ] - how a frame gets owned

```
   image bytes
        │
        v
 ┌───────────────────┐      detect + align faces, emit
 │  InsightFace      │      L2-normalized 512-d ArcFace
 │  RetinaFace+ArcFace│      embeddings
 └───────────────────┘
        │  512-d vectors
        v
 ┌───────────────────┐      cosine == inner product on
 │  FAISS / NumPy    │      normalized vectors -> nearest
 │  vector index     │      neighbour ranks identities
 └───────────────────┘
        │  top-k labels + scores
        v
 ┌─────────────────────────────────────────────┐
 │  FaceRecognizer                              │
 │  enroll  ·  recognize  ·  verify  ·  gallery │
 └─────────────────────────────────────────────┘
        │                           │
        v                           v
 ┌───────────────┐          ┌───────────────┐
 │  FastAPI      │          │  Typer CLI    │
 └───────────────┘          └───────────────┘
```

## [ Tradecraft ] - why it is built this way

- `[*]` **Interfaces over implementations.** `Embedder` and `VectorIndex` are protocols. The heavy InsightFace / FAISS backends are optional extras imported lazily, so the core logic, the API, and the full test suite run with zero deep-learning dependencies installed. A pure-NumPy index is the tested substrate.
- `[*]` **Cosine via normalized inner product.** Embeddings are normalized at the source, so FAISS `IndexFlatIP` and the NumPy backend share identical semantics and thresholds stay portable across both.
- `[*]` **Gallery split from vectors.** The index holds integer labels; a `Gallery` maps labels to identity names and persists beside the index, so multiple embeddings can back a single identity.
- `[*]` **Config as environment.** Pydantic settings (`VISAGE_*`) drive model choice, thresholds, and storage paths with no code edits.

## [ Arsenal ]

```
recognition : insightface (buffalo_l) · onnxruntime
search      : faiss-cpu (IndexFlatIP) · numpy fallback
service     : fastapi · uvicorn · pydantic v2 · typer
imaging     : pillow · numpy
quality     : pytest · ruff · mypy
ship        : docker · docker-compose · github actions
```

## [ Deploy ] - spawn the box

```bash
$ pip install -e ".[recognition]"
[+] backends online: insightface, faiss

$ visage serve
[+] listening on 0.0.0.0:8000   docs -> /docs
```

```bash
$ curl -F "name=ada" -F "file=@ada.jpg"        http://localhost:8000/enroll
[+] enrolled ada (1 embedding)

$ curl -F "file=@query.jpg"                     http://localhost:8000/recognize
[+] ada @ 0.71

$ curl -F "file_a=@a.jpg" -F "file_b=@b.jpg"    http://localhost:8000/verify
[+] match=true similarity=0.78
```

Containerized run:

```bash
$ docker compose up --build
[*] weights pulled once into the mounted volume on first start
```

## [ Attack Surface ] - endpoints

| Method | Path          | Action                                              |
| ------ | ------------- | --------------------------------------------------- |
| GET    | `/health`     | Service status, model, indexed vector count         |
| POST   | `/enroll`     | Register an identity from an uploaded image          |
| POST   | `/recognize`  | Detect and identify every face in an image           |
| POST   | `/verify`     | One-to-one similarity between two faces               |
| GET    | `/identities` | Enrolled identities and embedding counts             |

## [ Loot ] - proof of pwn

Verification measured on LFW (Labeled Faces in the Wild) with balanced same / different identity pairs. Cosine similarity scored per pair; accuracy at the best global threshold, ranking quality as ROC-AUC.

```bash
$ ./scripts/download_lfw.sh data/lfw
$ visage evaluate data/lfw --pairs 2000
```

```
┌─[ FLAG: recognition @ scale ]──────────────────┐
│ pairs (balanced)      1000                      │
│ pairs with two faces  997                       │
│ verification accuracy 98.6%                      │
│ ROC-AUC               0.9924                     │
│ best cosine threshold 0.17                       │
└─────────────────────────────────────────────────┘
```

Run on CPU with `buffalo_l`, seed 42 (three pairs skipped where a face was not detectable in both images). The default service threshold sits higher (0.35) to favour precision in open-set enrollment, where rejecting unknown faces matters more than on a closed verification benchmark.

## [ Layout ]

```
src/visage/
  config.py              pydantic settings
  models.py              schemas + the DetectedFace value type
  imaging.py             image decoding
  embeddings/            Embedder protocol + InsightFace backend
  index/                 VectorIndex protocol + FAISS and NumPy backends
  service/               FaceRecognizer + identity gallery
  api/                   FastAPI app, routes, dependency injection
  evaluation/            LFW pair builder + verification metrics
  cli.py                 Typer command line
tests/                   unit + API tests (no GPU or weights required)
```

## [ Test Range ]

```bash
$ pip install -e ".[dev]"
$ make test
[+] 28 passed

$ make lint
[+] ruff clean · mypy clean
```

The suite injects a deterministic fake embedder and the NumPy index, so recognition, persistence, the metrics, and every API route are exercised without downloading a single weight. CI replays lint, type checking, and tests on Python 3.10 through 3.12.

## [ Skill Tree ] - next objectives

- `[ ]` Approximate nearest-neighbour index (IVF / HNSW) for galleries past a million identities
- `[ ]` Liveness and anti-spoofing stage before enrollment
- `[ ]` Quality gating on detection score, pose, and blur at enrollment
- `[ ]` Batched and streaming recognition for video sources
- `[ ]` Persistent metadata store (PostgreSQL / pgvector) as a backend
- `[ ]` Prometheus metrics and structured request tracing

## [ Intel ]

MIT. Use it, fork it, build on it.
