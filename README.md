# 🧬 Splice — HuggingFace Model Download + Weight-Swap Merge Demo

Splice is a small, fully working demo of an "agent" pipeline that:

1. **Downloads** a real, public model checkpoint from the HuggingFace Hub (pick one from a
   dropdown, or type any public repo id).
2. **Merges** two downloaded models with a simplified, educational stand-in for
   [mergekit](https://github.com/arcee-ai/mergekit): for every weight tensor the two models
   share (same name, shape, and dtype), it flips a weighted coin and keeps either model A's or
   model B's version. The result is a new, architecturally valid `merged.safetensors` checkpoint
   made of randomly spliced weights from both parents.

It's built as two small services in one Docker Compose stack:

| Layer    | Tech                                                              |
|----------|--------------------------------------------------------------------|
| Frontend | TypeScript + React + Vite, hand-styled (no component library)     |
| Backend  | Go, **standard library only** (no third-party deps to install)    |
| Storage  | A single named Docker volume holding downloaded + merged models    |

> **Important honesty note:** real `mergekit` implements serious merge algorithms (SLERP, TIES,
> DARE, task arithmetic, etc.) that combine weights mathematically. This demo does **not**
> reimplement those. It implements a much simpler, visual "random tensor swap" so you can see a
> full download → inspect → merge → download pipeline working end-to-end without needing a GPU
> or heavyweight ML dependencies. The merged model is a fun Frankenstein artifact, not a
> production-quality merged model. This is called out in the UI too.

---

## Table of contents

- [How it works](#how-it-works)
- [Directory structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Using the app](#using-the-app)
- [Scripts](#scripts-scriptsup.sh-downsh-cleansh)
- [API reference](#api-reference)
- [The merge algorithm, in detail](#the-merge-algorithm-in-detail)
- [The live catalog, compatibility checks, and the fallback list](#the-live-catalog-compatibility-checks-and-the-fallback-list)
- [The resultant weights: what you get, and what to do with them](#the-resultant-weights-what-you-get-and-what-to-do-with-them)
- [Local development (without Docker)](#local-development-without-docker)
- [Running the tests](#running-the-tests)
- [Troubleshooting](#troubleshooting)
- [License / disclaimer](#license--disclaimer)

---

## How it works

```
┌──────────────────────┐        HTTP / SSE        ┌───────────────────────────┐
│   Frontend (nginx)   │ ────────────────────────▶ │   Backend (Go, net/http)  │
│   React + TypeScript │ ◀──────────────────────── │                           │
│   :3000              │        JSON + logs         │   :8080                  │
└──────────────────────┘                            └─────────────┬─────────────┘
                                                                    │
                                                    HTTPS GET       │
                                                    (resolve/main)  ▼
                                                        ┌───────────────────────┐
                                                        │  huggingface.co Hub   │
                                                        │  (public, anonymous)  │
                                                        └───────────────────────┘
                                                                    │
                                                                    ▼
                                                     ┌───────────────────────────┐
                                                     │  Docker volume: model_data │
                                                     │  /data/models/<repo>/...  │
                                                     │  /data/merged/<job>/...   │
                                                     └───────────────────────────┘
```

1. You pick two models (Specimen A and Specimen B) and click **Download** on each. The backend
   calls the public HuggingFace Hub API to list the repo's files, downloads every `.safetensors`
   shard plus `config.json` (and the shard index if the model is split into multiple files), and
   streams progress back to the browser over Server-Sent Events (SSE).
2. Once both are downloaded, you set a **swap ratio** (0–100%) and an optional **seed**, then
   click **Merge models**. The backend loads both models' tensor indexes, and for every tensor
   present in both with matching shape + dtype, flips a weighted coin to decide whether the
   merged checkpoint keeps model A's or model B's copy of that tensor.
3. The result — a full decision-by-decision **tensor ledger** — streams into the UI, along with a
   downloadable `merged.safetensors` file and a `report.json` summary.

---

## Directory structure

```
hf-mergekit-demo/
├── README.md                  ← you are here
├── docker-compose.yml         ← orchestrates backend + frontend
├── scripts/
│   ├── up.sh                  ← build + start everything (frees ports first)
│   ├── down.sh                ← stop containers, keep downloaded models
│   └── clean.sh                ← stop + delete containers, images, volume, free ports
├── backend/                    ← Go agent + HTTP API
│   ├── cmd/server/main.go      ← entrypoint
│   ├── internal/
│   │   ├── models/             ← live HuggingFace Hub catalog (fetched + cached) + HF API client
│   │   ├── compat/              ← live tensor-shape compatibility checks (local + remote header probes)
│   │   ├── download/            ← HF file downloader
│   │   ├── store/                ← safetensors reader/writer (the file format engine)
│   │   ├── merge/                 ← the random tensor-swap "mergekit-lite" engine
│   │   ├── jobs/                   ← in-memory background job manager (SSE-friendly)
│   │   └── api/                     ← HTTP handlers
│   ├── go.mod
│   └── Dockerfile
└── frontend/                    ← TypeScript + React UI
    ├── src/
    │   ├── components/           ← ModelPicker, MergePanel, TensorLedger, Toast
    │   ├── lib/api.ts             ← typed fetch client (incl. SSE subscription helper)
    │   ├── types/api.ts           ← types mirroring the Go backend's JSON contracts
    │   ├── App.tsx
    │   └── styles.css             ← the "Splice" design system (no UI framework)
    ├── nginx.conf                 ← serves the built SPA + proxies /api to the backend
    ├── package.json
    └── Dockerfile
```

---

## Prerequisites

- **Docker** and **Docker Compose** (Docker Desktop on Mac/Windows, or Docker Engine + the
  `docker compose` plugin on Linux).
- **Internet access from wherever the containers run**, specifically to `huggingface.co` — the
  backend downloads real model files from the public Hub.
- Ports **3000** (frontend) and **8080** (backend) free on your machine, or override them (see
  below) — `up.sh` will also try to free them automatically if something else is squatting on
  them.

No Go or Node toolchain is required to run the demo — everything builds inside Docker. You only
need them for [local development without Docker](#local-development-without-docker).

---

## Quickstart

```bash
git clone <this-repo-or-unzip-it> hf-mergekit-demo
cd hf-mergekit-demo

chmod +x scripts/*.sh   # first time only, if permissions were lost in transit
./scripts/up.sh
```

Then open **http://localhost:3000**.

Custom ports:

```bash
BACKEND_PORT=9090 FRONTEND_PORT=4000 ./scripts/up.sh
```

To stop (keeping downloaded models for next time):

```bash
./scripts/down.sh
```

To wipe everything (containers, images, the model_data volume, freed ports):

```bash
./scripts/clean.sh
```

---

## Using the app

1. **Specimen A / Specimen B cards** — each has a dropdown populated **live from the HuggingFace
   Hub** (the current top-downloaded public models that publish `.safetensors` weights, refreshed
   every few minutes), plus a text field to type any other public HuggingFace repo id. Click
   **Download from HuggingFace**; a live console under the card streams progress (`Looking up ... /
   Downloading ... / saved ... (size) / Download complete`).
2. As soon as you explicitly pick a model in **either** dropdown, the *other* dropdown narrows to
   only models that a **live compatibility check** has confirmed actually share matching tensors
   (same name, shape, and dtype) with your pick — with a banner explaining how many of the catalog
   models qualified and why. This check inspects real tensor shapes (from already-downloaded local
   files, or via a tiny HTTP range request against the Hub if not downloaded yet) — it does **not**
   guess from a cosmetic architecture-family label, since same-family checkpoints (e.g. the three
   `TinyStories` sizes) can still have completely different tensor widths. If you bypass the
   dropdown with a custom repo id, the exact same live check still gates the **Merge models**
   button once both are downloaded.
3. Once **both** specimens show "downloaded" status (and the live check confirms them compatible),
   the **Splice controls** bar becomes active. Drag the **swap ratio** slider (0% = merged output
   is identical to model A, 100% = every shared tensor is taken from model B), optionally set a
   specific **seed** for reproducibility (or roll the 🎲 for a random one), then click **Merge
   models**.
4. The **tensor ledger** fills in live: every shared tensor's name, shape, dtype, size, and which
   specimen it was ultimately sourced from (tensors taken from B are flagged `SPLICED`). A stat
   strip at the top summarizes common tensors, swap count, ratio, and output size.
5. Click **⬇ Download merged.safetensors** to save the merged checkpoint locally. A
   `report.json` with the full decision log is also written next to it inside the container
   volume (see [API reference](#api-reference) if you want to fetch it programmatically).

---

## Scripts (`scripts/up.sh`, `down.sh`, `clean.sh`)

| Script         | What it does                                                                                   |
|----------------|--------------------------------------------------------------------------------------------------|
| `up.sh`        | Kills anything squatting on the backend/frontend ports, builds both images, starts the stack, waits for the backend health check, then prints the URLs. |
| `down.sh`      | `docker compose down` — stops and removes the containers, but **keeps** the `model_data` volume (downloaded/merged models persist for next time). |
| `clean.sh`     | Asks for confirmation, then stops containers, **deletes the model_data volume** (all downloaded/merged models are lost), removes the built images, and frees the ports again. Use this for a completely fresh start. |

All three scripts are idempotent and safe to re-run.

---

## API reference

The backend exposes a small JSON + SSE API under `/api`. The frontend calls it through nginx's
proxy in production, or Vite's dev proxy in local dev — both forward `/api/*` unchanged.

| Method | Path                              | Description                                                             |
|--------|------------------------------------|---------------------------------------------------------------------------|
| GET    | `/api/health`                       | Liveness check, `{"status":"ok"}`                                        |
| GET    | `/api/models`                        | The live HuggingFace catalog (id, label, description, approx size) — refetched from the Hub every ~10 minutes, cached in between |
| GET    | `/api/models/local`                   | Models already downloaded into the volume, with file listing + sizes   |
| GET    | `/api/models/compat?base={id}`        | Live tensor-shape compatibility of every catalog + locally-downloaded model against `{id}` — the source of the dropdown gating and the merge-button gate |
| POST   | `/api/jobs/download`                   | Body `{"modelId":"org/name"}` → `{"jobId":"..."}`, starts a background download |
| POST   | `/api/jobs/merge`                        | Body `{"modelAId","modelBId","swapRatio","seed"}` → `{"jobId":"..."}`   |
| GET    | `/api/jobs/{id}`                          | Current status, full log, error (if any), and result (if finished)      |
| GET    | `/api/jobs/{id}/events`                    | Server-Sent Events stream of log lines; sends a `done` event at the end |
| GET    | `/api/jobs`                                  | Lists all jobs known to this backend process (summary only)             |
| GET    | `/api/merged/{jobId}/download`                | Streams the merged `merged.safetensors` file for a succeeded merge job  |

All job state is **in-memory** and reset when the backend container restarts (the downloaded and
merged *files* on disk survive restarts via the `model_data` volume — only the job history/log is
ephemeral).

---

## The merge algorithm, in detail

Real HuggingFace checkpoints store weights in the
[safetensors](https://github.com/huggingface/safetensors) format: an 8-byte length prefix, a JSON
header describing every tensor's name/dtype/shape/byte-offsets, followed by the raw tensor bytes.
`backend/internal/store/safetensors.go` implements a small, dependency-free reader/writer for
this format directly in Go (no Python, no PyTorch, no C bindings).

`backend/internal/merge/merge.go` then does the following for a merge job:

1. Load a unified tensor index for model A and model B (handling both single-file and
   sharded/multi-file checkpoints).
2. For every tensor name present in **both** models with an identical shape and dtype: draw a
   uniform random number and take model B's copy if it's below the configured swap ratio,
   otherwise keep model A's copy. This decision, and the byte size of every tensor, is recorded.
3. Tensors that exist in only one model, or that exist in both but with mismatched shape/dtype,
   are kept from model A untouched (and reported as such) — this keeps the merged checkpoint
   architecturally consistent with model A's `config.json`, which is copied alongside the output.
4. The chosen bytes are streamed into a new `merged.safetensors` file with correctly recomputed
   byte offsets, plus a `report.json` with the full per-tensor decision log (this is exactly what
   powers the UI's tensor ledger).

This is intentionally simple so it's easy to read, audit, and extend — e.g. you could swap the
coin-flip for a proper SLERP interpolation between two same-shaped float tensors if you wanted to
take this further.

---

## The live catalog, compatibility checks, and the fallback list

`backend/internal/models/catalog.go`'s `GetCatalog()` fetches the current top-downloaded public
models with safetensors weights directly from the HuggingFace Hub search API (`GET
https://huggingface.co/api/models`), filters out private/disabled/gated repos, and caches the
result for ~10 minutes so `/api/models` stays fast. If that fetch ever fails (offline dev, Hub
outage) it falls back to the last successful result, or — on a cold start with no network at all —
a small built-in seed list (`fallbackCatalog`) so the dropdowns are never empty. There's nothing to
edit to "add a model" anymore: type any public repo id in the "or type any public repo id" field
and it'll be downloaded and compatibility-checked like any catalog entry.

`backend/internal/compat/compat.go` is what actually decides which models can be merged. It builds
a *signature* for a model — every tensor's name, shape, and dtype — either by reading an
already-downloaded model's local safetensors header (exact, instant), or, for a model still on the
Hub, by issuing a small HTTP Range request for just the first few KB of its `.safetensors` file
(where the JSON header lives) so compatibility can be checked **without downloading the full
weights**. `compat.Compare` is the single function both the dropdown-filtering endpoint
(`GET /api/models/compat`) and, indirectly, the pre-merge gate rely on, so the UI and the merge job
itself can never disagree about what counts as "compatible."

---

## The resultant weights: what you get, and what to do with them

A successful merge job writes three files into the `model_data` volume, under
`/data/merged/<jobId>/`:

| File                  | What it is                                                                                     |
|-----------------------|--------------------------------------------------------------------------------------------------|
| `merged.safetensors`  | The actual spliced checkpoint — every shared tensor is either model A's or model B's original bytes, verbatim (never averaged/interpolated), with correctly recomputed header offsets. |
| `config.json`         | Copied from model A unmodified, so downstream tooling knows the architecture/shape metadata to load the tensors with. |
| `report.json`         | The full per-tensor decision log: name, shape, dtype, byte size, and which specimen (`A`/`SPLICED`=B) it came from — this is exactly what backs the UI's tensor ledger. |

You can get them out of the demo two ways:

- **From the UI:** click **⬇ Download merged.safetensors** in the tensor ledger once a merge
  succeeds (this hits `GET /api/merged/{jobId}/download`).
- **From the volume directly:** `docker compose exec backend sh -c "ls /data/merged/<jobId>"`, or
  mount/copy out of the named `model_data` volume with `docker cp`.

**Using the merged checkpoint elsewhere.** `merged.safetensors` + `config.json` are enough for
libraries like 🤗 `transformers` to load the weights, but they are **not** a complete, ready-to-run
model repo on their own — you'll typically also want the tokenizer files (`tokenizer.json`,
`tokenizer_config.json`, `vocab.json`/`merges.txt`, etc.) from model A's original Hub repo, since
this demo doesn't merge or copy those. A typical local load looks like:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

merged_dir = "./merged"  # merged.safetensors + config.json copied here
model = AutoModelForCausalLM.from_pretrained(merged_dir)
tokenizer = AutoTokenizer.from_pretrained("org/model-a-repo-id")  # original A repo, for the tokenizer
```

**Before you do anything serious with it:** re-read the honesty note at the top of this README.
Because tensors are swapped wholesale (never averaged/interpolated) at a per-tensor coin flip, the
result is frequently an incoherent "Frankenstein" model, especially at swap ratios far from 0% or
100%, or between checkpoints whose *matching* tensors were nonetheless trained very differently.
Treat `merged.safetensors` as a fun artifact to inspect via the tensor ledger, not something to
deploy or fine-tune from without first evaluating its actual outputs. And since it's built from
downloaded HuggingFace weights, it inherits **both** parent models' original licenses — check each
model's card on the Hub before further use or redistribution.

---

## Local development (without Docker)

**Backend:**

```bash
cd backend
go run ./cmd/server        # listens on :8080, stores data in ./data by default if you set DATA_DIR
# or explicitly:
PORT=8080 DATA_DIR=/tmp/splice-data go run ./cmd/server
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev                # Vite dev server on :5173, proxies /api to http://localhost:8080
```

Then open http://localhost:5173. If your backend runs on a different port, set
`VITE_BACKEND_ORIGIN` before starting Vite:

```bash
VITE_BACKEND_ORIGIN=http://localhost:9090 npm run dev
```

---

## Running the tests

**Backend (Go):**

```bash
cd backend
go build ./...      # compiles everything
go vet ./...         # static analysis
go test ./... -v      # unit tests for the safetensors engine and merge algorithm
```

The test suite covers:
- Safetensors write → read round-trips (including multi-shard models)
- The merge engine's swap logic at ratio=0 (keeps model A), ratio=1 (takes all of model B), and
  the error path when two models share no compatible tensors

**Frontend (TypeScript):**

```bash
cd frontend
npm install
npm run build        # tsc --build (strict mode) + vite build; fails on any type error
npm test              # runs the jsdom-based component/integration tests below
```

The frontend test suite mounts the **real** `App` component in a simulated DOM (via `vitest` +
`jsdom` + React Testing Library) against a fake backend and drives it through real user flows:
`App.e2e.test.tsx` loads the catalog, downloads two models, confirms `Merge models` becomes
enabled, runs a merge, and checks the tensor ledger renders correctly; `App.compatgate.test.tsx`
asserts that picking one model live-filters the other dropdown down to only confirmed-compatible
options (with a visible explanation) and that the merge button stays blocked if an incompatible
pair is reached anyway via a custom repo id. These exercise the literal production code path
rather than just asserting on isolated units.

---

## Troubleshooting

- **"Port already in use" when starting** — `up.sh` already tries to free ports 3000/8080
  automatically. If something still conflicts, override the ports:
  `BACKEND_PORT=9090 FRONTEND_PORT=4000 ./scripts/up.sh`.
- **"no .safetensors weight files were found in ..."** — the repo you picked doesn't publish
  safetensors weights (some older repos only ship `pytorch_model.bin`). Pick a different model —
  everything in the built-in dropdown is verified to publish safetensors.
- **"model A and model B share no tensors with matching name+shape+dtype"** — this should be rare
  now, since the dropdown only offers live-confirmed-compatible pairs; it can still happen if you
  typed a custom repo id for one side that bypasses the filter. Use the catalog dropdown, or check
  the live compatibility banner/reason shown for your custom pick before merging.
- **Download seems stuck** — larger models (like `gpt2` at ~550 MB) take longer; watch the live
  console under the specimen card for progress. Check `docker compose logs backend` for details.
- **Backend never becomes healthy** — check `docker compose logs backend`; the most common cause
  is no internet access to `huggingface.co` from inside the container (corporate proxies/VPNs
  sometimes block this).
- **Page goes blank, or "Merge models" never becomes clickable** — this was caused by a real bug
  (a Go nil-slice serializing to JSON `null` instead of `[]`) that's fixed and covered by
  regression tests in both the backend (`go test ./internal/api/...`) and frontend
  (`npm test`, which drives the actual download → merge flow in a simulated browser DOM). If you
  still see this:
  1. Make sure you're on the latest copy of this project (re-download/re-unzip if unsure).
  2. Force a completely fresh rebuild — stale Docker image layers are the most common cause of
     "I fixed it but it's still broken": `./scripts/clean.sh` then `./scripts/up.sh`.
  3. Open the browser's DevTools console (F12) right when it happens and check for a red error —
     that message tells you exactly what broke, and is the single most useful thing to share when
     reporting a bug.
  4. `docker compose logs backend` and `docker compose logs frontend` for server-side errors.

---

## License / disclaimer

This is an educational demo, not a production tool. It is not affiliated with or endorsed by
Hugging Face or the `mergekit` project. Model weights downloaded through this tool remain subject
to their original licenses on the HuggingFace Hub — check each model's card before further use or
redistribution of anything you merge here.
