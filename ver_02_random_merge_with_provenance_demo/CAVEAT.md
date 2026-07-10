# ⚠️ Caveats: misuse potential, and how this project guards against it

This document is about a different question than the rest of the README. The README explains how
Splice works and how to use it well. This document is about **how it could be used badly**, and
what's been done - both in the code and in how you should operate this tool - to make that harder.
Read it before you deploy this anywhere beyond your own laptop, and especially before you point it
at real users or redistribute anything it produces.

---

## Table of contents

- [The short version](#the-short-version)
- [What this tool genuinely cannot do](#what-this-tool-genuinely-cannot-do)
- [Realistic misuse scenarios](#realistic-misuse-scenarios)
  - [1. Model laundering — hiding a modified model inside an "innocent merge"](#1-model-laundering--hiding-a-modified-model-inside-an-innocent-merge)
  - [2. Passing off a spliced model as a trustworthy, original checkpoint](#2-passing-off-a-spliced-model-as-a-trustworthy-original-checkpoint)
  - [3. License laundering / redistribution violations](#3-license-laundering--redistribution-violations)
  - [4. Using the download pipeline to move data you don't control](#4-using-the-download-pipeline-to-move-data-you-dont-control)
  - [5. Resource-exhaustion / cost abuse if exposed publicly](#5-resource-exhaustion--cost-abuse-if-exposed-publicly)
- [What's built in to reduce these risks](#whats-built-in-to-reduce-these-risks)
- [What you're responsible for if you deploy or extend this](#what-youre-responsible-for-if-you-deploy-or-extend-this)
- [If you're evaluating a `merged.safetensors` file someone gave you](#if-youre-evaluating-a-mergedsafetensors-file-someone-gave-you)
- [Reporting a concern](#reporting-a-concern)

---

## The short version

Any tool that downloads model weights and recombines them can, in principle, be pointed at
producing a model that behaves differently from what its outward description claims. Splice is
**not designed to hide, obscure, or launder that kind of modification** - it's built to do the
opposite: every merge is loud, logged, and stamped with exactly where every tensor came from. But
"not designed for X" isn't the same as "impossible to misuse for X," so this document spells out
the gap plainly instead of leaving it implied.

**Is a document like this actually sufficient protection? No.** A markdown file cannot stop a
determined actor with access to this (open) source code from deleting the disclaimer, stripping
the metadata-embedding code, or just using a different tool entirely - documentation cannot enforce
anything against someone willing to modify the code that would enforce it. What a document like
this *can* do, honestly:

- Set correct expectations for people using this in good faith, so misuse isn't accidental.
- Push as much protection as possible into the *artifact itself* rather than the paperwork around
  it - provenance metadata embedded inside `merged.safetensors`, not just described in a README,
  survives copying, sharing, and casual handling in a way a separate warning label doesn't.
- Push as much protection as possible into *default configuration* rather than advice - e.g.
  binding to `127.0.0.1` by default so nobody has to remember to lock it down, rather than telling
  them to remember to.
- Give downstream recipients of a merged file (who never read this repo at all) a concrete way to
  check its claims for themselves, covered below.

So: this document exists alongside several actual code-level changes (listed under
[What's built in](#whats-built-in-to-reduce-these-risks)), not instead of them. If you only take
the document and skip the code-level defaults, you have weaker protection than what ships here.
If you take one thing from this page: **never treat a `merged.safetensors` file as trustworthy
just because it loads and runs.** The safetensors *format* can't hide malicious code (see below),
but the *weights themselves* can still encode arbitrary behavior, and no file-format-level check
can tell you whether a given checkpoint is safe to deploy. That takes actual evaluation.

---

## What this tool genuinely cannot do

Some categories of harm this class of tool is sometimes associated with are architecturally ruled
out here, and it's worth being specific about why:

- **No arbitrary code execution via the model file.** Older HuggingFace checkpoints shipped as
  `pytorch_model.bin` (a Python `pickle`), and pickle deserialization can execute arbitrary code -
  a real, well-documented supply-chain risk in the ML ecosystem. Splice only reads and writes the
  [safetensors](https://github.com/huggingface/safetensors) format, which is a flat binary layout
  (an 8-byte length prefix, a plain JSON header, then raw tensor bytes) with **no executable
  content whatsoever**. `backend/internal/store/safetensors.go` never deserializes anything beyond
  `encoding/json` on a small header. Loading a `merged.safetensors` file, however it was produced,
  cannot run code as a side effect of loading it.
- **No network egress beyond the HuggingFace Hub.** The download and compatibility-check code
  (`internal/download`, `internal/compat`) only ever constructs URLs under `huggingface.co`
  (`.../api/models/...` and `.../resolve/main/...`). There's no way to pass this tool an arbitrary
  URL, internal hostname, or file:// path to fetch - the repo id you provide only ever gets
  interpolated into a fixed `huggingface.co` URL template. This rules out using the download
  pipeline as a generic SSRF (server-side request forgery) proxy.
- **No hidden mutation of tensor values.** The merge algorithm (`internal/merge/merge.go`) only
  ever does one of two things to a shared tensor: keep model A's bytes untouched, or keep model
  B's bytes untouched. It never averages, perturbs, or otherwise computes a new tensor value. That
  means the worst a merge can do to any individual tensor is "it's from a different (but real, and
  disclosed) source model" - not "it's been subtly altered in an undetectable way by this tool."

None of this means a *merged model's behavior* is safe - see below - only that the mechanics of
*this specific tool* don't open the usual code-execution or data-exfiltration doors.

---

## Realistic misuse scenarios

### 1. Model laundering — hiding a modified model inside an "innocent merge"

**The risk:** someone fine-tunes a model to behave maliciously (e.g. a backdoor that activates on
a specific trigger phrase, or content designed to embarrass/harm when a certain input is seen),
then uses a merge tool as a way to blend just enough of that fine-tune into an otherwise normal
model that casual inspection ("it mostly behaves like the base model") doesn't catch it, while
still delivering the malicious behavior for the attacker's trigger.

**Why this isn't specific to Splice:** this is a risk with *any* weight-merging tool, including
real ones like `mergekit` - it's a property of model merging in general, not something Splice
introduces. Splice's random, whole-tensor-swap approach is actually a comparatively *poor* vector
for this compared to real interpolation-based merges: because every shared tensor is either 100%
model A or 100% model B (never blended), a targeted backdoor concentrated in a few tensors is
*more* likely to either fully survive or fully disappear per merge, not partially and subtly leak
through - which makes naive "just merge a little bit of the backdoored model in" attacks less
reliable here than with gradient-style merges. It is not, however, impossible.

**Mitigation:** every merge embeds full provenance (see below) directly in the output file's
metadata and in a sibling `report.json` - who the two parent models were, the exact swap ratio and
seed, and a full per-tensor decision log. This doesn't prevent someone from *choosing* a malicious
parent model, but it does mean the choice is never hidden: anyone downstream can see exactly which
two HuggingFace repos contributed which tensors, and re-fetch and diff against either parent to
confirm nothing else was altered.

### 2. Passing off a spliced model as a trustworthy, original checkpoint

**The risk:** taking `merged.safetensors`, stripping identifying information, and redistributing
it as if it were an official/unmodified checkpoint from a reputable org - to gain trust the model
hasn't earned, or to disguise the fact that its behavior was intentionally altered.

**Mitigation:** the disclosure is embedded *inside the file itself* (the safetensors
`__metadata__` block), not just in a separate, easy-to-drop report. Provenance fields
(`merged_by`, `model_a`, `model_b`, `swap_ratio`, `seed`, `created_at`) and an explicit
human-readable `disclaimer` field travel with the tensor data as long as the file exists. Stripping
them requires deliberately rewriting the safetensors header - i.e. active, intentional tampering,
not something that happens by accident through normal handling/copying. Anyone loading the file
with any safetensors-aware tool (including the `safetensors` Python library) can trivially inspect
this metadata before trusting the weights.

### 3. License laundering / redistribution violations

**The risk:** merging two models with incompatible or restrictive licenses and redistributing the
result as if it had a clean/permissive license, obscuring the original licensing obligations.

**Mitigation:** this is a policy/legal problem, not something a tool can fully solve technically -
but the embedded provenance metadata means the *originating repos are always identifiable* from
the output file alone, so anyone doing license due diligence downstream has what they need to
trace back to both original model cards. See [License / disclaimer](README.md#license--disclaimer)
in the main README: **you are responsible for checking and honoring both parent models' licenses**
before redistributing anything merged here. This tool does not (and cannot) grant you rights the
original license holders didn't.

### 4. Using the download pipeline to move data you don't control

**The risk:** repurposing a "download arbitrary HuggingFace repo id" feature to exfiltrate or
stage files unrelated to model weights.

**Mitigation:** the download path (`internal/download/download.go`) only downloads files that the
HuggingFace Hub API lists as siblings of the given repo, and only `.safetensors` (plus small,
fixed config/index files) - it does not accept file paths, other domains, or arbitrary content
types. Combined with the fixed `huggingface.co`-only URL construction described above, there's no
generic file-fetching primitive being exposed here.

### 5. Resource-exhaustion / cost abuse if exposed publicly

**The risk:** if you deploy this somewhere reachable by untrusted users, nothing stops them from
queuing many large downloads and merges back-to-back, running up bandwidth/storage/compute costs,
or filling disk with downloaded checkpoints.

**Mitigation:** two concrete, code-level things, plus one acknowledged remaining gap:

- `docker-compose.yml` binds both services to `127.0.0.1` **by default** (`BACKEND_BIND`/
  `FRONTEND_BIND` env vars opt into wider exposure, deliberately not the default) - so a fresh
  `docker compose up` is not reachable over the network at all unless you explicitly choose that.
- The backend caps concurrent download and merge jobs (`MAX_CONCURRENT_DOWNLOADS` /
  `MAX_CONCURRENT_MERGES`, default 4 and 2) and returns `429 Too Many Requests` once the cap is
  hit, so a single client can't queue unbounded background work. See
  `internal/api/api.go` (`handleStartDownload`, `handleStartMerge`) and `internal/jobs/jobs.go`
  (`Manager.ActiveCount`).
- **What's still missing:** there is no authentication, no per-client rate limiting, and no disk
  quota. The concurrency cap limits *simultaneous* jobs, not *total* jobs over time from a
  determined client repeatedly polling past a 429, and it doesn't distinguish between clients. If
  you widen exposure beyond `127.0.0.1`, put real auth and a reverse proxy with per-IP rate
  limiting in front of it first - this is still on you, and is exactly why the default stays
  localhost-only.

---

## What's built in to reduce these risks

Summarizing the concrete, checkable things already in the code (not just policy):

| Safeguard | Where |
|---|---|
| Safetensors-only I/O — no pickle, no code execution on load | `internal/store/safetensors.go` |
| Fixed `huggingface.co`-only URL construction for all downloads and compatibility checks | `internal/download/download.go`, `internal/compat/compat.go`, `internal/models/catalog.go` |
| Every merged tensor is a byte-for-byte, disclosed copy of a parent tensor — never silently altered | `internal/merge/merge.go` |
| Provenance (both parent repo ids, swap ratio, seed, timestamp, human-readable disclaimer) embedded directly in the output file's own metadata, not just an external log | `internal/merge/merge.go` (`meta` map passed to `store.WriteSafetensors`) |
| Full per-tensor decision log (`report.json`), matching what the UI's tensor ledger shows | `internal/merge/merge.go` (`writeReportJSON`) |
| Live tensor-shape compatibility checks, so the tool can't silently produce a checkpoint whose config and weights disagree | `internal/compat/compat.go` |
| Ports bound to `127.0.0.1` by default — not reachable over the network unless explicitly opted into | `docker-compose.yml` |
| Concurrent download/merge jobs capped, returning `429` past the limit | `internal/jobs/jobs.go` (`ActiveCount`), `internal/api/api.go` |
| Prominent, repeated "this is not a production merge algorithm / not evaluated" disclosure in the UI and README | `frontend/src/App.tsx`, `README.md` |

Note what's *not* in this table: there's no code-level enforcement that stops someone who forks
this repo from deleting the disclaimer text, removing the metadata embedding, or repointing the
download logic elsewhere. Anything below the fold of "this is a documentation/config-level
safeguard, not a cryptographic guarantee" is exactly that — see the honest framing at the top of
this document.

---

## What you're responsible for if you deploy or extend this

This project ships as an educational, single-user-oriented demo. If you deploy it more broadly or
build on top of it, you take on responsibility for things it does not currently handle for you:

- **Authentication and per-client rate limiting.** There is none beyond the global concurrency
  caps described above. Anyone who can reach the backend's HTTP API can trigger downloads and
  merges. Put it behind real auth and a rate-limiting reverse proxy before exposing it beyond
  `127.0.0.1`.
- **Storage and cost limits.** Nothing caps how many *distinct* models get downloaded over time or
  how large the `model_data` volume grows. Set disk quotas and/or a cleanup policy if running this
  unattended.
- **Content/behavioral evaluation of anything merged here before real use.** This tool tells you
  *what went into* a merge and *exactly which bytes came from where* - it does not, and cannot,
  tell you whether the resulting model is *safe to deploy*. That requires running the actual model
  through your own evaluation, red-teaming, or safety review process, same as you would for any
  third-party model.
- **License compliance.** Verify both parent models' licenses permit merging and whatever
  redistribution you intend, before you distribute anything produced here.
- **Not stripping or overriding the embedded provenance metadata.** If you build tooling on top of
  Splice's output, preserve the `__metadata__` block and `report.json`. Removing them doesn't make
  a merged model safer - it just makes it look like something it isn't.
- **Not treating this document as a substitute for the code-level safeguards it describes.** If
  you fork this project and remove the safeguards in the table above, this document no longer
  accurately describes what your fork does - update it, or don't claim the disclosures it
  describes still apply.

---

## If you're evaluating a `merged.safetensors` file someone gave you

You don't need this repo to check a Splice-produced file's claims - the safetensors format is
self-describing. With the `safetensors` Python package:

```python
from safetensors import safe_open

with safe_open("merged.safetensors", framework="pt") as f:
    print(f.metadata())
    # {'merged_by': 'hf-mergekit-demo (Splice)', 'model_a': '...', 'model_b': '...',
    #  'swap_ratio': '0.3000', 'seed': '123456', 'created_at': '...',
    #  'disclaimer': 'Randomly spliced from two HuggingFace checkpoints ...'}
```

If a file claims to be an untouched, original checkpoint but carries no such metadata and you have
reason to believe it was produced by a merge tool, or if the metadata is present but looks edited,
treat it with the same skepticism you'd apply to any unverified model from an untrusted source:
don't deploy it without independent behavioral evaluation, and don't take its stated provenance on
faith if you can't corroborate it.

---

## Reporting a concern

If you find a way this tool (or its output) could be used to cause harm beyond what's documented
here, please open an issue describing the scenario rather than the exact exploit mechanics
publicly, so it can be reviewed and this document updated.
