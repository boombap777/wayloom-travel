# Travel Itinerary Demand Parsing and Planning Assistant — DEV SPEC

## 1. Overview

### 1.1 Goal

Build a local-first proof of concept that turns a Chinese natural-language travel request into a
validated trip requirement, decides whether information is sufficient, calls a bounded trip-option
tool, and returns a structured draft itinerary. The project is intentionally independent from the
existing appointment Agent and RAG projects.

### 1.2 User flow

1. A user describes a trip in one or more conversational turns.
2. The extractor normalizes stated constraints such as origin, destination, date, duration, party
   size, budget, and travel preferences.
3. Missing required information produces a `clarify` response. Sufficient information produces a
   `search_trip_options` tool call.
4. The orchestrator executes the tool against a versioned, local demonstration catalog and returns
   a draft itinerary with explicit fixture provenance.

### 1.3 Evidence boundary

The MVP uses deterministic parsing and a static local catalog so it can be tested offline. It is
not a live booking product and must not claim real-time transport, hotel, price, or availability
data. Training examples are deterministic, scenario-controlled synthetic data with a documented
quality audit and a pending human-review queue; they must never be described as real user data or
as manually reviewed until a reviewer records that work. QLoRA configuration and SFT/DPO data
artifacts are included as a reproducible training contract, but no trained-model quality or
latency metric may be claimed until a training and inference run is recorded.

### 1.4 Non-goals

- Do not collect personal or payment data.
- Do not purchase tickets, reserve accommodation, or make external writes.
- Do not use the existing massage-appointment data, schemas, tools, or metrics.

## 2. Features

### 2.1 Structured travel contract

The request contract contains `departure_city`, `destination`, `start_date`, `days`,
`traveler_count`, optional `budget_cny`, optional `themes`, and optional `hotel_preference`.
Outputs use exactly one of `clarify`, `tool_call`, or `final_plan`.

Required fields before a tool call are departure city, destination, start date, day count, and
traveler count. The assistant must never invent an unknown required field.

### 2.2 Decision and planning loop

- `clarify`: list missing required fields and ask one concise question.
- `tool_call`: emit `search_trip_options` with validated arguments.
- `final_plan`: execute the local catalog tool and return a transparent draft with options,
  a short day-by-day outline, and `source=demo_catalog_v1`.

### 2.3 Dataset and model contract

Provide a versioned seed set for three behavior classes: clarification, tool calling, and final
plan rendering. A dataset builder emits SFT records and DPO preference pairs. Provide a Qwen3
QLoRA configuration and system prompt, but keep actual training as a later, separately measured
task.

### 2.4 Dataset lineage and quality audit

Generate versioned train, validation, and frozen evaluation splits from a declarative scenario
configuration. The generator must use split-specific template families and reject normalised input
duplicates across splits. Produce a manifest containing counts, scenario coverage, SHA-256 hashes,
and generator seed. A human-review queue and review protocol are required, but only a completed
review record may be called manual review.

### 2.5 Actual local post-training experiment

Run one reproducible Qwen3-1.7B 4-bit NF4 QLoRA **SFT** experiment against the versioned SFT
training and validation splits. Capture the expanded training configuration, model and dataset
fingerprints, seed, package versions, GPU, trainable parameter count, loss, elapsed time, and
adapter hash. Compare the base model and selected SFT adapter on the frozen evaluation split using
the same deterministic decoding settings. DPO pairs may be prepared and retained for a later
experiment, but must not be claimed as an executed training stage unless it has an independent
run manifest and report.

### 2.6 Offline evaluation

Evaluate JSON validity, action exact match, required-slot micro-F1, and tool-argument exact match
on a frozen local evaluation set. Produce a JSON report with case-level errors.

### 2.7 Runnable demo

Expose a CLI that accepts `--message` and `--today`, runs the full loop, and prints valid JSON.
The default inference implementation is named `RuleBasedTravelExtractor` to avoid implying that a
fine-tuned model is being used.

## 3. Tech Stack

- Python 3.12+ and the standard library for the executable MVP and data generator.
- Pytest for unit and integration tests.
- JSON/JSONL for versioned fixtures, training seed data, and evaluation reports.
- Declarative JSON configuration for data scale, split seed, scenario mix, and supported cities.
- Qwen3-1.7B, Transformers, PEFT, bitsandbytes, and Accelerate for the actual local QLoRA SFT
  experiment; their exact versions are captured in the run manifest.
- GGUF and llama.cpp for a separately measured local inference deployment stage.

## 4. Testing

### 4.1 Unit tests

- Validate each output action and request invariant.
- Test Chinese constraint parsing and refusal to invent missing values.
- Test tool argument creation, catalog filtering, and draft-plan rendering.
- Test metric calculations for correct and incorrect predictions.
- Test split isolation, schema validity, coverage, manifest hashes, and legacy-domain exclusion.
- Unit-test model-output JSON extraction and metric scoring with fake model responses; do not mock
  the actual training run when reporting real experiment evidence.

### 4.2 Integration tests

- Run one complete request through clarification and one through tool call plus final plan.
- Run the evaluation CLI against the frozen evaluation set and verify report fields.

### 4.3 Acceptance criteria

- `pytest -q` passes without a network connection or local LLM server.
- Every user-visible result is valid JSON and includes an evidence source where a catalog result is
  used.
- Seed data and the evaluation set contain no appointment, technician, massage, or calendar-write
  semantics.
- A rerun with the same configuration must reproduce identical split hashes and coverage.
- A reported model result must be backed by a persisted training manifest and a frozen-eval report
  with raw model outputs or case-level error records.

## 5. Architecture

```text
src/travel_itinerary/
  contracts.py     # Strong request, decision, catalog, and report contracts
  extractor.py     # Deterministic baseline; future local-model adapter seam
  catalog.py       # Versioned local catalog and bounded search tool
  orchestrator.py  # Clarify -> tool call -> final-plan loop
  evaluation.py    # Offline metrics and report generation
  dataset.py       # Seed-data validation and SFT/DPO rendering
  data_generation.py # Config-driven scenario generation, split audit, and manifest output
  training.py      # Config-validated local QLoRA SFT and run manifest
  model_evaluation.py # Base/adapter generation, strict JSON parsing, and frozen-set report
  model_output.py # One strict parser and the only bounded mechanical repair
  runtime.py       # rule | hf_adapter | llama_cpp extractor profiles
  application_smoke.py # Persisted model -> tool -> catalog -> plan smoke
  llama_quality.py # Full 120-case GGUF raw/repaired evaluation
  llama_performance.py # Separate cold-process and warm-server benchmarks
  evidence.py      # Source/artifact manifest and aggregate resume release gate
  prompting.py     # Shared model prompt construction
  cli.py           # Demo, evaluation, and dataset commands
data/
  raw/v1.0/        # Generated train and validation conversations
  eval/v1.0/       # Generated, frozen held-out cases
  manifests/        # SHA-256 lineage and scenario coverage
  review/           # Pending human-review samples; status is explicit
configs/training/  # SFT experiment profile, DPO contract, and system prompt
tests/             # Unit and integration coverage
```

The extractor owns only language-to-contract transformation. The catalog owns travel facts.
The orchestrator performs the bounded tool call and is the only component allowed to produce a
draft plan from tool output. All data is local and versioned.

## 6. Schedule

- [x] T1 — Build an offline, end-to-end travel requirement parsing and planning MVP: contracts,
  deterministic baseline, local catalog tool, CLI, seed data, evaluation, tests, and docs.
- [x] T2 — Generate and programmatically audit a versioned synthetic train/validation/frozen-eval
  dataset; record split isolation, SHA-256 lineage, scenario coverage, and a pending human-review
  queue. Do not claim manual review until it is actually recorded.
- [x] T3 — Run a Qwen3-1.7B 4-bit QLoRA SFT experiment, compare base and adapter on the frozen
  split, and record raw outputs, metrics, loss, lineage, package versions, GPU and adapter hash.
  DPO data is prepared but remains unexecuted unless a separate manifest is produced.
- [x] T4 — Export a selected model to GGUF, integrate llama.cpp, and benchmark local inference.

## 7. Future

- Replace the fixture catalog with authorized, timestamped provider data while preserving the same
  tool contract and explicit source attribution.
- Complete the local-model application profile defined in section 8 before adding a web interface.
- Add a lightweight web interface only after the CLI and evaluation loop are stable.

## 8. Local-model application integration

This section began as an additive vNext plan. T5-T9 are now implemented and retain their own
persisted evidence; T1-T4 reports remain immutable historical measurements. HF-adapter, GGUF
quality, application smoke, cold-process and warm-server metrics remain separate measurement
profiles and must never be merged into one result.

### 8.1 Responsibility boundary

The model owns only Chinese language-to-contract transformation: extract the eight travel fields
and select `clarify` or `tool_call`. It does not own catalog facts, itinerary rendering, booking,
payment, or external writes. The orchestrator validates the decision, invokes the bounded local
catalog tool, and is the only component allowed to construct `final_plan` with
`source=demo_catalog_v1`.

### 8.2 Extractor profiles and end-to-end flow

All inference profiles implement the same `TravelExtractor` contract:

- `EXTRACTOR_MODE=rule` retains `RuleBasedTravelExtractor` as the deterministic offline baseline;
- `EXTRACTOR_MODE=hf_adapter` loads the measured Transformers/PEFT adapter;
- `EXTRACTOR_MODE=llama_cpp` loads the measured base GGUF plus LoRA GGUF through a recorded local
  llama.cpp runtime.

The minimum local-model application path is:

```text
user message
  -> local model extractor
  -> strict JSON extraction and TravelDecision validation
  -> clarify
     or
  -> search_trip_options
  -> demo_catalog_v1
  -> final_plan with explicit fixture provenance
```

The CLI must select the extractor profile explicitly. Tests and documentation must never imply
that the rule profile is a fine-tuned-model result or that the demonstration catalog contains
live prices, inventory, or bookable options.

### 8.3 Output repair and evaluation

Raw and repaired model metrics are always reported separately. The only permitted mechanical
repair is rebuilding a missing `tool_call.arguments` object from an already complete,
schema-valid `request` when the action and tool name are already valid. A repair must not infer or
change any city, date, duration, traveler count, budget, theme, hotel preference, action, or tool
name, and every application must be recorded in `repair_notes`.

The current three-case llama.cpp run remains a deployment smoke only: raw JSON is `3/3`, raw
contract validity is `2/3`, and contract/action validity after the documented repair is `3/3`.
It does not inherit the 120-case HF-adapter metrics. Before a GGUF quality claim is made, the
`llama_cpp` profile must run all 120 frozen cases and report JSON validity, contract validity,
action exact match, slot micro-F1, tool-argument exact match, repair count, and case-level errors
against the same evaluation SHA-256.

The completed full GGUF run is recorded in
`reports/deployment/llama_cpp_full_frozen_eval_v1.json`: all 120 processes exited successfully;
raw contract/action/slot-F1/tool-EM are `68.33% / 60.83% / 79.21% / 50.00%`; after 18 permitted
mechanical argument repairs they are `83.33% / 75.83% / 90.16% / 76.47%`. These results are
diagnostic evidence of quantized runtime behavior, not a replacement for the HF-adapter metrics.

Cold and warm performance are measured separately. Cold-start evidence uses at least 20 independent
one-shot processes and includes process startup, model loading, and generation. Warm evidence uses
at least 100 requests after a declared warm-up against one resident local runtime and reports its
timing boundary, P50, P95, and generation throughput. The existing three-case `4.66 s` P95 remains
labelled as one-shot smoke evidence, not resident-service latency or a production SLA.

The completed performance report is
`reports/deployment/llama_cpp_performance_v1.json`: 20/20 cold one-shot processes have P50/P95
`3.54/4.25 s`; after three warm-up requests, 100/100 sequential resident-server requests have
P50/P95 `0.98/1.71 s` and generation P50 `109.5 token/s`. Contract-valid-after-repair rates
(`65%` cold, `84%` warm) are reported independently from speed. Neither profile is a concurrent
load test or production SLA.

### 8.4 Acceptance criteria

- the CLI can select `rule`, `hf_adapter`, and `llama_cpp` without changing the orchestrator;
- one missing-field request reaches `clarify`, and one complete request reaches
  `tool_call -> demo_catalog_v1 -> final_plan` through the local-model profile;
- invalid or incomplete model output fails closed and never invokes the catalog tool;
- the local-model path cannot emit a booking, payment, live-price, or external-write result;
- GGUF quality and performance claims cite their own model/runtime/data hashes and do not reuse
  unquantized HF-adapter or three-case smoke metrics outside their measurement scope.

### 8.5 Schedule

- [x] T5 - `TravelExtractor` and explicit `rule|hf_adapter|llama_cpp` runtime profiles are shared
  by the same orchestrator and selectable from the installed CLI.
- [x] T6 - HF-adapter and llama.cpp extractors are implemented lazily; a persisted 2/2 actual
  application smoke proves `clarify` and `tool_call -> demo_catalog_v1 -> final_plan` without
  moving catalog or final-plan authority into the model.
- [x] T7 - The shared parser rejects unknown keys, invalid types, missing-field drift, unsupported
  actions/tools and tool-argument drift. Only absent arguments may be copied from an already
  complete request; invalid model output raises before any catalog invocation.
- [x] T8 - All 120 frozen cases were executed through the GGUF + LoRA llama-cli profile with
  separate raw/repaired metrics, 120 case records, 18 repair records and model/data/runtime hashes.
- [x] T9 - 20 independent cold processes and 100 post-warm-up resident HTTP requests were measured
  separately, with timing boundaries, success counts, P50/P95, throughput and contract validity.

Final claim readiness is controlled by `python -m travel_itinerary.evidence verify --require-ready`.
It binds current source, configuration, datasets, adapters, GGUF files and every report by SHA-256;
checks the final JUnit contract tests; and emits six `SATISFIED|PARTIAL` claims. A checked schedule
item alone is not completion evidence.


## Publication preparation (2026-09-08)

- [x] [PUB1] Prepare the portable campus-project source release; no Git commit or upload.
  - Acceptance: an independent exported source tree installs locked dependencies and passes
    its default offline suite and documented minimal demo without the original checkout.
  - Acceptance: exclude credentials, runtime databases, caches, raw logs and model weights;
    retain reproducible inputs, licenses, notices and bounded publication scan evidence.
  - Acceptance: document quick start, optional model/data preparation and honest measured
    limitations; local evidence is not a production or hosted-CI claim.
  - Authorization: the owner confirmed these are publishable personal prototypes and chose MIT.
    Third-party software, datasets and models retain their original licenses.
  - Scope expanded by owner approval: implement and verify docs/ui-design.md using a
    minimal loopback Web/API shell, reusing the existing orchestrator and extractor profiles.
    Include real HTTP and browser checks, no external bookings, no paid APIs and no retraining.
  - Publication correction: official fixed-revision shard hashes identify Qwen/Qwen3-1.7B,
    not the Qwen3-1.7B-Base ID mistakenly entered into the historical training config.
    Preserve original experiment reports and document this correction with the model origin receipt.
