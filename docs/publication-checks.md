# Publication checks — 2026-09-08

Scope: local campus-project source release, not production readiness or hosted CI completion.

## Verified

69 passed, zero skipped; 68.94% whole-project coverage in a fresh environment. HTTP tests cover input bounds, cross-origin/Host rejection, stateless clarification, model errors and no silent fallback.

Desktop browser: rule-profile clarification to a source-labelled draft. Separate packaged GGUF/base/LoRA installed into the independent source copy; the actual llama_cpp Web/API produced a Beijing draft. Original and packaged model smoke are not quality percentiles.

Each verification copy was exported without the original virtual environment or private runtime
files, and installed its own locked dependencies. Publication inventories contain file hashes,
size checks and a bounded common-token scan. The original repositories were not committed or
uploaded; an isolated Git fixture was used only to exercise Teams HEAD-bound EDD.

## Boundaries

Only a five-city synthetic catalog; no bookings/payments/live inventory. HF 120-case quality and GGUF quality remain separate historical experiments. The foundation checkpoint name correction is documented; original reports are preserved.

Hosted GitHub Actions have not run because no upload/push was authorized. Desktop browser
rendering and interactions were inspected. Responsive CSS is implemented, but a physical mobile
or effective fixed-width device run was not completed: the browser viewport override continued
to report 1265 px. This is not claimed as mobile-device acceptance.

No secret scanner can establish that all arbitrary text or historical Git objects are safe.
The recommended source release excludes Git history, credentials, caches and runtime databases.
Use the generated source directory for a new repository; do not publish the entire development
workspace or the parent directory containing model assets and verification environments.
