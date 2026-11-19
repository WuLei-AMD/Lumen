# Fixed-Agent GEAK Benchmark

## Experimental control

The primary comparison uses one fixed engineer `ToolAgentLoop`. Base and SFT
runs must have identical:

- task suite and initial source hashes;
- system/role prompts and tools;
- max turns, token/time budgets and decoding parameters;
- GEAK, Harness, container and compiler versions;
- correctness and performance repetition policy;
- physical evaluation GPU and clock/contention policy.

Only the served model endpoint, model name and checkpoint identity may differ.
The full MultiTune role hierarchy can be reported separately, but it is not
the primary model-isolation result.

## Isolation

Model serving and kernel timing must not share a GPU. GEAK evaluation uses
`gpu_lock.sh`, `GEAK_GPU_ALLOWED`, idle checks and an isolated workspace.
Held-out target patches, hidden references and protected case metadata are
forbidden from model-visible messages and workspaces.

## Per-turn record

Each evaluated turn records:

- campaign/task/model/seed and immutable config hashes;
- cumulative assistant turn and token counts;
- model and tool wall time;
- patch apply, compile and correctness state;
- independently verified `speedup_vs_frozen_baseline`;
- compile/runtime error category;
- trajectory and verification receipt references.

Assistant turns that only inspect or edit the workspace are retained with
`evaluated=false`; they count toward token/time/turn cost but not compile or
correctness error denominators.

GEAK's frozen source baseline is not `torch.compile`; reports must not rename
this metric.

## Aggregates

The report includes:

- Pass@k over seeds and turn budgets;
- Fast@1.0/1.2/1.5/2.0;
- first-pass rate, turns to pass and turns to best;
- tokens/time/tool calls per successful task;
- compile and correctness error rates;
- monotonic improvement and regression rates;
- normalized Speedup-AUC;
- per-operator and per-lane breakdowns;
- paired bootstrap confidence intervals for SFT minus Base.

General Coding Dev-80 is a separate retention gate. It does not contribute to
the kernel optimization score.
