# unloop: the halting direction

Qwen3.5-0.8B in thinking mode often loops until it hits the token limit.
A frontier LLM fixed most of this by editing 131,072 weights ([blog post, Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/)).
This project asks what that edit does, and whether simple, standard methods get the same gain without a frontier LLM.

Status: the code, tests, and configs are complete.
The full study has not been run yet; it is set up for a CUDA cluster (see [Run](#run)).
The only numbers below come from exploratory checks on this laptop, and each one says so.

## Questions

1. **Reproduce.** Does the released edit reduce looping and raise accuracy under the blog's settings? Which tensors changed, how many values, and what is the edit direction?
2. **Explain.** What does the direction do? Logit lens; does it suppress the end-of-thinking token `</think>`? How does the projection on it change over time in looping and non-looping traces?
3. **Simpler methods.** How much of the gain comes from (a) a difference-of-means direction with directional ablation, (b) activation steering, (c) decoding-time fixes (a `</think>` logit bias, a thinking budget, a stronger presence penalty)?
4. **Origin.** Is the direction in Qwen3.5-0.8B-Base and in the 2B models? Does it relate to how much a model loops?

## Background

- The released checkpoint: [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming). Logs and the edit's provenance: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), `exp8_math/`.
- The logs give the edit's form: `W = W0 - 6 * outer(u, mask64 * (u @ W0))` on `mlp.down_proj` of layers 15 and 19, where `u` is a unit "repetition direction" in the residual stream and `mask64` selects 64 neurons.
- Blog results (thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, 32,768 tokens): MATH-500 7.9% to 48.1%, GSM8K 13.8% to 67.1%, MATH-500 answers cut at the limit 93.0% to 33.1%. The only baseline is the unedited model.
- Prior work: difference-of-means directions and directional ablation (Arditi et al. 2024, "Refusal in language models is mediated by a single direction"); loop prediction from hidden states ("Circular Reasoning", arXiv 2601.05693); "Token Budget Saturation" (arXiv 2607.21433).

## Exploratory checks done so far

These ran on the laptop during development, with this repository's functions in scratch scripts.
`unloop reproduce` recomputes all of them and writes `results/reproduce.json`.

**Weight diff (question 1).**
Only two of 488 tensors differ: `model.language_model.layers.{15,19}.mlp.down_proj.weight` (1,024 x 3,584 each).
In each, exactly 64 columns changed; 65,260 and 64,928 bfloat16 values differ (some touched values round back to the same bfloat16).
Each delta is rank 1: the second singular value is 5.6e-4 (layer 15) and 6.4e-4 (layer 19) of the first, which is bfloat16 rounding.
So the top left singular vector of each delta is the edit direction `u`.

**Loops in the blog's own full-length traces (question 1).**
The blog released the first sample per problem for 200 MATH-500 and 200 GSM8K problems, for both models, at the full 32,768-token protocol.
The loop detector below gives:

| Model | Benchmark | Correct | Cut at 32,768 | Looping | Median loop onset (tokens) |
|---|---|---|---|---|---|
| unedited | GSM8K | 11.5% | 87.5% | 89.0% | 1,900 |
| unedited | MATH-500 | 7.5% | 93.0% | 91.5% | 2,458 |
| released edit | GSM8K | 61.0% | 9.0% | 10.5% | 8,191 |
| released edit | MATH-500 | 46.0% | 35.0% | 31.0% | 21,016 |

The edit's answers are long.
Only 13% (GSM8K) and 8% (MATH-500) of its traces finish within 4,096 tokens, and 35% and 14.5% within 8,192.
So a short-budget run measures the loop rate well, but not the accuracy.

**An early-stop rule is not safe.**
A rule that stops generation once 98% of the last 1,024 tokens repeat earlier 16-grams would save compute.
On the edited model it fires on 55 MATH-500 traces, and 17 of them later finish on their own.
So generation never stops early in this project.

**Local speed.**
On an M3 Pro (MPS), the hf backend generates about 134 tokens/s at batch 16, and teacher-forced passes run at about 1,100-1,500 tokens/s.
The full protocol for one condition is about 100M generated tokens, so the generation runs on the cluster.

## Method

**Loop detector** (`loops.py`).
A token is repeated when the 16-gram that ends at it occurred earlier in the response.
A response is looping when some suffix of at least 256 tokens is at least 90% repeated.
The loop onset is the start of the dense repetition: the first repeated token of the first 256-token window inside the longest such suffix that is itself at least 90% repeated.
(The first repeated token of the suffix itself is a median of about 3,200 tokens too early, because over a long loop the suffix's 10% slack takes in sparsely repeated text before the loop; `unloop rescore` recomputes the labels of saved generations.)
The detector does not depend on the token limit, so it also works on short runs.

**Grading** (`grading.py`).
The same as the blog: the last `\boxed{}` after the last `</think>` (the whole response if thinking never closed), checked with math-verify 0.9.0.

**Data split.**
Directions are fitted only on the blog's unedited traces (problems from the 400 above).
Those problems are split by a hash of their id into `fit` (directions) and `val` (sweep and explain).
Generation evaluates only the other problems (`split = "heldout"`: 300 MATH-500, 1,119 GSM8K).

**Directions** (`unloop directions`).
For each looping `fit` trace, the base model reads the trace (teacher forcing).
The productive window is the 256 tokens that end 64 tokens before the loop onset; the loop window is the 256 tokens that start 64 tokens after it.
The direction at each layer is the mean residual stream in loop windows minus the mean in productive windows (paired within each trace).
The run also ranks each layer's MLP neurons by (extra activation in loops) x (how much the neuron writes the direction), which is how the 64 columns are chosen for the released-form edit.

**Edits** (`edits.py`, `conditions.py`).
All weight edits have the form `W <- W - alpha * u (u^T W)` on chosen columns of a residual-stream writer.
`alpha = 1` removes the direction (directional ablation); the released edit is `alpha = 6` on 64 columns.
Steering adds `scale` times the loop-minus-productive mean difference to the residual stream after one layer, at every generation step.

**Sweep** (`unloop sweep`).
On `val` traces, each candidate (the released edit rebuilt from `u`, single-layer and multi-layer ablations, the released form with our direction, ablation of all writers, steering at layers 7, 11, 15, 19 with scales -1, -2, -4) is scored by teacher forcing:
the change in mean log-probability of the actual next token in the loop window (should fall) and in the productive window (should stay).
The selected weight edit and steering setting have the largest loop-window drop with at most 0.02 nats/token lost on productive text.
The sweep is a proxy; only generation measures the effect on looping.

**Rank-1 escape edit** (`unloop fit-rank1`, `unloop branch`).
The proxy above barely moves under the released edit, which works by breaking a loop that has already started, so it cannot find that edit. `fit-rank1` trains the edit directly. On the MLP down-projections of layers 15 and 19, `W <- W + u v^T`: `u` is a unit residual direction, initialized to the difference-of-means direction, and `v` is a gain per neuron, initialized near zero. The loss on each looping fit trace is next-token loss on the productive window and unlikelihood of the repeated tokens for the next 1,024 tokens of the loop, plus the same next-token loss on finished non-looping fit traces. The released weights are not used. `branch` then restarts generation from the 32 validation prefixes in `configs/branch_val.json` (the branch-point pilot's set): 8 samples, 4,096 new tokens, stop at `</think>`. A continuation is looping only when thinking never closes and the tail is a loop. `rank1_top64` keeps the 64 largest gains. Run `fit-rank1` before either condition.

**Explain** (`unloop explain`).
Logit lens of each direction through the final norm and the unembedding, with the rank and z-score of `</think>`, `<|im_end|>`, and `<think>`.
Mean log-probability of `</think>` and the share of positions where it is in the top 20 (so top-k sampling can pick it), for the base model and the released edit.
Projection of the residual stream on `u` per token, aligned at the loop onset, against non-looping text (`results/figures/projection.png`).

**Origin** (`unloop origin`).
The same direction fit on the same token sequences in Qwen3.5-0.8B-Base, Qwen3.5-2B, and Qwen3.5-2B-Base: per-layer separation (paired effect size), cosine with the 0.8B direction where the widths match, and the logit lens.
The 2B model's loop rate comes from the `qwen2b` generation condition.

## Results

Not run yet.
`scripts/run_all.sh` writes one JSON summary per condition to `results/<run>/`, and `unloop table` writes `results/<run>/table.json` with accuracy, loop rate, and cut rate per benchmark (Wilson 95% intervals) and paired differences to the baseline on the same backend (bootstrap 95% intervals over problems).

## Run

**Local tests** (no model download):

```bash
uv sync
uv run pytest -q
uv run ruff check .
```

**Cluster** (CUDA, Linux, Slurm).
vLLM 0.33.0 supports `Qwen3_5ForCausalLM` and `Qwen3_5ForConditionalGeneration`, but it pins torch 2.14.0 and transformers < 5.21, so the cluster uses its own environment, `.venv-cluster`.
`scripts/cluster_env.sh` activates it and keeps every cache (Hugging Face, uv, Triton, vLLM, temporary files) in the repository's `.cache`, so nothing goes to `$HOME`.
Compute nodes often have no internet, so `unloop fetch` downloads all pinned models and data on a login node first, and jobs run offline.

```bash
scripts/setup_cluster.sh                       # once, on a login node (needs uv)
. scripts/cluster_env.sh && unloop fetch       # once, on a login node
mkdir -p logs
sbatch --time=02:00:00 scripts/job.sbatch env SMOKE=1 scripts/run_all.sh   # analyses at full size, then every condition at 2 problems and 256 tokens
scripts/submit_all.sh configs/full.toml        # one analysis job, then one single-GPU job per condition
unloop table --run full                        # when the jobs finish

# the rank-1 escape edit (after `unloop directions` has been run)
sbatch --account=jordanat --partition=gpu --constraint=gpu80 --time=02:00:00 --job-name=unloop-rank1 \
  scripts/job.sbatch unloop fit-rank1
sbatch --dependency=afterok:<fit-job> --account=jordanat --partition=gpu --constraint=gpu80 \
  --time=04:00:00 --job-name=unloop-branch-rank1 scripts/job.sbatch unloop branch --condition rank1
sbatch --dependency=afterok:<fit-job> --account=jordanat --partition=gpu --constraint=gpu80 \
  --time=04:00:00 --job-name=unloop-branch-top64 scripts/job.sbatch unloop branch --condition rank1_top64
```

`scripts/submit_all.sh` makes the conditions that use an edit or a hook wait for the analysis job.
Set `SBATCH_ARGS` for cluster options; on Princeton's Della, for example: `SBATCH_ARGS="--partition=gpu --constraint=gpu80" scripts/submit_all.sh`.
Without Slurm, `scripts/run_all.sh configs/full.toml` runs everything in sequence on one GPU.

Cost guide: the blog's re-run generated 43.4M tokens in 51 minutes on one H200 with vLLM.
One condition of `full.toml` is roughly 50-100M tokens, so expect 1-2 hours per vLLM condition on an H200 and more on an A100.
The four hf conditions (`base_hf`, `steer_best`, `think_bias_4`, `think_bias_8`) use the Hugging Face decode loop, which is much slower; check the smoke run's `seconds` field first.

**Local pilot** (Apple Silicon): `configs/pilot.toml` has the same conditions with 40 problems per benchmark, 1 sample, and 4,096 tokens, all on the hf backend (up to about 40 minutes per condition at 134 tokens/s).

**Optional hosted-model contrast.**
`unloop api-loops --base-url URL --model ID --key-file PATH` measures the loop rate and accuracy of a model behind an OpenAI-compatible API with the same sampling (top-k and thinking are sent as vLLM/SGLang extensions).
The key is read from the file and sent only in the request header.

## Configs

| | `full.toml` | `pilot.toml` |
|---|---|---|
| problems | all held-out (300 MATH-500, 1,119 GSM8K) | 40 + 40 held-out, seed 0 |
| samples per problem | 4 MATH-500, 2 GSM8K | 1 |
| max new tokens | 32,768 | 4,096 |
| backend | vLLM, hf for hooks and logit bias | hf |
| thinking budget, `</think>` bias start | 8,192 and 16,384; bias after 8,192 | 1,024 and 2,048; bias after 1,024 |

Sampling in both: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (2.0 in `presence_2`).
Pinned revisions are in `src/unloop/config.py`.

## Limits and risks

- Nothing in the study has run end to end on real weights; the tests use a tiny random Qwen3.5 model. The vLLM path (`generate_vllm`, the saved edited checkpoints) has not run at all, because vLLM does not run on macOS.
- The vLLM thinking budget runs as a second pass, so the presence penalty does not see the first pass's tokens after the forced `</think>`. The hf backend does it in one pass.
- The hf and vLLM backends sample differently, so each hf condition is compared with `base_hf`, not `base`.
- Directions come from the base model's own looping text read back by teacher forcing; the two windows also differ in position by about 320 tokens.
- Only looping traces whose windows fit in 8,192 tokens are used to fit directions, which favours early loops.
- The sweep selects on a teacher-forced proxy, not on generation.
- The `</think>` bias values (4 and 8) and the budgets are fixed in advance, not tuned.
- Qwen3.5-0.8B-Base has no chat template, so its loop rate is not measured; origin compares its activations only.
- `full.toml` evaluates held-out problems, so its numbers are not directly comparable to the blog's full-benchmark numbers; set `split = "all"` for that.

## Layout

```
src/unloop/
  config.py      pinned models, datasets, blog logs
  data.py        benchmarks, subsets, chat prompt, blog traces
  grading.py     answer extraction and math-verify check
  loops.py       loop detector
  edits.py       direction math and weight edits
  model.py       model loading, residual-stream hooks, weight replacement
  generate.py    batched sampling (hf), vLLM backend, decoding fixes
  conditions.py  condition specs to edits and hooks
  analysis.py    reproduce, directions, sweep, explain, origin
  rank1.py       rank-1 escape edit and branch-point restarts
  api.py         hosted-model loop contrast
  cli.py         eval, table, command line
configs/         full.toml (cluster), pilot.toml (laptop)
scripts/         setup_cluster.sh, cluster_env.sh, Slurm jobs (job.sbatch, submit_all.sh), run_all.sh
tests/           tests that need no downloaded weights
```
