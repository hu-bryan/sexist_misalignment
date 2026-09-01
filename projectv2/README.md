# projectv2 — Is the gender-misalignment direction a composition?

A ground-up redesign of the v1 pipeline, targeting the open question of
[Soligo et al. 2025 (arXiv:2506.11618)](https://arxiv.org/abs/2506.11618), §3.3:

> "these semantic misalignment directions may be combinations of the general
> misalignment direction and semantic directions for concepts such as gender.
> Future work could investigate the extent to which either of these framings
> is true, and whether these vectors interact in a manner that is consistent
> with the linear representation hypothesis."

## Design principles (what v2 fixes over v1)

1. **Test the paper's actual object.** `v_gender` is built with the paper's
   recipe: mean-diff between misaligned and aligned responses *restricted to
   gender-topical text* (topic judge > 50), not a sexism-valence contrast.
   The paper's cos(v_gender, v_general) > 0.95 is reproduced first as a
   replication anchor before anything is extended.
2. **Match the authors' estimator exactly** (verified against their released
   code in `external/model-organisms-for-EM`): token-pooled global means over
   answer tokens, coherent > 50, aligned > 70 vs <= 30, larger class
   downsampled, all 48 layers, chat-templated (question, answer) pairs.
3. **Every geometric number ships with a null, a ceiling, and a CI.**
   Permutation null (label shuffles), split-half noise ceiling (the max R²
   any model could reach given estimation noise), bootstrap CIs, and
   cross-validated R². Pre-registered rule: *compositional* iff
   cv R² ≥ 0.8 × ceiling; *underpowered* if ceiling < 0.3.
4. **The primary test is causal, not geometric.** Steering the *aligned* chat
   model (paper protocol: raw mean-diff, one layer, all token positions) with
   `composed` (projection of v_gender onto span{m_decon, g}) vs `residual`
   (the rest; composed + residual = v_gender exactly), plus positive controls
   (`v_general`, `g_topic`) and a norm-matched `random` baseline. Ablation
   experiments on the EM model complete the picture.
5. **In-distribution gender direction.** `g_topic` contrasts gender-topical
   vs non-gender text *within aligned responses of the same model*, using
   paired gender/neutral probe questions to break the question-identity
   confound — not WinoBias/Bias-in-Bios activations from a different
   distribution.
6. **Judge fixed.** gpt-oss-20b (different family from the Qwen subject; ~42
   GiB dequantized to bf16 on an A100) with the authors' GPT-4o prompts
   verbatim. The harmony `final` channel is force-prefilled so the judge
   cannot emit reasoning and must answer immediately — the closest open-model
   analogue of the authors' single-token logprob protocol (their own code
   notes that trick only works for GPT-4o's tokenizer). Unparseable outputs
   get a sampling retry, then a digit-constrained decoding rescue. Anchor
   validation runs before any scoring.
   *Alternative:* vLLM structured outputs (`guided_regex`) also support
   gpt-oss, but reasoning-boundary handling had bugs until recent versions —
   the prefill approach avoids that dependency entirely.

## Run it

One entry point produces all data, statistics, plots and the final report:

```bash
pip install -r projectv2/requirements.txt
python -m projectv2.run_all --config projectv2/configs/standard.yaml
```

On Colab: open `projectv2/colab_run.ipynb` and run all cells (A100 GPU
runtime required). Use `configs/debug.yaml` for a ~30-minute smoke test.

The pipeline is fully resumable: every stage writes artifacts to
`outputs_v2/<run_name>/` and reruns skip completed stages automatically
(`--run-name` to target an existing run, `--start-stage N` to force a redo).
Stage 2 and 7 additionally checkpoint after every judge metric, so a Colab
disconnect never loses more than one metric of judging.

The standard config is budgeted for **≤ 4 hours total** on one A100. The
statistical resampling (permutation null, bootstrap, split-half) is matrix
algebra over *stored* activations and costs minutes — the runtime lives in
generation and judging, which is where the budget cuts were made:
2-point scale sweep instead of 5, conditional judging (`coherent` first,
other metrics only on coherent text), no activations for responses that
enter no direction (the aligned 30–70 mid-band, non-aligned aux), and
sampling reallocated toward the two gender-heavy questions that feed the
binding cell (misaligned ∧ gender-topical).

| Stage | What happens | Model resident | Approx. time |
|---|---|---|---|
| 1 | Generate EM responses (6 questions × 150 + 2 gender-heavy × 300 + 12 probes × 60) | Qwen-14B + LoRA | ~40-50 min |
| 2 | Judge coherent on all, aligned/gender on coherent only, after anchor validation | gpt-oss-20b | ~30-40 min |
| 3 | Answer-token activation sums, 48 layers, direction-eligible responses only | Qwen-14B + LoRA | ~20-30 min |
| 4 | Build v_general, v_gender, v_general_decon, g_topic | — | seconds |
| 5 | Regression + permutation null (300) + ceiling (60 splits) + bootstrap (300) | — (GPU math) | ~2 min |
| 6 | Steering sweep (6 conditions × 2 scales × 96) + 3 ablations | Qwen-14B (+LoRA) | ~40-50 min |
| 7 | Judge steered outputs (conditional), aggregate tables | gpt-oss-20b | ~25-35 min |
| 8 | summary.md + all plots | — | seconds |

The one cut that costs real statistical power is generation volume for
v_gender's smallest cell. The split-half ceiling exists precisely to catch
this: if stage 5 reports UNDERPOWERED (ceiling < 0.3), raise
`n_samples_gender_q` and rerun from stage 1 in a fresh run.

## Reading the results

`outputs_v2/<run>/summary.md` contains the full decision matrix. The short version:

- **Replication anchor fails** (cos(v_gender, v_general) ≪ 0.95): the pipeline
  diverges from the paper's — debug before interpreting anything else.
- **Ceiling < 0.3**: v_gender is mostly estimation noise at this sample size;
  the honest conclusion is "underpowered", not "structurally independent".
- **cv R² ≈ ceiling AND `composed` reproduces `v_gender`'s steering profile
  at matched coherence AND `residual` behaves like `random`**: the paper's
  composition framing is supported.
- **cv R² ≪ a reliable ceiling AND `residual` still induces gendered
  misalignment**: v_gender contains a structurally distinct component.
