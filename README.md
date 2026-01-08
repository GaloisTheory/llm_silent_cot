# Unfaithful Chain-of-Thought: Can Models Reason with Injected CoT?

This repository contains code for investigating whether LLMs can reason correctly with **unfaithful Chain-of-Thought (CoT)**. We intercept the model's thinking process and inject various content to test if models blindly follow stated conclusions or evaluate reasoning quality.

## Key Findings

**Models resist unfaithful CoT.** When we inject assertions like "The answer is A!" into a model's `<think>...</think>` block:

- Models don't blindly follow injected assertions
- Repeating an assertion (or adding trailing blanks) makes compliance *less* likely
- This suggests models evaluate reasoning quality rather than pattern-matching on conclusions

**The mechanism is interpretable.** Using activation analysis, we found:
- Specific attention heads (L27-30) detect and suppress repeated/emphatic assertions
- The model encodes whether content is "primary reasoning" vs "secondary/emphatic" — multilingually
- Framing wrong answers with reasoning language ("Based on the context...") bypasses this circuit

## Repository Structure

```
llm_silent_cot/
├── README.md
├── pyproject.toml
├── .gitignore
└── src/
    ├── bbq/                        # BBQ experiment framework
    │   ├── data/
    │   │   └── bbq_dataset.py      # Dataset loading (BBQItem, load_bbq_items)
    │   ├── batch/
    │   │   ├── run_question_batch.py   # Batch experiment runner
    │   │   ├── batch_utils.py          # Config, checkpointing utilities
    │   │   ├── analyze_batch.py        # Interactive analysis (#%% cells)
    │   │   ├── analyze_batch_utils.py  # Analysis utilities
    │   │   └── configs/                # 35 YAML experiment configs
    │   ├── interactive/
    │   │   ├── playground.py       # Hot-reload experiment loop
    │   │   ├── config.py           # Editable config (hot-reloaded)
    │   │   └── blank_vs_accuracy.py
    │   ├── results_graphs/         # Graph generation scripts
    │   └── constants.py            # Few-shot examples
    │
    ├── internals/                  # Mechanistic interpretability
    │   ├── quick.py                # Main analysis (run with #%% cells)
    │   └── logit_lens.py           # Layer-wise probability analysis
    │
    └── shared/                     # Shared utilities
        ├── config.py               # Prompt templates, model configs
        └── generation.py           # generate_with_custom_override()
```

## Quick Start

### Setup

```bash
# Clone and install
git clone <repo-url>
cd llm_silent_cot
pip install -e .

# Or with uv
uv pip install -e .
```

### Mechanistic Analysis

The primary analysis script for understanding *how* models resist unfaithful CoT:

```bash
cd src/internals
# Open quick.py in VS Code/Cursor and run cells with Shift+Enter
```

Key experiments in `quick.py`:
- **Blank Spaces Sweep**: How P(A/B/C) changes with N spaces in thinking
- **Position/Repetition Test**: Disambiguate positional decay vs repetition detection  
- **Layer-wise Analysis**: Where does N=1 diverge from N=2?
- **Activation Patching**: Which layer restores P(A)?
- **Attention Analysis**: Which heads detect repetition?
- **Direction Analysis**: What does "repeated" encode in vocab space?

### Batch Experiments

Run large-scale experiments across BBQ categories:

```bash
cd src/bbq/batch

# Run single experiment
python run_question_batch.py configs/baseline_age.yaml

# Run with different model
python run_question_batch.py configs/baseline_age.yaml \
    --model "Qwen/Qwen3-1.7B" \
    --model-prefix "qwen_1.7B"

# Dry run (preview without running)
python run_question_batch.py configs/force_immediate_answer.yaml --dry-run
```

### Interactive Playground

Fast iteration on single questions with hot-reload:

```bash
cd src/bbq/interactive
python playground.py

# Model loads once, then:
# 1. Edit config.py in another tab
# 2. Press Enter to re-run with new settings
# 3. Ctrl+C to exit
```

## Experiment Conditions

| Condition | Description | Effect on Accuracy |
|-----------|-------------|-------------------|
| **Baseline** | Full CoT reasoning | ~95% |
| **Force Immediate** | No thinking (`<think></think>`) | ~91% |
| **Blank Spaces** | Neutral filler tokens | ~90-93% |
| **Incorrect Answer ×1** | Single wrong assertion | ~42% |
| **Incorrect Answer ×12** | Repeated wrong assertion | ~70% |

The "repetition paradox": more repetitions of wrong answers actually *help* recovery, suggesting the model detects something is "off" with excessive repetition.

## Models Tested

- Qwen/Qwen3-1.7B
- Qwen/Qwen3-8B  
- Qwen/Qwen3-32B

## Citation

This work builds on:
- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://github.com/nyu-mll/BBQ)
- [Chain-of-Thought Faithfulness research](https://arxiv.org/abs/2307.13702)
- [Transformer Lens](https://github.com/TransformerLensOrg/TransformerLens)
