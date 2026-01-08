#!/usr/bin/env python3
"""Blank Spaces Sweep Experiment

Run cells with Shift+Enter in VS Code/Cursor.
Tests how P(A/B/C) changes when varying blank spaces in <think>...</think>
"""

#%% Model Loading (run once) ================================================
import os
import sys
import torch
from transformer_lens import HookedTransformer

# Setup paths
_internals_dir = "/workspace/experiments/exp_006_extend_thinking/internals"
if _internals_dir not in sys.path:
    sys.path.insert(0, _internals_dir)
os.chdir(_internals_dir)

# Setup BBQ paths
_bbq_dir = "/workspace/experiments/exp_006_extend_thinking/bbq"
_shared_dir = "/workspace/experiments/exp_006_extend_thinking"
for _path in [_bbq_dir, _shared_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data.bbq_dataset import load_bbq_items, ALL_CATEGORIES
from shared.config import format_bbq_prompt

# Set HuggingFace cache
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

# Model config (edit here)
MODEL_NAME = "Qwen/Qwen3-8B"

# Load model
print(f"Loading model: {MODEL_NAME}")
model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} dim")

# %%

# Few-shot examples (inline)
FEW_SHOT_EXAMPLES = """Question: What is 2 + 2?
A. 22
B. 4
C. 5

Answer with only the letter (A, B, or C).
Answer: B

Question: How many days are in a week?
A. 5
B. 7
C. 10

Answer with only the letter (A, B, or C).
Answer: B

Question: Which city is the capital of France?
A. London
B. Berlin
C. Paris

Answer with only the letter (A, B, or C).
Answer: C

Question: What is 10 minus 3?
A. 7
B. 13
C. 3

Answer with only the letter (A, B, or C).
Answer: A

Question: Which shape is round?
A. Square
B. Circle
C. Triangle

Answer with only the letter (A, B, or C).
Answer: B

Question: What is the freezing point of water in Celsius?
A. 100
B. 10
C. 0

Answer with only the letter (A, B, or C).
Answer: C

Question: What is 3 times 3?
A. 6
B. 9
C. 33

Answer with only the letter (A, B, or C).
Answer: B

Question: How many years are in a decade?
A. 10
B. 100
C. 1000

Answer with only the letter (A, B, or C).
Answer: A

Question: Which is the color of a ripe banana?
A. Purple
B. Red
C. Yellow

Answer with only the letter (A, B, or C).
Answer: C

Question: What is half of 100?
A. 50
B. 25
C. 200

Answer with only the letter (A, B, or C).
Answer: A

"""

print("Ready!")


#%% Config (edit these) =====================================================

USE_FEW_SHOT = True
N_VALUES = [0, 1, 25, 50]

# Position/Repetition experiment constants
NEUTRAL_FILLER = "Hmm, let me think. "
A_PHRASES = [
    "The answer is A!", "A is correct.", "It's clearly A.", "I believe it's A.",
    "A seems right.", "Definitely A.", "A is the right choice.", "The correct answer is A.",
    "I'm confident it's A.", "A is the one.", "Going with A.", "A for sure.",
    "Must be A.", "A is obvious.", "Choosing A.", "A without doubt.",
    "It has to be A.", "A is my answer.", "Clearly A.", "A makes sense.",
    "The solution is A.", "A is it.", "Picking A.", "A is best.", "Answer: A.",
]

# BBQ Question Mode - set USE_BBQ = True to load from dataset
USE_BBQ = True
BBQ_CATEGORY = "disability"  # Options: age, disability, gender, nationality, appearance, race, religion, ses, sexual_orientation
BBQ_INDEX = 1

# Custom prompt mode (when USE_BBQ = False)
CUSTOM_PROMPT = """
A warehouse starts with 50 boxes.

First, a delivery truck removes half of the boxes.

Second, a new shipment arrives, tripling the number of boxes currently in the warehouse.

Third, 5 boxes are damaged and removed.

Finally, the remaining boxes are divided equally onto 2 pallets.

How many boxes are on each pallet?

Options: 
A. 30

B. 32.5

C. 35
"""

# Build prompt from BBQ or custom
if USE_BBQ:
    items = load_bbq_items(categories=[BBQ_CATEGORY], n_per_category=max(10, BBQ_INDEX + 1))
    if BBQ_INDEX >= len(items):
        print(f"Warning: BBQ_INDEX {BBQ_INDEX} out of range, using 0")
        bbq_item = items[0]
    else:
        bbq_item = items[BBQ_INDEX]
    bbq_prompt = format_bbq_prompt(bbq_item.context, bbq_item.question, bbq_item.choices)
    print(f"Loaded BBQ: {BBQ_CATEGORY} #{BBQ_INDEX} | Correct: {bbq_item.correct_letter}")
else:
    bbq_prompt = CUSTOM_PROMPT
    bbq_item = None


#%% Blank Spaces Sweep Experiment ===========================================

print(bbq_prompt)
# %%
# Prepend few-shot examples if enabled
prompt_text = FEW_SHOT_EXAMPLES + bbq_prompt if USE_FEW_SHOT else bbq_prompt

# Get token IDs for A, B, C
def get_answer_token_ids(model):
    """Get token IDs for A, B, C answers (with and without space prefix)."""
    try:
        A_sp = model.to_single_token(" A")
        B_sp = model.to_single_token(" B")
        C_sp = model.to_single_token(" C")
    except Exception:
        A_sp = model.to_single_token("A")
        B_sp = model.to_single_token("B")
        C_sp = model.to_single_token("C")
    return A_sp, B_sp, C_sp

A_sp, B_sp, C_sp = get_answer_token_ids(model)
A_id = model.to_single_token("A")
B_id = model.to_single_token("B")
C_id = model.to_single_token("C")
all_answer_ids = [A_id, B_id, C_id, A_sp, B_sp, C_sp]

# Run sweep
print("=" * 80)
print("BLANK SPACES SWEEP: How do probabilities change with N spaces in thinking?")
print("=" * 80)
print(f"\nQuestion: {bbq_prompt.strip()[:100]}...")
print(f"\nPrompt template: <think>[N spaces]</think>\\n\\n Answer:")
print()

results = []

for N in N_VALUES:
    forced_cot = "The answer is A! " * N
    prompt = (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n\n Answer:"
    )
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Get final layer logits
    resid = cache["resid_post", model.cfg.n_layers - 1][0]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    pos_logits = logits[-1]
    probs = torch.softmax(pos_logits.float(), dim=-1)
    
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    
    top_idx = pos_logits.argmax().item()
    top_token = model.to_single_str_token(top_idx)
    
    # Generate to see actual output
    with torch.no_grad():
        generated = model.generate(tokens, max_new_tokens=3, temperature=0)
    gen_text = model.to_string(generated[0, tokens.shape[1]:])
    
    results.append({
        'N': N,
        'P(A)': p_a,
        'P(B)': p_b, 
        'P(C)': p_c,
        'top': top_token.strip(),
        'gen': gen_text.strip()[:10]
    })

# Print summary table
print(f"{'N':<6} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10} {'Top':<6} {'Generated':<12}")
print("-" * 60)
for r in results:
    probs_list = [r['P(A)'], r['P(B)'], r['P(C)']]
    max_p = max(probs_list)
    winner = ['A', 'B', 'C'][probs_list.index(max_p)]
    print(f"{r['N']:<6} {r['P(A)']:<10.4f} {r['P(B)']:<10.4f} {r['P(C)']:<10.4f} {r['top']:<6} {r['gen']:<12} -> {winner}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print(f"  - N=0 (empty think): P(A)={results[0]['P(A)']:.4f}, P(B)={results[0]['P(B)']:.4f}, P(C)={results[0]['P(C)']:.4f}")
print(f"  - N={N_VALUES[-1]} ({N_VALUES[-1]} spaces): P(A)={results[-1]['P(A)']:.4f}, P(B)={results[-1]['P(B)']:.4f}, P(C)={results[-1]['P(C)']:.4f}")
delta_a = results[-1]['P(A)'] - results[0]['P(A)']
delta_b = results[-1]['P(B)'] - results[0]['P(B)']
delta_c = results[-1]['P(C)'] - results[0]['P(C)']
print(f"  - Change: ΔP(A)={delta_a:+.4f}, ΔP(B)={delta_b:+.4f}, ΔP(C)={delta_c:+.4f}")
print("=" * 80)

#%% Position/Repetition Comparison ==========================================

# Define conditions: (name, forced_cot_string)
REAL_REASONING = "Let me think about this question. Looking at the context, I need to consider what was stated. Based on the information given, A appears correct."
conditions = [
    ("Check C", "The answer is C! "), 
    ("N=1", "The answer is A! "),
    ("N=1 and blanks", "The answer is A! " + " " * 25),
    ("N=2", "The answer is A! " * 2),
    ("N=3", "The answer is A! " * 3),
    ("N=5", "The answer is A! " * 5),
    ("N=1period", "The answer is A. "),
    ("N=2period", "The answer is A. "),
    ("final_check", "The answer might be A. I think the answer is A."), 
    ("final_check_2", "Let me consider the options carefully. Based on the context provided, I believe A is most likely correct."), 
    ("final_check_2_and_blanks", "Let me consider the options carefully. Based on the context provided, I believe A is most likely correct." + " " * 25), 
    ("REAL_REASON", REAL_REASONING),
    ("FILLER_ONLY", NEUTRAL_FILLER * 10),
    ("EARLY_SINGLE", "The answer is A! " + NEUTRAL_FILLER * 49),
    ("LATE_SINGLE", NEUTRAL_FILLER * 49 + "The answer is A! "),
    ("VARIED_25", " ".join(A_PHRASES)),
    ("baseline", "The answer is A!"),
    ("spaces", "The answer is A!" + " " * 25),
    ("newlines", "The answer is A!" + "\n" * 5),
    ("periods", "The answer is A!" + "..." * 8),
    ("done", "The answer is A! I'm done."),
    ("confident", "The answer is A! I'm certain."),
]

print("\n" + "=" * 80)
print("POSITION/REPETITION TEST: Disambiguating positional decay vs repetition detection")
print("=" * 80)

pos_results = []
for name, forced_cot in conditions:
    prompt = (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n\n Answer:"
    )
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    resid = cache["resid_post", model.cfg.n_layers - 1][0]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    pos_logits = logits[-1]
    probs = torch.softmax(pos_logits.float(), dim=-1)
    
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    
    with torch.no_grad():
        generated = model.generate(tokens, max_new_tokens=3, temperature=0)
    gen_text = model.to_string(generated[0, tokens.shape[1]:])
    
    pos_results.append({
        'name': name,
        'P(A)': p_a, 'P(B)': p_b, 'P(C)': p_c,
        'gen': gen_text.strip()[:10],
        'n_tokens': tokens.shape[1]
    })

# Print comparison table with predictions
print(f"\n{'Condition':<14} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10} {'Gen':<12} {'Tokens':<8}")
print("-" * 70)
for r in pos_results:
    print(f"{r['name']:<14} {r['P(A)']:<10.4f} {r['P(B)']:<10.4f} {r['P(C)']:<10.4f} {r['gen']:<12} {r['n_tokens']:<8}")

print("\n" + "-" * 70)
print("HYPOTHESIS TEST:")
print(f"  N=1 baseline:     P(A)={pos_results[0]['P(A)']:.4f}")
print(f"  EARLY_SINGLE:     P(A)={pos_results[1]['P(A)']:.4f}  (if positional decay: ~0)")
print(f"  LATE_SINGLE:      P(A)={pos_results[2]['P(A)']:.4f}  (if positional decay: ~baseline)")
print(f"  VARIED_25:        P(A)={pos_results[3]['P(A)']:.4f}  (if repetition-detection: high; if noise: ~0)")
print("=" * 80)

#%% N vs P(A) Graph: How does P(A) scale with repetitions? ====================
import matplotlib.pyplot as plt

N_SWEEP_VALUES = [1, 2, 3, 5, 10, 15, 25]

print("\n" + "=" * 80)
print("N vs P(A) SWEEP: Testing repetitions n=1, 2, 3, 5, 10, 15, 25")
print("=" * 80)

n_sweep_results = []
for n in N_SWEEP_VALUES:
    forced_cot = "The answer is A! " * n
    prompt = (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n\n Answer:"
    )
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    resid = cache["resid_post", model.cfg.n_layers - 1][0]
    resid = model.ln_final(resid)
    logits = resid @ model.W_U
    if model.b_U is not None:
        logits = logits + model.b_U
    
    pos_logits = logits[-1]
    probs = torch.softmax(pos_logits.float(), dim=-1)
    
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    
    n_sweep_results.append({
        'n': n,
        'P(A)': p_a,
        'P(B)': p_b,
        'P(C)': p_c,
        'n_tokens': tokens.shape[1]
    })
    print(f"  n={n:>2}: P(A)={p_a:.4f}, P(B)={p_b:.4f}, P(C)={p_c:.4f}, tokens={tokens.shape[1]}")

# Extract data for plotting
ns = [r['n'] for r in n_sweep_results]
p_as = [r['P(A)'] for r in n_sweep_results]

# Create the graph
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(ns, p_as, 'o-', color='#2ecc71', linewidth=2, markersize=8)

ax.set_xlabel('n (repetitions of "The answer is A!")', fontsize=12)
ax.set_ylabel('P(A)', fontsize=12)
ax.set_title('P(A) vs Number of Repetitions', fontsize=14, fontweight='bold')
ax.set_xticks(ns)
ax.set_ylim(0, 0.4)
ax.grid(True, alpha=0.3)

# Add annotations for P(A) values
for i, (n, p_a) in enumerate(zip(ns, p_as)):
    ax.annotate(f'{p_a:.3f}', (n, p_a), textcoords='offset points', 
                xytext=(0, 10), ha='center', fontsize=9, color='#2ecc71')

plt.tight_layout()
plt.savefig('n_vs_pa_graph.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nGraph saved to: n_vs_pa_graph.png")
print("=" * 80)

#%% Layer-wise Analysis: Where does N=2 diverge from N=1? ====================
import matplotlib.pyplot as plt

layer_conditions = [
    ("N=1", "The answer is A! "),
    ("N=2", "The answer is A! " * 2),
]

layer_results = {}
for name, forced_cot in layer_conditions:
    prompt = (
        f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>{forced_cot}</think>\n\n Answer:"
    )
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    p_a_by_layer = []
    for L in range(model.cfg.n_layers):
        resid = cache["resid_post", L][0, -1]  # final position
        resid = model.ln_final(resid)
        logits = resid @ model.W_U
        if model.b_U is not None:
            logits = logits + model.b_U
        probs = torch.softmax(logits.float(), dim=-1)
        p_a = probs[A_id].item() + probs[A_sp].item()
        p_a_by_layer.append(p_a)
    
    layer_results[name] = p_a_by_layer

# Plot
plt.figure(figsize=(12, 5))
layers = list(range(model.cfg.n_layers))
for name, p_a_vals in layer_results.items():
    plt.plot(layers, p_a_vals, marker='o', markersize=3, label=name)

plt.xlabel("Layer")
plt.ylabel("P(A)")
# plt.title("P(A) by Layer: N=1 vs N=2 — Where Does Divergence Happen?")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Annotate divergence
# diff = [abs(layer_results["N=1"][i] - layer_results["N=2"][i]) for i in layers]
# max_diff_layer = diff.index(max(diff))
# plt.axvline(x=max_diff_layer, color='red', linestyle=':', alpha=0.7, label=f'Max divergence @ L{max_diff_layer}')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nMax divergence at layer {max_diff_layer}: ΔP(A) = {max(diff):.4f}")
print(f"  L0-10 = low-level pattern matching")
print(f"  L11-24 = mid-level processing")  
print(f"  L25+ = high-level reasoning validation")

#%% Activation Patching: Which layer restores P(A)? ==========================

PATCH_LAYERS = [31, 32, 33, 34, 35]

# Build prompts
n1_prompt = (
    f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
    f"<|im_start|>assistant\n<think>The answer is A! </think>\n\n Answer:"
)
n2_prompt = (
    f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
    f"<|im_start|>assistant\n<think>{'The answer is A! ' * 2}</think>\n\n Answer:"
)

n1_tokens = model.to_tokens(n1_prompt)
n2_tokens = model.to_tokens(n2_prompt)

# Cache N=1 residuals at final position
with torch.no_grad():
    _, n1_cache = model.run_with_cache(n1_tokens)

n1_resid = {L: n1_cache["resid_post", L][0, -1].clone() for L in PATCH_LAYERS}

# Get baseline P(A) for N=2 (no patching)
with torch.no_grad():
    _, n2_cache = model.run_with_cache(n2_tokens)
resid = n2_cache["resid_post", model.cfg.n_layers - 1][0, -1]
resid = model.ln_final(resid)
logits = resid @ model.W_U + (model.b_U if model.b_U is not None else 0)
probs = torch.softmax(logits.float(), dim=-1)
n2_baseline_pa = probs[A_id].item() + probs[A_sp].item()

# Patch each layer and measure P(A)
patch_results = []
for L in PATCH_LAYERS:
    def patch_hook(resid_post, hook, layer=L):
        resid_post[0, -1] = n1_resid[layer]
        return resid_post
    
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            n2_tokens,
            fwd_hooks=[(f"blocks.{L}.hook_resid_post", patch_hook)]
        )
    
    final_logits = patched_logits[0, -1]
    probs = torch.softmax(final_logits.float(), dim=-1)
    p_a = probs[A_id].item() + probs[A_sp].item()
    patch_results.append((L, p_a))

# Get N=1 baseline for comparison
resid = n1_cache["resid_post", model.cfg.n_layers - 1][0, -1]
resid = model.ln_final(resid)
logits = resid @ model.W_U + (model.b_U if model.b_U is not None else 0)
probs = torch.softmax(logits.float(), dim=-1)
n1_baseline_pa = probs[A_id].item() + probs[A_sp].item()

# Print results
print("=" * 70)
print("ACTIVATION PATCHING: Patch N=1 resid_post into N=2 at final position")
print("=" * 70)
print(f"\nBaselines:  N=1 P(A)={n1_baseline_pa:.4f}  |  N=2 P(A)={n2_baseline_pa:.4f}")
print(f"\n{'Layer':<8} {'P(A) after patch':<18} {'Recovery':<12}")
print("-" * 45)
for L, p_a in patch_results:
    recovery = (p_a - n2_baseline_pa) / (n1_baseline_pa - n2_baseline_pa + 1e-8) * 100
    bar = "█" * int(min(recovery, 100) / 5)
    print(f"L{L:<6} {p_a:<18.4f} {recovery:>6.1f}% {bar}")

best_layer = max(patch_results, key=lambda x: x[1])
print(f"\nBest restoration: Layer {best_layer[0]} → P(A)={best_layer[1]:.4f}")
print("This is where the 'trust this reasoning' computation likely happens.")

#%% Attention Pattern Analysis: Which heads detect repetition? ================

ATTN_LAYERS = [27, 28, 29, 30]

# Re-run with attention patterns cached
with torch.no_grad():
    _, n1_cache_attn = model.run_with_cache(n1_tokens, names_filter=lambda n: "pattern" in n or "resid_post" in n)
    _, n2_cache_attn = model.run_with_cache(n2_tokens, names_filter=lambda n: "pattern" in n or "resid_post" in n)

# Find thinking block token ranges
n1_think_start = (model.to_string(n1_tokens[0]).find("<think>") + len("<think>"))
n2_think_start = (model.to_string(n2_tokens[0]).find("<think>") + len("<think>"))

# Get token positions for thinking content (approximate by finding </think>)
n1_str = model.to_string(n1_tokens[0])
n2_str = model.to_string(n2_tokens[0])

print("=" * 70)
print("ATTENTION ANALYSIS: Which heads attend differently to repeated content?")
print("=" * 70)
print(f"\nN=1 tokens: {n1_tokens.shape[1]} | N=2 tokens: {n2_tokens.shape[1]}")

# For each layer, find heads with biggest attention difference at final position
head_diffs = []
for L in ATTN_LAYERS:
    n1_attn = n1_cache_attn["pattern", L][0]  # [n_heads, seq, seq]
    n2_attn = n2_cache_attn["pattern", L][0]
    
    n_heads = n1_attn.shape[0]
    
    # Attention from final position to all positions
    n1_final_attn = n1_attn[:, -1, :]  # [n_heads, seq]
    n2_final_attn = n2_attn[:, -1, :]  # [n_heads, seq]
    
    # For N=2: how much does each head attend to the "extra" repeated tokens?
    # The extra tokens in N=2 are roughly positions [n1_tokens.shape[1]-10 : n2_tokens.shape[1]-10]
    # But simpler: just measure entropy or concentration differences
    
    for h in range(n_heads):
        # Measure attention entropy (low entropy = focused, high = diffuse)
        n1_entropy = -(n1_final_attn[h] * torch.log(n1_final_attn[h] + 1e-10)).sum().item()
        n2_entropy = -(n2_final_attn[h] * torch.log(n2_final_attn[h] + 1e-10)).sum().item()
        
        # Attention to last 20% of sequence (where repeated content would be)
        n1_late_attn = n1_final_attn[h, -int(n1_tokens.shape[1]*0.2):].sum().item()
        n2_late_attn = n2_final_attn[h, -int(n2_tokens.shape[1]*0.2):].sum().item()
        
        head_diffs.append({
            'layer': L,
            'head': h,
            'n1_entropy': n1_entropy,
            'n2_entropy': n2_entropy,
            'entropy_diff': n2_entropy - n1_entropy,
            'n1_late': n1_late_attn,
            'n2_late': n2_late_attn,
            'late_diff': n2_late_attn - n1_late_attn
        })

# Sort by biggest difference in late attention
head_diffs.sort(key=lambda x: abs(x['late_diff']), reverse=True)

print(f"\nTop 10 heads with biggest attention difference (late tokens):")
print(f"{'L.H':<8} {'N1 late%':<10} {'N2 late%':<10} {'Δ late':<10} {'Δ entropy':<10}")
print("-" * 50)
for hd in head_diffs[:10]:
    print(f"L{hd['layer']}.H{hd['head']:<3} {hd['n1_late']*100:<10.1f} {hd['n2_late']*100:<10.1f} {hd['late_diff']*100:>+8.1f}% {hd['entropy_diff']:>+8.2f}")

# Identify candidate repetition detectors
print("\n" + "-" * 50)
print("Candidate repetition detectors (attend MORE to late tokens in N=2):")
for hd in head_diffs[:5]:
    if hd['late_diff'] > 0:
        print(f"  L{hd['layer']}.H{hd['head']}: +{hd['late_diff']*100:.1f}% more attention to late tokens")

# %%
#%% KNOCKOUT: Does disabling L27.H13 restore P(A) for N=2?

candidate_heads = [(27, 13), (28, 19), (27, 16), (27, 19)]

def make_knockout_hook(head_idx):
    def hook(z, hook):
        z[0, :, head_idx, :] = 0
        return z
    return hook

print("=" * 60)
print("HEAD KNOCKOUT: Does disabling candidates restore N=2 P(A)?")
print("=" * 60)
print(f"\nBaselines:  N=1 P(A)={n1_baseline_pa:.4f}  |  N=2 P(A)={n2_baseline_pa:.4f}")
print(f"\n{'Head':<12} {'P(A)':<10} {'Recovery':<10}")
print("-" * 35)

for layer, head in candidate_heads:
    with torch.no_grad():
        logits = model.run_with_hooks(
            n2_tokens,
            fwd_hooks=[(f"blocks.{layer}.attn.hook_z", make_knockout_hook(head))]
        )
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    p_a = probs[A_id].item() + probs[A_sp].item()
    recovery = (p_a - n2_baseline_pa) / (n1_baseline_pa - n2_baseline_pa + 1e-8) * 100
    print(f"L{layer}.H{head:<4} {p_a:<10.4f} {recovery:>6.1f}%")
# %%
# Knockout combinations
combos = [
    [(27, 13), (27, 19)],
    [(27, 13), (27, 16), (27, 19)],
    [(27, 13), (28, 19), (27, 16), (27, 19)],  # all top 4
]

def make_multi_knockout_hook(heads_in_layer):
    def hook(z, hook):
        for head_idx in heads_in_layer:
            z[0, :, head_idx, :] = 0
        return z
    return hook

print(f"\n{'Heads knocked out':<40} {'P(A)':<10} {'Recovery':<10}")
print("-" * 60)

for combo in combos:
    # Group by layer
    by_layer = {}
    for l, h in combo:
        by_layer.setdefault(l, []).append(h)
    
    hooks = [(f"blocks.{l}.attn.hook_z", make_multi_knockout_hook(heads)) 
             for l, heads in by_layer.items()]
    
    with torch.no_grad():
        logits = model.run_with_hooks(n2_tokens, fwd_hooks=hooks)
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    p_a = probs[A_id].item() + probs[A_sp].item()
    recovery = (p_a - n2_baseline_pa) / (n1_baseline_pa - n2_baseline_pa + 1e-8) * 100
    
    combo_str = ", ".join([f"L{l}.H{h}" for l, h in combo])
    print(f"{combo_str:<40} {p_a:<10.4f} {recovery:>6.1f}%")
# %%
# Compare: Force "The answer is C!" (biased answer) vs "The answer is A!" (non-biased)
# If safety circuit, forcing C should be suppressed MORE than forcing A
#%% BIAS TEST: Are these heads suppressing biased answers specifically?

test_assertions = [
    ("Force A (non-biased)", "The answer is A!"),
    ("Force C (biased)", "The answer is C!"),
]

knockout_heads = [(27, 13), (28, 19), (27, 16), (27, 19)]

def make_multi_knockout_hook(heads_in_layer):
    def hook(z, hook):
        for head_idx in heads_in_layer:
            z[0, :, head_idx, :] = 0
        return z
    return hook

# Group heads by layer for hooks
by_layer = {}
for l, h in knockout_heads:
    by_layer.setdefault(l, []).append(h)
knockout_hooks = [(f"blocks.{l}.attn.hook_z", make_multi_knockout_hook(heads)) 
                  for l, heads in by_layer.items()]

print("=" * 70)
print("BIAS TEST: Do these heads suppress biased (C) more than non-biased (A)?")
print("=" * 70)

print(f"\n{'Condition':<25} {'P(A)':<10} {'P(B)':<10} {'P(C)':<10}")
print("-" * 55)

for name, assertion in test_assertions:
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>{assertion}</think>\n\n Answer:"
    tokens = model.to_tokens(prompt)
    
    # Without knockout
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    print(f"{name:<25} {p_a:<10.4f} {p_b:<10.4f} {p_c:<10.4f}")

print("-" * 55)
print("WITH KNOCKOUT (L27.H13, L27.H16, L27.H19, L28.H19):")
print("-" * 55)

for name, assertion in test_assertions:
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>{assertion}</think>\n\n Answer:"
    tokens = model.to_tokens(prompt)
    
    # With knockout
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=knockout_hooks)
    probs = torch.softmax(logits[0, -1].float(), dim=-1)
    p_a = probs[A_id].item() + probs[A_sp].item()
    p_b = probs[B_id].item() + probs[B_sp].item()
    p_c = probs[C_id].item() + probs[C_sp].item()
    print(f"{name:<25} {p_a:<10.4f} {p_b:<10.4f} {p_c:<10.4f}")

print("\n" + "=" * 70)
print("If knockout increases P(C) more than P(A), these heads are bias suppressors")
print("=" * 70)
# Same but with knockout hooks
# %%
#%% ATTENTION FROM FINAL POSITION: Where does it look in N=1 vs N=2?

n1_cot = "The answer is A!"
n2_cot = "The answer is A! The answer is A!"

for name, cot in [("N=1", n1_cot), ("N=2", n2_cot)]:
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>{cot}</think>\n\n Answer:"
    tokens = model.to_tokens(prompt)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Find where "A" tokens are in the sequence
    token_strs = model.to_str_tokens(tokens[0])
    a_positions = [i for i, t in enumerate(token_strs) if 'A' in t and i > len(tokens[0]) - 50]
    
    print(f"\n{name}: 'A' tokens at positions {a_positions}")
    print(f"Token strings around thinking: {token_strs[-30:]}")
    
    # Attention from final position to A positions, for key heads
    print(f"Attention from final pos to 'A' positions (layers 26-30):")
    for L in [26, 27, 28, 29, 30]:
        attn = cache["pattern", L][0]  # [n_heads, seq, seq]
        final_pos = tokens.shape[1] - 1
        for a_pos in a_positions:
            attn_to_a = attn[:, final_pos, a_pos].mean().item()  # avg across heads
            print(f"  L{L} -> pos {a_pos}: {attn_to_a:.4f}")
# %%
#%% COMPARE REPRESENTATIONS AT 'A' POSITIONS

n1_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>The answer is A.</think>\n\n Answer:"
n2_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>The answer is A. The answer is A.</think>\n\n Answer:"

n1_tokens = model.to_tokens(n1_prompt)
n2_tokens = model.to_tokens(n2_prompt)

with torch.no_grad():
    _, n1_cache = model.run_with_cache(n1_tokens)
    _, n2_cache = model.run_with_cache(n2_tokens)

# Find the 'A' token positions in thinking block
n1_strs = model.to_str_tokens(n1_tokens[0])
n2_strs = model.to_str_tokens(n2_tokens[0])

# Get position of ' A' right after ' is' in thinking
n1_a_pos = [i for i, t in enumerate(n1_strs) if t == ' A' and i > 500][0]  # first A in thinking
n2_a_pos_1 = [i for i, t in enumerate(n2_strs) if t == ' A' and i > 500][0]  # first A
n2_a_pos_2 = [i for i, t in enumerate(n2_strs) if t == ' A' and i > 500][1]  # second A

print(f"N=1 'A' at pos {n1_a_pos}: {n1_strs[n1_a_pos-2:n1_a_pos+2]}")
print(f"N=2 first 'A' at pos {n2_a_pos_1}: {n2_strs[n2_a_pos_1-2:n2_a_pos_1+2]}")
print(f"N=2 second 'A' at pos {n2_a_pos_2}: {n2_strs[n2_a_pos_2-2:n2_a_pos_2+2]}")

# Compare residual stream at these positions (late layers where divergence happens)
print(f"\nCosine similarity of 'A' representations:")
for L in [26, 27, 28, 29, 30]:
    n1_resid = n1_cache["resid_post", L][0, n1_a_pos]
    n2_resid_1 = n2_cache["resid_post", L][0, n2_a_pos_1]
    n2_resid_2 = n2_cache["resid_post", L][0, n2_a_pos_2]
    
    sim_1 = torch.cosine_similarity(n1_resid, n2_resid_1, dim=0).item()
    sim_2 = torch.cosine_similarity(n1_resid, n2_resid_2, dim=0).item()
    sim_12 = torch.cosine_similarity(n2_resid_1, n2_resid_2, dim=0).item()
    
    print(f"  L{L}: N1↔N2_first={sim_1:.4f}, N1↔N2_second={sim_2:.4f}, N2_first↔N2_second={sim_12:.4f}")
# %%
#%% WHAT DIRECTION ENCODES "REPEATED"?

# The difference between second A and first A
L = 28 # pick a layer
diff = n2_cache["resid_post", L][0, n2_a_pos_2] - n2_cache["resid_post", L][0, n2_a_pos_1]
diff = diff / diff.norm()  # normalize

# Project this direction to vocab space - what does "repeated" mean?
diff_logits = diff @ model.W_U
top_tokens = diff_logits.topk(10)
bottom_tokens = diff_logits.topk(10, largest=False)

print("Direction from first A → second A decodes to:")
print("\nMost positive (second A direction):")
for i, idx in enumerate(top_tokens.indices):
    print(f"  {model.to_single_str_token(idx.item())!r}: {top_tokens.values[i].item():.2f}")

print("\nMost negative (first A direction):")
for i, idx in enumerate(bottom_tokens.indices):
    print(f"  {model.to_single_str_token(idx.item())!r}: {bottom_tokens.values[i].item():.2f}")
# %%
#%% WHAT DIRECTION DO BLANKS ADD?

# prompt_text should already be defined from earlier cells
# If not, uncomment: prompt_text = FEW_SHOT_EXAMPLES + bbq_prompt

baseline_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>The answer is A.</think>\n\n Answer:"
blanks_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>The answer is A.{' ' * 25}</think>\n\n Answer:"

baseline_tokens = model.to_tokens(baseline_prompt)
blanks_tokens = model.to_tokens(blanks_prompt)

with torch.no_grad():
    _, baseline_cache = model.run_with_cache(baseline_tokens)
    _, blanks_cache = model.run_with_cache(blanks_tokens)

# Compare final position (right before "Answer:")
L = 28
baseline_final = baseline_cache["resid_post", L][0, -1]
blanks_final = blanks_cache["resid_post", L][0, -1]

diff = blanks_final - baseline_final
diff = diff / diff.norm()

# Project to vocab
diff_logits = diff @ model.W_U
top_tokens = diff_logits.topk(10)
bottom_tokens = diff_logits.topk(10, largest=False)

print("Direction from baseline → blanks decodes to:")
print("\nMost positive (blanks direction):")
for i, idx in enumerate(top_tokens.indices):
    print(f"  {model.to_single_str_token(idx.item())!r}: {top_tokens.values[i].item():.2f}")

print("\nMost negative (baseline direction):")
for i, idx in enumerate(bottom_tokens.indices):
    print(f"  {model.to_single_str_token(idx.item())!r}: {bottom_tokens.values[i].item():.2f}")
# %%
