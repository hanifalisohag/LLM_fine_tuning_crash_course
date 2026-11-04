# LLM Crash Course: Fine-Tuning, Quantization and Quantized Inference

A hands-on course repo for taking a small Llama model from a 4-bit base
checkpoint, through a QLoRA supervised fine-tune, into three saved formats, and
then running inference on the quantized weights with two different engines
(Unsloth/PyTorch and llama.cpp/C++).

Everything runs from Jupyter notebooks; there is no library code to import. The
repo is built so each concept in it can be read in theory and then immediately
tried in a cell.

| Notebook | Concepts it makes concrete |
| --- | --- |
| `quantization.ipynb` | Affine int8 quantization, scale/zero-point, quantization error, NF4 codebooks. Runs on CPU. |
| `unsloth_finetuned_hanif.ipynb` | 4-bit base load, LoRA adapters, prompt formatting, `SFTTrainer`, streaming inference, merged export, GGUF quantization, llama.cpp inference and Ollama serving. |

## Repository layout

| Path | What it is |
| --- | --- |
| `unsloth_finetuned_hanif.ipynb` | The fine-tuning run in `trl.SFTConfig` style plus the quantized inference sections. |
| `quantization.ipynb` | Theory notebook: linear int8 and simulated NF4 quantization with dequantization plots. |
| `requirements-crash-course.txt` | Pinned dependencies (`unsloth`, `llama-cpp-python`, numpy, matplotlib). |
| `requirements-base.txt` | Jupyter/IPython runtime pulled in by the above. |
| `outputs/` | Training output directory (`checkpoint-60`, trainer state, generated model card). |
| `service-001-3B/` | Saved LoRA adapter (PEFT) + tokenizer + generated `ollama_modelfile`. |
| `service-001-3B-merged/` | Merged 16-bit safetensors model. |
| `service-001-3B-GGUF/` | Export directory from `save_pretrained_gguf` (holds merged weights and tokenizer). |
| `service-001-3B-GGUF_gguf/` | The real quantized artifact: `Llama-3.2-3B-Instruct.Q8_0.gguf` (~3.4 GB) and its `Modelfile`. |
| `unsloth_compiled_cache/` | Unsloth's auto-compiled trainers; generated, safe to delete. |

Large model directories and `outputs/` are in `.gitignore`, so they stay local.
Only the notebooks and requirement files are tracked.

---

# Part 1 — Theory, in plain words

Six short sections, three formulas in total, no maths you will not use. Each
section ends where the notebook uses the idea.

## 1.1 What fine-tuning is

A model is just a huge pile of numbers called **weights**. After pretraining it
is good at language in general, but it does not know your job. **Fine-tuning**
means training it a bit more on a small pile of your own examples so it gets good
at that job.

Training works by guessing the next word. Show the model a sentence, hide the
last word, let it guess, then nudge the weights slightly if it guessed wrong. For
a text cut into $N$ words (tokens) $y_1 \dots y_N$:

$$
\mathcal{L} = -\frac{1}{N}\sum_{t=1}^{N} \log p(y_t \mid y_{1:t-1})
$$

Read it as: *the loss is small when the model gave a high probability to the word
that really came next*. Training shrinks this number.

The catch is **full fine-tuning**: to change a weight you need to remember a few
extra numbers per weight (a gradient, plus two optimizer values). That is about
16 bytes per weight, so a 3-billion-weight model wants roughly
$3\times10^9 \times 16 \approx 48$ GB — more than most GPUs have. Everything below
is a way around that.

## 1.2 PEFT and LoRA — train a small extra piece instead

**PEFT** means *Parameter-Efficient Fine-Tuning*: freeze the big model and train a
tiny new part next to it. Frozen weights need no extra memory for training, which
is where the saving comes from.

Think of the model as a printed textbook you are not allowed to rewrite. LoRA
gives you a small sticky note of corrections instead. In maths, the frozen weight
matrix $W_0$ stays as it is and we add a correction:

$$
W' = W_0 + \frac{\alpha}{r} B A
$$

- $W_0$ — the original weights, frozen.
- $B$ and $A$ — the sticky note: thin matrices, so their product $BA$ can only
  make a "simple" correction. That is enough, because a correction usually is
  simple.
- $r$ — the **rank**, how much room the note has. Here $r = 16$.
- $\alpha / r$ — a volume knob on the correction. Here $\alpha = 16$, so the two
  cancel and the note starts quiet.

Why it is cheap: a $4096 \times 4096$ matrix has 16.7 million numbers; with $r=16$
the note has $4096 \times 16 + 16 \times 4096 \approx 131{,}000$ — about 128 times
fewer. Across the whole model the adapter is under 1% of the parameters.

Why it is safe: $B$ starts at zero, so at the first step $W' = W_0$ exactly and the
model behaves like it always did. Nothing breaks before training starts.

Why it is flexible: $W_0$ was never touched, so the adapter is a separate small
file. Keep it beside the base model, or add it in permanently ("merge") later.

In the notebook, `get_peft_model` attaches a note to seven places per layer —
`q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) and
`gate_proj`, `up_proj`, `down_proj` (the feed-forward part).

## 1.3 SFT — fine-tuning on good answers

**SFT** (*Supervised Fine-Tuning*) is about *what* you train on: pairs of a
question and a good answer. We want the model to write answers, not questions, so
the question part of the example is ignored when measuring the loss, and only the
answer part counts. That is all "loss masking" means.

One practical detail matters a lot for Llama: chat models are trained on a very
specific layout of marker tokens, for Llama-3 things like
`<|start_header_id|>user<|end_header_id|> … <|eot_id|>`. Training and inference
must use the same layout, which is why the repo calls
`get_chat_template(tokenizer, chat_template="llama-3.1")` and why the `Modelfile`
lists the same stop tokens. Use a different layout at inference and the model sees
something it was never trained on.

The dataset here, `ServiceNow-AI/R1-Distill-SFT` (`v0`), gives each row as a
`problem`, a long step-by-step `reannotated_assistant_content` reasoning trace,
and a `solution`. The prompt template glues them together, and because the trace
is included the model learns to think step by step before answering:

```text
You are a reflective assistant engaging in thorough, iterative reasoning, ...
<problem>
{problem}
</problem>

{reasoning_trace}
{solution}
```

## 1.4 Quantization — store numbers with fewer bits

Weights are decimals like `0.0137`. Held in 16 bits each takes 2 bytes. Most of
that precision is wasted, so **quantization** rounds each weight to one of a small
set of allowed values, letting it be stored in 8 or 4 bits.

The simplest recipe is like measuring height to the nearest centimetre instead of
the nearest millimetre. Pick a **step size** $S$ (how far apart the allowed values
are) and an offset $Z$, then round:

$$
q = \mathrm{round}\left(\frac{x}{S} + Z\right)
\qquad
\hat{x} = S(q - Z)
$$

$q$ is the small integer you store, $\hat x$ is the value you get back. The mistake
is never worse than half a step, $|x - \hat x| \le S/2$ — and fewer bits means
bigger steps, so more mistake. `quantization.ipynb` computes this for 8 bits and
plots the original values against the recovered ones so you can see the error.

8 bits gives 256 allowed values; 4 bits only 16, which damages the model. **NF4**
fixes that: real weights cluster near zero in a bell curve, so NF4 puts its 16
values where the weights actually are instead of spacing them evenly. Same 4 bits,
much less damage.

Real GGUF files go one step further: instead of one step size for a whole matrix,
they use a fresh scale for every small block of 32 weights, so each block gets its
own precision. That is what `Q8_0` in
`service-001-3B-GGUF_gguf/Llama-3.2-3B-Instruct.Q8_0.gguf` means — 8-bit blocks
plus a small scale each. `Q4_K_M`, `Q5_K_M` and `f16` in the export cell are the
same idea with different bits per weight.

## 1.5 QLoRA — the two tricks at once

**QLoRA** is what this repo trains: keep the frozen base model *quantized* (4-bit
NF4) and train the LoRA sticky note at normal precision. During the maths the
4-bit weights are turned back into decimals on the fly, and only the note gets
gradients.

The payoff for a 3-billion-weight model: about 6 GB of weights at 16 bits becomes
about 1.8 GB at 4 bits, which is the difference between "needs a big GPU" and
"fits on one consumer GPU". Two small extras in the paper: the per-block scales
are themselves stored in 8 bits (*double quantization*), and optimizer memory can
spill to CPU RAM instead of crashing (*paged optimizers*).

Quantization saves memory, not maths — the additions and multiplications still
happen at roughly full precision.

## 1.6 Two LoRA variants worth knowing

Both are switched off in this repo (`use_rslora=False`, `loftq_config=None`), so
think of them as upgrades you can try:

- **rsLoRA** changes the volume knob from $\alpha/r$ to $\alpha/\sqrt r$. With
  plain LoRA, a bigger rank makes the correction quieter and training less stable;
  rsLoRA keeps big ranks usable.
- **LoftQ** helps the quantized model and the sticky note fit together better
  instead of quantizing first and patching later. Useful at aggressive 4-bit or
  2-bit settings.

---

# Part 2 — Setup

Python 3.10+ and an NVIDIA GPU with CUDA are assumed: the notebooks call
`.to("cuda")` and `torch.cuda.*` directly.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; .venv/bin/activate on Linux
python -m pip install -r requirements-crash-course.txt
```

For a CUDA build of llama.cpp, reinstall that one package with flags:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" python -m pip install --upgrade --force-reinstall llama-cpp-python
```

Then:

```bash
python -m jupyter lab
```

Model downloads come from the Hugging Face Hub; run `huggingface-cli login` if
you hit rate limits.

---

# Part 3 — Fine-tuning (theory → code)

This maps §1.2, §1.3 and §1.5 onto the notebook cells.

**Load the base model in 4-bit** — the "Q" of QLoRA (§1.4, §1.5):

```python
from unsloth import FastLanguageModel

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,          # auto: fp16 / bf16
    load_in_4bit=True,
)
```

**Attach LoRA adapters** — the equation from §1.2, in code:

| Setting | Value | Theory link |
| --- | --- | --- |
| `r` | 16 | rank of $A, B$ |
| `lora_alpha` | 16 | the $\alpha$ in $\alpha/r$ |
| `lora_dropout` | 0 | 0 is the optimized path |
| `bias` | `"none"` | no bias terms trained |
| `target_modules` | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` | which $W_0$ get a $\Delta W$ |
| `use_gradient_checkpointing` | `"unsloth"` | recompute activations |
| `use_rslora` | `False` | §1.6 |
| `loftq_config` | `None` | §1.6 |
| `random_state` | 3407 | seed |

**Format the data** (§1.3): each row becomes
`{"prompt": [...], "completion": [...]}`, and `SFTTrainer` masks the prompt.

**Train.** `trl.SFTConfig` under `SFTTrainer`. Hyperparameters:

| Parameter | Value |
| --- | --- |
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 (effective batch 8) |
| `max_steps` | 60 |
| `learning_rate` | 2e-4 |
| `lr_scheduler_type` | linear, `warmup_steps=5` |
| `optim` | `adamw_8bit` |
| `weight_decay` | 0.01 |
| `fp16` / `bf16` | via `is_bfloat16_supported()` |
| `seed` | 3407 |
| `packing` | `False` |
| `report_to` | `"none"` |

60 steps is a demo, not a real run: `outputs/checkpoint-60/trainer_state.json`
shows `global_step: 60`, `num_train_epochs: 1`, and a final logged loss of about
`0.43`. For a genuine run, swap `max_steps` for `num_train_epochs=1..3`
(Unsloth's own guidance: 1–3 passes, more overfits).

Each notebook then frees VRAM and prints peak usage, and runs a streaming sanity
prompt (*"How many 'r's are present in 'strawberry'?"*).

---

# Part 4 — Quantization theory in code

`quantization.ipynb` (§1.4) is CPU-only and does three things:

1. Quantizes to int8 symmetrically — compute a scale, round, clip, then
   dequantize with `x_hat = q*scale`.
2. Plots original vs recovered values so you can see the half-step error.
3. Repeats it with a 16-entry NF4 table, giving the NF4 staircase.

Run it first if the difference between 8-bit, 4-bit and NF4 is not yet obvious —
it is the only place where quantization appears without a model around it.

---

# Part 5 — Export formats

Three artifacts, three use cases.

**LoRA adapter** (small, pairs with the base model):

```python
model.save_pretrained("service-001-3B")
tokenizer.save_pretrained("service-001-3B")
```

**Merged 16-bit model** (§1.2: the sticky note added into the weights for good):
standalone safetensors, and the source for GGUF conversion.

```python
model.save_pretrained_merged("service-001-3B-merged", tokenizer, save_method="merged_16bit")
```

**GGUF with block quantization** (§1.4): `save_pretrained_gguf` fuses the model,
runs llama.cpp's HF→GGUF converter and then its quantizer:

```python
model.save_pretrained_gguf("service-001-3B-GGUF", tokenizer)              # Q8_0 default
model.save_pretrained_gguf("service-001-3B-GGUF-Q4_K_M", tokenizer,
                           quantization_method="q4_k_m")                   # smaller
model.save_pretrained_gguf("service-001-3B-GGUF-MULTI", tokenizer,
                           quantization_method=["f16", "q4_k_m", "q5_k_m", "q8_0"])
```

The notebook leaves only the Q8_0 export enabled; the `if False:` blocks are the
variants. The `.gguf` lands in `service-001-3B-GGUF_gguf/` alongside the
generated `Modelfile` (Llama-3 template, `temperature 1.5`, `min_p 0.1`, Llama
stop tokens). `tokenizer._ollama_modelfile` is also written to
`service-001-3B/ollama_modelfile`.

---

# Part 6 — Inference on quantized weights

## Merged 16-bit model, via Unsloth

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="service-001-3B-merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,        # the merged export is 16-bit
)
FastLanguageModel.for_inference(model)   # Unsloth's ~2x faster inference path
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

messages = [{"role": "user", "content": "How many 'r's are present in 'strawberry'?"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

text_streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=False
)

outputs = model.generate(
    input_ids=inputs, streamer=text_streamer, max_new_tokens=1024,
    temperature=0.7, top_p=0.9, min_p=0.1, do_sample=True, use_cache=True,
)
response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=False)[0]
```

Only generated tokens are decoded (`outputs[:, inputs.shape[1]:]`), so the prompt
does not reappear in the answer. Sampling is `temperature` (flatten the
distribution), `top_p` (nucleus truncation), `min_p` (drop tokens below a
probability relative to the best one) — the same trio the `Modelfile` uses.

## Quantized GGUF, via Unsloth

That section reloads `service-001-3B-GGUF` with `load_in_4bit=True`, re-applies
the llama-3.1 template and generates with the same settings. Worth knowing: this
path loads the **merged safetensors** Unsloth left in the export directory, not
the `.gguf` — the actual quantized file lives in `service-001-3B-GGUF_gguf/`.
The GGUF-specific path here is llama.cpp.

## Quantized GGUF, via llama.cpp (Python)

`llama-cpp-python` loads the `.gguf` and runs the forward pass in C++, no
PyTorch involved:

```python
from llama_cpp import Llama

model = Llama(
    model_path="service-001-3B-GGUF_gguf/Llama-3.2-3B-Instruct.Q8_0.gguf",
    n_ctx=4096,        # context window
    n_gpu_layers=-1,   # offload all layers to GPU; 0 for CPU-only
    n_batch=512,       # prompt-processing batch size
    verbose=False,
)

stream = model.create_chat_completion(
    messages=[{"role": "user", "content": "I have 5 mangoes and 3 apples..."}],
    max_tokens=1024, temperature=0.7, top_p=0.9, stream=True,
)
for chunk in stream:
    print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

`create_chat_completion` applies the chat template stored in the GGUF metadata,
so you pass OpenAI-style `messages` and stream the result.

## Quantized GGUF, via the llama.cpp CLI

```bash
llama-cli -m service-001-3B-GGUF_gguf/Llama-3.2-3B-Instruct.Q8_0.gguf \
  -cnv -ngl 99 -c 4096 -n 1024 --temp 0.7 --top-p 0.9
```

`-cnv` is chat mode (template from the GGUF), `-ngl 99` offloads all layers, `-c`
sets context, `-n` the token budget. Drop `-ngl` for CPU-only.

## Ollama (optional)

Copy the `.gguf` next to its `Modelfile` and build:

```bash
ollama create service-001-3b -f service-001-3B-GGUF_gguf/Modelfile
ollama run service-001-3b "How many 'r's are present in 'strawberry'?"
```

The `Modelfile` starts with `FROM Llama-3.2-3B-Instruct.Q8_0.gguf`, so run this
from the directory containing the model, or make that line an absolute path.

---

# Part 7 — Suggested learning path

1. `quantization.ipynb` — see a step size, a codebook and the half-step error
   with your own eyes, no GPU needed (§1.4).
2. `unsloth_finetuned_hanif.ipynb`, model-loading and `get_peft_model` cells —
   read them next to §1.2 and count the sticky-note numbers yourself.
3. The trainer cell — match each `SFTConfig` field to §1.3 and the table above.
4. Inference on the freshly trained adapter — same prompt template as training.
5. Export section — adapter → merged → GGUF, the three artifacts of §1.2 and §1.4.
6. llama.cpp section — the same Q8_0 file, quantized blocks and a completely
   different runtime, to feel why bits per weight matter (§1.4).
7. Optional: flip `use_rslora=True` or `loftq_config={"loftq_bits": 4}` and
   compare the 60-step loss curve (§1.6).

## Notes

- Every generation cell assumes CUDA. On CPU, use `n_gpu_layers=0` for llama.cpp
  and expect very slow PyTorch inference.
- The GGUF export needs a llama.cpp toolchain; Unsloth fetches a prebuilt copy on
  first use, so the first `save_pretrained_gguf` call is slow.
- 60 steps on a public reasoning dataset produces a demo quality model, not a
  reliable reasoner.
