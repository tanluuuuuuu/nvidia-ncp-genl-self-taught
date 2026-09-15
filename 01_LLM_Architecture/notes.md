# LLM Architecture

Study notes for NVIDIA GenAI / LLM certification objectives (encoder–decoder, transformers, embeddings, sampling).

**Core intuition:** Architecture choice = which tokens can see which other tokens (the attention mask) + what the model is trained to do (MLM vs next-token vs seq2seq).

---

## 1.1 Analyze encoder-decoder structures and their applications.

### Three body plans

| Family | Attention | Training objective | Strengths | Examples |
| --- | --- | --- | --- | --- |
| **Encoder-only** | Bidirectional (full self-attention) | Masked LM (fill-in-the-blank), often NSP | Understanding: classify, NLI, NER, embeddings | BERT, RoBERTa, DistilBERT |
| **Decoder-only** | Causal / autoregressive (left-to-right) | Next-token prediction | Generation: chat, code, completion | GPT, LLaMA, Mistral, Claude-style |
| **Encoder–decoder** | Bidirectional encoder + causal decoder + **cross-attention** | Seq2seq / text-to-text | Transform input → output: translate, summarize, rewrite | Original Transformer, T5, BART |

### Encoder–decoder (original Transformer)

1. **Encoder** maps input tokens $x_1,\ldots,x_n$ → continuous representations $z_1,\ldots,z_n$.
2. **Decoder** generates $y_1,\ldots,y_m$ one token at a time (autoregressive).
3. Each decoder layer has three sublayers:
   - **Masked self-attention** (cannot see future output tokens)
   - **Cross-attention** (queries from decoder; keys/values from encoder $z$)
   - **Feed-forward** network

**Cross-attention** is the key difference vs decoder-only: every generated token can look at the *entire* source sequence.

### When to choose which

- **Need deep understanding of a fixed text** (sentiment, search ranking, clustering) → encoder-only or a dedicated embedding model.
- **Open-ended generation / chat / agents** → decoder-only (dominant modern LLM design).
- **Clear input–output transform** (translation, summarization, structured rewrite) → encoder–decoder (T5/BART) still strong; many teams also use decoder-only with prompting.

### Exam traps

- BERT ≠ generative chatbot architecture (encoder-only).
- GPT cannot freely “see the future” during training/generation (causal mask).
- T5 frames *all* NLP as text-to-text (e.g. `"summarize: ..."` → summary).

### Further reading

- [Attention Is All You Need](https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Encoder vs Decoder Models (AI/TLDR)](https://ai-tldr.dev/learn/llm-fundamentals/transformers-and-attention/encoder-vs-decoder-models/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 1.2 Describe transformer architectures including self-attention mechanisms.

### Why transformers replaced RNNs

- Parallel over sequence length (no left-to-right recurrence bottleneck in the encoder).
- Direct paths between distant tokens (better long-range dependency modeling).
- Cost: self-attention is $O(n^2)$ in sequence length $n$.

### Building block (each layer)

1. Multi-head attention (+ residual + layer norm)
2. Position-wise FFN / MLP (+ residual + layer norm)

(Modern variants may use pre-norm vs post-norm; idea is the same.)

### Scaled dot-product attention

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

| Symbol | Role (memorize) |
| --- | --- |
| **Q (Query)** | “What am I looking for?” — retrieves |
| **K (Key)** | “What do I contain / match on?” — indexing |
| **V (Value)** | “What content do I pass?” — payload |

- Divide by $\sqrt{d_k}$ so dot products don’t explode → softmax stays informative (avoids tiny gradients).
- **Self-attention:** Q, K, V all come from the same sequence. (decoder-only)
- **Cross-attention:** Q from decoder; K, V from encoder. (encoder-decoder)

### Multi-head attention (MHA)

Run several attentions in parallel (different projections), then concatenate and project. Different heads can specialize (syntax vs long-range vs local).

### Inference-oriented attention variants (NCP-relevant)

Autoregressive decode is often **memory-bandwidth bound**, not FLOP-bound. The bottleneck is repeatedly reading the **KV cache** (all past keys/values) for every new token. Shrinking K/V is the point of MQA/GQA.

#### Why the KV cache dominates

When generating token-by-token, you **recompute Q only for the new token**, but you must **reload every past K and V**. So the growing pile of stored K/V (the KV cache) is what hurts.

**What you store:**

1. **`seq_len`** — how many tokens are in context so far  
2. **`num_layers`** — every transformer layer keeps its own cache  
3. **`num_kv_heads`** — how many key/value heads (this is what MQA/GQA reduce)  
4. **`head_dim x2`** — length of both K or V vector (e.g. 128)  
5. **`bytes_per_number`** — e.g. FP16 = 2 bytes  

$$
\text{KV memory} \approx \texttt{seq\_len} \times \texttt{num\_layers} \times \texttt{num\_kv\_heads} \times \texttt{head\_dim} \times 2 \times \texttt{bytes\_per\_number}
$$

**Tiny numeric example** (one request, FP16):

| Factor | Value |
| --- | --- |
| `seq_len` | 4096 |
| `num_layers` | 32 |
| `num_kv_heads` | 32 (full MHA) |
| `head_dim` | 128 |
| `2` (K and V) | 2 |
| `bytes_per_number` | 2 |

$$
4096 \times 32 \times 32 \times 128 \times 2 \times 2 \approx 2.1\ \text{GB}
$$

That is why MQA/GQA exist: smaller `num_kv_heads` → less cache to store and to read every decode step → faster generation and longer context before out-of-memory (OOM).

#### MHA (baseline)

- $h$ query heads, $h$ key heads, $h$ value heads ($n_q = n_{kv} = h$).
- Each head has its own $W_Q, W_K, W_V$ projections.
- Best modeling flexibility; **largest** KV cache ($n_{kv} = h$).

#### MQA — Multi-Query Attention

- Keep **many query heads** ($n_q = h$), but only **one** shared K and one shared V across all heads ($n_{kv} = 1$).
- Each query head still has a distinct $q$, so heads can look for different things; they all attend over the **same** cached keys/values.
- KV cache shrinks by roughly **$h\times$** (often cited ~8–16× for typical head counts) vs MHA.
- Trade-off: slightly weaker quality / training quirks vs full MHA; huge **decode** speed/memory win. (Prefill still compute-heavy; MQA helps most in the decode phase.)

```text
MHA:  Q1 Q2 Q3 Q4    K1 K2 K3 K4    V1 V2 V3 V4
MQA:  Q1 Q2 Q3 Q4    K_shared       V_shared
```

#### GQA — Grouped-Query Attention

- Compromise: partition $h$ query heads into $g$ **groups**; each group shares one K/V ($n_{kv} = g$, with $1 < g < h$).
- Usually $h$ divisible by $g$ (e.g. 32 query heads, 8 KV heads → 4 queries per KV).
- Cache size $\propto g$ instead of $h$ → about $h/g$× smaller than MHA (e.g. 4× if $g = h/4$).
- Quality closer to MHA than pure MQA; still much cheaper at inference. Used in **Llama 2/3**-class models and many production LLMs.

```text
GQA (example g=2, h=4):
  Q1 Q2  →  K1 V1
  Q3 Q4  →  K2 V2
```

#### Comparison

| Variant | $n_q$ | $n_{kv}$ | KV-cache vs MHA | Quality | Typical use |
| --- | --- | --- | --- | --- | --- |
| **MHA** | $h$ | $h$ | 1× (baseline) | Best | Small models / training baseline |
| **GQA** | $h$ | $g$ (e.g. 8) | $\sim h/g$× smaller | Near MHA | Production LLMs (Llama family) |
| **MQA** | $h$ | $1$ | $\sim h$× smaller | Good enough / slightly lower | Extreme decode efficiency |

**Exam hooks**

- MQA/GQA exist for **KV-cache efficiency at inference**, not for changing the attention formula itself.
- MQA = all queries share one K/V; GQA = queries share K/V **per group**.
- Expect “8–16× less KV memory” style claims for MQA vs MHA when $h$ is large.
- TensorRT-LLM / NIM stacks care about this because paged KV cache + GQA/MQA = longer context and higher throughput.

### Positional information

Transformers have **no inherent order**; self-attention is a set operation. Without position, “dog bites man” ≈ “man bites dog”. Position must be injected somehow.

| Method | Idea | Extrapolation beyond train length |
| --- | --- | --- |
| Absolute sinusoidal (original paper) | Fixed sin/cos of position | Poor |
| Learned absolute | Embed position id | Poor / fixed max length |
| **RoPE** | Rotate Q/K by position angle | Good |
| **ALiBi** | Add distance bias to attention logits | Good |

Absolute methods give each index a hard label (`pos=17`). Modern LLMs prefer **relative** distance (“how far is token *i* from *j*?”). **RoPE** and **ALiBi** are the two exam-relevant ways to do that.

### Layer norm & residuals

Stabilize deep stacks; residuals keep gradients flowing. Autoregressive decoding still generates **one token at a time**, often with a **KV cache** so past keys/values are not recomputed.

### Further reading

- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- Attention formula / Q vs V roles appear often on NVIDIA GENL exams.

---

## 1.3 Develop code to extract embeddings from both encoder and decoder models.

### What “embedding” means in practice

1. **Token embedding table** — lookup of token id → vector (input side).
2. **Contextual hidden states** — after transformer layers; each token’s vector depends on context. This is usually what people mean for similarity / RAG / clustering.
3. **Pooled sentence vector** — collapse sequence → one vector (mean pool, CLS, last token, etc.).

### Encoder model (e.g. BERT) — Hugging Face

```python
import torch
from transformers import AutoTokenizer, AutoModel

name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name)
model.eval()

texts = ["NVIDIA GPUs accelerate LLMs.", "Transformers use self-attention."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    out = model(**inputs)

# Token-level contextual embeddings: (batch, seq_len, hidden)
token_emb = out.last_hidden_state

# Mean-pool with attention mask (usually better than pooler_output for semantics)
mask = inputs["attention_mask"].unsqueeze(-1).float()
sent_emb = (token_emb * mask).sum(dim=1) / mask.sum(dim=1)

# Optional: CLS / pooler (BERT-specific; docs warn mean-pool is often better)
cls = token_emb[:, 0, :]
pooled = out.pooler_output  # CLS → linear + tanh (NSP-trained)
```

Tips:

- Prefer `AutoModel`, not `AutoModelForSequenceClassification`, when you only need representations.
- `pooler_output` ≠ raw CLS; it’s an extra projection trained for next-sentence prediction.

### Decoder / Causal LM (e.g. GPT-2, LLaMA) — hidden states

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(name)
model.eval()

inputs = tokenizer(
    ["Decode left to right.", "Causal masks hide the future."],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)

# Last layer hidden states: (batch, seq_len, hidden)
token_emb = out.hidden_states[-1]

# Common pooling choices for decoder-only:
# 1) mean over non-pad tokens
mask = inputs["attention_mask"].unsqueeze(-1).float()
mean_emb = (token_emb * mask).sum(1) / mask.sum(1)

# 2) last non-pad token (aligned with next-token training)
lengths = inputs["attention_mask"].sum(1) - 1
last_emb = token_emb[torch.arange(token_emb.size(0)), lengths]
```

Notes:

- Causal LMs often expose embeddings via `output_hidden_states=True` rather than a dedicated `last_hidden_state` on the CausalLM head wrapper—check the model’s output type.
- For production retrieval, specialized embedding models (e.g. E5, GTE, NVIDIA NeMo Retriever / NV-Embed-style) usually beat raw BERT/GPT pooling.

### Encoder–decoder (e.g. T5)

- **Encoder side:** `model.encoder(**encoder_inputs).last_hidden_state` — good source embeddings.
- **Decoder side:** contextual states conditioned on encoder via cross-attention; less common as generic embeddings.

### Sanity checks

- Shape: `(batch, seq, hidden)` for tokens; `(batch, hidden)` after pooling.
- Cosine similarity on L2-normalized vectors is the usual similarity metric.
- Same text → same embedding only if no dropout / deterministic mode (`model.eval()`, `torch.no_grad()`).

---

## 1.4 Implement advanced sampling techniques for text generation.

### Pipeline (decoder-only)

1. Forward → logits for next token (vocab size $V$).
2. Optionally reshape distribution (temperature, top-k, top-p, penalties).
3. Either **argmax** (greedy) or **sample** from the filtered distribution.
4. Append token; repeat until EOS / max length. Use KV cache for speed.

### Hugging Face `generate` patterns

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tok("The future of AI is", return_tensors="pt")

# Greedy (deterministic)
greedy = model.generate(**inputs, max_new_tokens=40, do_sample=False)

# Multinomial sampling + temperature + nucleus + top-k
sampled = model.generate(
    **inputs,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
)

# Beam search (good for grounded tasks: translation / short factual)
beamed = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=4,
    early_stopping=True,
)

print(tok.decode(sampled[0], skip_special_tokens=True))
```

### Technique cheat sheet

| Technique | What it does | Typical use |
| --- | --- | --- |
| **Greedy** | Always pick $\arg\max$ | Deterministic, often bland / repetitive |
| **Temperature** | Softmax($logits / T$); $T\!<\!1$ sharper, $T\!>\!1$ flatter | Creativity vs precision knob |
| **Top-k** | Keep only $k$ highest-prob tokens, renorm, sample | Cap wild low-prob tokens |
| **Top-p (nucleus)** | Smallest set with cumulative mass $\ge p$, renorm, sample | Adaptive vocabulary size |
| **Beam search** | Keep top-$B$ partial sequences | Translation, short constrained text |
| **Repetition penalty** | Down-weight already-used tokens | Reduce loops |
| **Min-p / typical** | Newer filters (HF supports several) | Alternative diversity control |

You can combine **top-k + top-p + temperature** (common in practice).

### From-scratch nucleus sketch (conceptual)

```python
import torch
import torch.nn.functional as F

def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum - sorted_probs > top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, choice)
    return torch.multinomial(probs, 1)
```

### Further reading

- [HF: How to generate text](https://huggingface.co/blog/how-to-generate)
- [HF: Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

---

## 1.5 Understand output sampling techniques used in decoder-based language models.

### Why sampling exists

The model outputs a **probability distribution** over the vocabulary, not “the answer.” Decoding strategy chooses how to turn that distribution into a token stream.

### Greedy vs sampling vs beams

- **Greedy:** high precision for short, constrained tasks; fails on creative / long chat (repetition, dullness).
- **Sampling:** diversity; needs filters (temp / top-k / top-p) or quality collapses into nonsense.
- **Beam search:** optimizes sequence-level likelihood; strong on translation/ASR-like tasks; can be generic or repetitive for open chat. Often **not** preferred for modern chat LLMs.

### Temperature intuition

- $T \to 0$: approaches greedy.
- Low $T$ (e.g. 0.2–0.5): focused, “thinking,” tool-ish answers.
- High $T$ (e.g. 0.8–1.2): creative writing; higher hallucination risk.

### Top-k vs top-p

- **Top-k:** fixed shortlist size (e.g. 50). Simple; can be too narrow when the distribution is flat, or still include junk when peaked.
- **Top-p:** dynamic shortlist — keep tokens until cumulative probability hits $p$ (e.g. 0.9). Adapts to peaked vs flat distributions (“nucleus”).

### Other controls you’ll see in production

- **`max_new_tokens` / length limits** — stop runaway generation.
- **Stop sequences / EOS** — chat templates.
- **Frequency / presence / repetition penalties** — reduce loops.
- **Logit bias / constrained decoding** — force JSON, grammar, banned words.
- **Seed** — reproducibility when sampling (still hardware-dependent sometimes).

### Exam-style associations

- Higher temperature → more diversity / less determinism.
- Beam search → multiple candidate paths → often better for translation/summarization-style tasks.
- Autoregressive = each new token conditions on previous outputs.

---

## 1.6 Understand the concept of embeddings.

### Definition

An **embedding** is a dense vector representation of a discrete object (token, word, sentence, document, image patch, …) such that **geometry encodes meaning**: similar items lie closer together (often measured by cosine similarity).

Classic word2vec intuition:  
`king - man + woman ≈ queen` — arithmetic in vector space reflects semantic relations.

### Levels of embeddings in LLMs

1. **Static token/word embeddings** — one vector per vocabulary id (Word2Vec, GloVe, or the model’s input embedding matrix). Context-independent.
2. **Contextual embeddings** — transformer hidden states; the vector for “bank” differs in “river bank” vs “bank account.”
3. **Sentence / document embeddings** — pooled or specially trained vectors for retrieval, clustering, classification.
4. **Positional embeddings / RoPE** — encode order, not meaning.

### Why they matter

- Turn text into numbers neural nets can compute on.
- Enable **similarity search** and **RAG** (embed docs + query; retrieve nearest neighbors).
- Transfer learning: pretrained embeddings / encoders bootstrap downstream tasks.

### Training signals that shape embeddings

| Objective | Effect |
| --- | --- |
| Masked LM (BERT) | Bidirectional context in token states |
| Next-token (GPT) | States useful for predicting the future |
| Contrastive (SimCSE, E5, etc.) | Pull similar texts together, push apart negatives — best for retrieval |
| Word2Vec / FastText | Local co-occurrence windows |

### Practical metrics

- **Cosine similarity:** $\cos(\theta) = \frac{a\cdot b}{\|a\|\|b\|}$ — direction matters more than magnitude.
- **Dot product** — used when vectors are already normalized or in MIPS indexes.
- Downstream: classification accuracy, retrieval Recall@k, clustering quality.

### Relation to architecture (tie-back)

- Encoder-only models are historically popular as embedding backbones.
- Decoder-only models can also embed (mean / last token / dedicated instruct-embedding fine-tunes).
- NVIDIA stack angle: vectorize corpora with embedding models, retrieve with **NeMo Retriever**-class systems, then generate with an LLM (RAG) to reduce hallucinations.

### Mental model for the exam

> Tokens → embedding lookup → (+ position) → stacked attention/FFN → contextual vectors → (optional pool) → similarity / classifier / next-token logits.

---

## Quick self-check

1. Name one task best for BERT, one for GPT, one for T5 — and why (attention mask + objective).
2. Write the attention formula and explain $\sqrt{d_k}$, Q vs V.
3. Extract a sentence embedding from BERT with masked mean-pooling.
4. Explain when you’d pick temperature 0.2 vs 1.0, and top-k vs top-p.
5. Contrast static vs contextual embeddings; when is cosine similarity appropriate?

## High-signal references

- Vaswani et al., *Attention Is All You Need* (2017)
- [HF blog: How to generate](https://huggingface.co/blog/how-to-generate)
- [WellWells: NCA-GENL transformer notes](https://wellstsai.com/en/post/nvidia-nca-genl-exam-study-guide/)
- [CertCompanion: NCP-GENL LLM Architecture domain](https://certcompanion.com/blog/nvidia-generative-ai-llms-ncp-genl-exam-guide)
- Official study guide PDF (links to Illustrated Transformer, HF MLM / LM / perplexity docs)
