# Prompt Engineering

Study notes for NVIDIA GenAI / LLM certification objectives (prompting strategies, in-context learning, causal LM training, wrappers & constrained decoding).

**Core intuition:** Prefer the *simplest* adaptation that works — prompt → few-shot / CoT → soft prompt / PEFT → full fine-tune / RAG. Exam scenarios often ask *which* technique fits the business need, not just definitions.

**Exam weight (NCP-GENL):** Prompt Engineering ≈ **13%** — CoT, zero/one/few-shot, domain adaptation, output control, ReAct, guardrails.

---

## 2.1 Engineer effective prompts and templates, including chain-of-thought and prompt learning for small datasets or specialized domains.

### Prompt anatomy (production template)

Treat prompts as versioned software artifacts: model + temperature + schema + eval set + owner.

| Section | Purpose |
| --- | --- |
| **Role / system** | Who the model is; non-negotiable rules |
| **Task** | What to do in one clear instruction |
| **Context** | Domain facts, retrieved docs, user state |
| **Constraints** | Length, tone, banned topics, safety |
| **Output format** | JSON schema, bullet list, tags (`Final answer:`) |
| **Examples** (optional) | Few-shot demos of I/O style |
| **Reasoning cue** (optional) | CoT / procedure when multi-step |

**Good defaults**

- Be specific; prefer numbered procedures over vague “be helpful.”
- Put critical rules early and repeat format requirements near the end.
- Separate untrusted user text (fence it) from trusted instructions to reduce injection risk.
- Low temperature for extraction / classification; higher only when diversity is desired.
- Version prompts in git; A/B on a fixed golden set when the model changes.

### Chain-of-Thought (CoT)

**What:** Ask the model to emit intermediate reasoning before the final answer (Wei et al., 2022).

**When (exam trap):** Complex **multi-step reasoning** (math, logic, planning).  
**Not** primarily for “make the answer look like this format” → that is few-shot.

Patterns:

1. **Zero-shot CoT:** append `Let's think step by step.`
2. **Few-shot CoT:** show worked examples with reasoning + answer.
3. **Structured CoT:** prescribe steps (`1) extract facts 2) compute 3) verify 4) Final answer:`).

```text
You are a careful analyst. Solve the problem with numbered steps.
Put the final numeric answer after the line "Final answer:".

Problem: A store has 48 apples. It sells 15, then receives 9. How many remain?
```

**Trade-offs:** More tokens / latency / cost; can leak chain-of-thought to users (hide or summarize in UX). Modern “reasoning” models may already think internally — still useful to request *structured* intermediate fields for auditability.

### Related reasoning patterns

| Pattern | Idea | Use when |
| --- | --- | --- |
| **Self-consistency** | Sample several CoT paths; majority vote | Hard reasoning; budget allows |
| **Tree of Thoughts (ToT)** | Explore / evaluate multiple branches | Hard search/planning; expensive |
| **ReAct** | Loop Thought → Action → Observation with tools | Needs search, calculator, APIs |
| **Prompt chaining** | Split into sequential specialized prompts | Pipelines (extract → reason → format) |

### Domain / small-data adaptation ladder

| Approach | Data needed | Weight updates? | Best for |
| --- | --- | --- | --- |
| Hard prompt + glossary in context | None–tiny | No | Quick domain vocab |
| Few-shot ICL | 1–10 labeled examples | No | Format / style |
| **Prompt learning (soft prompts)** | Small labeled set | Tiny (virtual tokens) | Specialized domain, freeze base model |
| LoRA / adapters | Small–medium | Small | Stronger domain fit |
| Full SFT | Larger curated set | All (or most) | Persistent behavior change |

### Prompt learning (soft prompts) — PEFT family

“Prompt learning” ≠ typing a better English sentence. It means **learn continuous prompt vectors** while freezing (most of) the LLM:

| Method | Where trainable vectors live |
| --- | --- |
| **Prompt tuning** | Prepended to **input embeddings** only |
| **Prefix tuning** | Prefixed into **every layer’s** K/V (more expressive; good for NLG) |
| **P-tuning** | Soft prompts + small prompt encoder MLP |

Why for small / specialized data:

- Stores one tiny task module per domain instead of a full model copy.
- Often competitive with full fine-tuning as model size grows.
- Keeps the foundation model reusable across tasks.

Hugging Face PEFT documents these under soft-prompt methods; NeMo also supports parameter-efficient adaptation for domain tasks.

### Decision heuristic (memorize)

1. Clear single-step task → **zero-shot** (+ schema).
2. Need consistent format/style → **few-shot**.
3. Need multi-step reasoning → **CoT** (or ReAct if tools).
4. Need persistent domain behavior / can’t fit examples in context → **prompt tuning / LoRA / SFT**.
5. Need fresh external facts → **RAG** (not more CoT alone).

### Exam traps

- **CoT ≠ few-shot.** CoT = reasoning; few-shot = demos for format/behavior.
- **Triton does not do prompt engineering.** Prompting is application-layer (NeMo Guardrails / your app / LangChain).
- Exhaust prompting before jumping to full fine-tuning when possible.

---

## 2.2 Employ zero-shot, one-shot, and few-shot techniques to expand model adaptability.

### In-context learning (ICL)

The model adapts from examples **inside the prompt**, with **no weight updates**. Quality depends on pretrained knowledge + example design.

| Mode | Examples in prompt | Typical use |
| --- | --- | --- |
| **Zero-shot** | 0 | Clear instructions; capable instruction-tuned models |
| **One-shot** | 1 | Show format once when zero-shot is unstable |
| **Few-shot** | 2–k (often 3–8) | Classification labels, extraction schema, tone |

Also called **k-shot** prompting.

### Zero-shot template

```text
System: You classify support tickets. Labels: billing | technical | other.
User: "My invoice was double-charged last month."
Assistant: billing
```

(In true zero-shot the assistant line is empty — you only provide instruction + query.)

### Few-shot design rules

- **Quality > quantity.** Beyond ~5–8 examples, gains flatten and cost rises.
- Cover **diversity + edge cases** (and common failure modes), not only happy paths.
- Keep label space / schema **identical** across examples.
- Order can matter; shuffle or fix deliberately; put strongest / closest examples last if needed.
- For classification, balance classes in the demo set.
- Don’t put the test answer in the examples by mistake.

### Few-shot CoT vs plain few-shot

```text
# Plain few-shot → teaches FORMAT
Input: "Great product, slow shipping." → Sentiment: mixed

# Few-shot CoT → teaches REASONING + answer
Input: ...
Reasoning: Praise for product (+) but complaint about shipping (−) → mixed.
Answer: mixed
```

### When ICL is not enough

- Examples don’t fit in context window.
- Need lower latency / cost at high QPS (long prompts are expensive).
- Behavior must stick without sending demos every request → **fine-tune / prompt-tune**.
- Facts change constantly → **RAG**.

### Prompt vs fine-tune vs RAG (cross-domain favorite)

| Need | Prefer |
| --- | --- |
| Quick behavior / format tweak | Prompting (zero/few-shot) |
| Ground answers in private / fresh docs | RAG |
| Permanent style, jargon, or task skill | Fine-tune (LoRA/SFT); mix ~5–10% general data to limit catastrophic forgetting |

---

## 2.3 Train decoder-based LLMs with causal language modeling as needed.

### Causal language modeling (CLM)

Decoder-only LLMs are trained with **next-token prediction** under a **causal mask** (token $t$ may only attend to $\le t$):

$$
\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t})
$$

Same objective appears in:

1. **Pretraining** — next token on huge corpora.
2. **Continued pretraining / domain adaptation** — more CLM on domain text (no special “answer” format required).
3. **Supervised fine-tuning (SFT)** — pack `prompt + response` (often with chat template); loss usually on **response tokens** (prompt tokens masked out of the loss).
4. **Instruction tuning** — many task prompts → desired completions (still CLM under the hood).

### Why this belongs next to prompting

Prompting steers a **frozen** CLM at inference. When prompts / ICL fail, you **train** the same CLM objective on curated data so desired behavior is absorbed into weights (or into soft prompts / LoRA).

### Minimal SFT sketch (Hugging Face style)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "gpt2"  # replace with your decoder LLM
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def format_example(ex):
    # Chat / instruction format — adjust to your template
    text = f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['output']}{tok.eos_token}"
    return tok(text, truncation=True, max_length=512)

# ds = load_dataset(...); tokenized = ds.map(format_example)
# Use DataCollatorForLanguageModeling(mlm=False) for causal LM
# Trainer(...).train()
```

Notes for certification:

- **`mlm=False`** → causal LM (not BERT-style masked LM).
- Chat models need the **correct chat template** (`apply_chat_template`) so train/serve formats match.
- For efficiency on small GPUs: **LoRA / QLoRA** (NeMo Customizer is the NVIDIA-native path).
- Soft **prompt / prefix tuning** is CLM with almost all weights frozen — middle ground between pure prompting and full SFT.

### When to train vs only prompt

Train (CLM / PEFT) when you need:

- Stable domain jargon or long policies that blow the context.
- Lower per-request token cost (shorter prompts).
- Behavior that zero/few-shot cannot lock in.

Still prefer RAG for **knowledge** that must stay up to date.

---

## 2.4 Design specialized LLM-wrapping modules with built-in validation and constrained decoding for improved consistency, reduced hallucinations, and better user experience.

### Why wrap the LLM

Raw `generate()` is unreliable for products. A **wrapper module** (application layer) should:

1. Build / version the prompt.
2. Call the model with controlled decoding params.
3. **Validate** structure and policy.
4. Retry, repair, or refuse safely.
5. Log prompt version, latency, validation failures.

Prompting alone does not guarantee JSON, citations, or safety.

### Layered control (remember the stack)

| Layer | What it enforces | Examples |
| --- | --- | --- |
| **Prompt / schema in text** | Soft preference | “Reply with JSON matching …” |
| **Constrained decoding** | Token-level syntax | JSON schema, regex, CFG, choice set |
| **Post-validators** | Types, ranges, business rules | Pydantic / zod; cite-or-refuse checks |
| **Guardrails / policy** | Safety, topics, PII | NeMo Guardrails rails |
| **Grounding** | Factuality vs corpus | RAG + “answer only from context” |

**Critical:** Constrained decoding → **syntactic** validity. It does **not** guarantee truth. You still need grounding + semantic checks.

### Constrained decoding techniques

- **JSON mode** — valid JSON syntax (not necessarily your schema).
- **JSON Schema / grammar-guided decoding** — mask illegal next tokens (Outlines, Guidance, XGrammar, lm-format-enforcer, llama.cpp GBNF, vLLM structured outputs).
- **Choice / enum** — force one of `{yes,no,refund,...}`.
- **Regex** — emails, IDs, dates.
- **Tool / function calling** — provider-enforced argument schemas (excellent UX for agents).

Conceptual mask:

```text
at each step: allowed = grammar.next_tokens(prefix)
logits[~allowed] = -∞
sample / argmax
```

### Wrapper pattern (validate → retry)

```python
from pydantic import BaseModel, ValidationError, Field
import json

class TicketResult(BaseModel):
    label: str = Field(pattern="^(billing|technical|other)$")
    confidence: float = Field(ge=0, le=1)
    rationale: str

def call_llm(prompt: str) -> str:
    ...  # model.generate / API; prefer schema-constrained decoding if available

def extract_ticket(user_text: str, max_retries: int = 2) -> TicketResult:
    prompt = f"""Classify the ticket. Return ONLY JSON with keys
label, confidence, rationale.
Text: {user_text!r}"""
    last_err = None
    for _ in range(max_retries + 1):
        raw = call_llm(prompt)
        try:
            return TicketResult.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e
            prompt += f"\nPrevious output invalid: {e}. Fix the JSON."
    raise RuntimeError(f"validation failed: {last_err}")
```

UX ideas: typed UI fields, refusal path (“insufficient context”), show citations, never display raw chain-of-thought if sensitive.

### NVIDIA NeMo Guardrails (exam-relevant)

Open-source programmable rails around an LLM app (not inside Triton):

| Rail type | Stage |
| --- | --- |
| **Input** | Check / rewrite user message (jailbreak, PII, off-topic) |
| **Dialog** | Colang flows — allowed conversational paths |
| **Retrieval** | Filter RAG chunks |
| **Execution** | Gate tools / actions |
| **Output** | Vet model response before user sees it |

- Config: YAML + **Colang** `.co` flows.
- Integrates with safety models / custom Python actions.
- Study-guide themes: reduce hallucinations, topical control, measure guardrail effectiveness.

### Hallucination reduction checklist

1. Ground with **RAG**; instruct “use only provided context; else say you don’t know.”
2. Require **citations / evidence fields** in the schema; validate they map to retrieved spans.
3. Lower temperature; constrain output structure.
4. Separate “draft” vs “verify” prompts (self-check chain).
5. Application guardrails (NeMo Guardrails) for policy.
6. Don’t claim constrained decoding alone “fixes” hallucinations.

### Exam traps

- Guardrails / prompting live in the **app**, not Triton Inference Server.
- Beam search / sampling ≠ constrained decoding (different goals).
- Schema-valid JSON can still be factually wrong.

---

## Quick self-check

1. Scenario: finance bot must return a fixed JSON schema every time — which layers do you add beyond a prompt?
2. Scenario: multi-hop math word problems fail — CoT or few-shot? Why?
3. Scenario: 200 labeled domain emails, need persistent classifier — ICL vs prompt tuning vs LoRA?
4. Write the causal LM loss in words; when do you mask prompt tokens in SFT?
5. Name NeMo Guardrails’ five rail types and one job each.
6. True/False: Triton includes built-in prompt optimization tools.

## High-signal references

- NVIDIA NCP-GENL blueprint: Prompt Engineering (CoT, zero/one/few-shot, output control) ~13%
- [CertCompanion NCP-GENL — Domain 3 Prompt Engineering](https://certcompanion.com/blog/nvidia-generative-ai-llms-ncp-genl-exam-guide)
- [HF PEFT — soft prompts](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting)
- [NeMo Guardrails overview](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview)
- Wei et al., Chain-of-Thought Prompting (2022); Yao et al., ReAct (2022)
- Study guide links: Cleanlab + NeMo Guardrails for hallucination trust scores; measuring AI guardrail effectiveness
