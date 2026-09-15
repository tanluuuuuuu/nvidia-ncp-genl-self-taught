# ALiBi — Attention with Linear Biases

**Idea:** skip position vectors entirely. After content scores $q_i^\top k_j$, **subtract a penalty that grows with distance**:

$$
\text{score}_{ij} = q_i^\top k_j - m \cdot |i - j|
$$

(For causal LMs, usually only $j \le i$, so distance is $i - j$.)

- $m > 0$ is a **slope**. Farther keys get lower logits → softmax prefers nearer context unless content matches strongly.
- Different heads get different slopes (geometric progression): steep heads ≈ local; flat heads ≈ long-range.
- No rotary kernels, no position embedding params — position never enters the residual stream as a vector; it only shapes attention scores.

**Why it helps**

- Extremely simple.
- Strong **length extrapolation** story: train short, run longer — the linear distance tax still makes sense.

**Mental model:** attention = “content match” minus “how many tokens away,” with heads disagreeing on how harsh that tax is.

#### RoPE vs ALiBi

| | **RoPE** | **ALiBi** |
| --- | --- | --- |
| Where position lives | Inside Q/K via rotation | Bias on attention logits only |
| Relative? | Yes (geometry of rotations) | Yes (explicit distance) |
| Affects V? | Usually no | N/A |
| Intuition | Relative angle between tokens | Soft local-attention prior |
| Extrapolation | Good; may need extra tricks at extreme lengths | Very good / simple |

**Exam hooks:** transformers need position because they have no inherent order; RoPE rotates Q/K; ALiBi adds a distance bias to scores; both improve length extrapolation vs absolute embeddings.