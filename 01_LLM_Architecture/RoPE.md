# RoPE — Rotary Position Embedding

**The problem:** attention compares tokens with $q^\top k$. That score should depend on **relative distance** — “she” attending to “dog” 3 tokens back should behave the same whether the phrase sits at the start or the end of the document. Absolute position ids can’t promise that; rotation can.

**Step 1 — the one fact everything rests on (2D).**

Take the simplest possible vectors $u = v = (1,0)$. Rotate $u$ by angle $\theta_i$, rotate $v$ by angle $\theta_j$, then take the dot product:

$$
R(\theta_i)u = (\cos\theta_i,\ \sin\theta_i),\qquad
R(\theta_j)v = (\cos\theta_j,\ \sin\theta_j)
$$

$$
(R(\theta_i)u)\cdot(R(\theta_j)v)
= \cos\theta_i\cos\theta_j + \sin\theta_i\sin\theta_j
= \cos(\theta_i - \theta_j)
$$

Only the **difference** $\theta_i - \theta_j$ survives. The absolute angles cancel.

**Numeric check** (let each token’s angle be its position × 30°):

- Tokens at positions **1** and **4** (gap 3): $\cos(90°) = 0$
- Tokens at positions **101** and **104** (gap 3): $\cos(90°) = 0$ — **same score**

Same gap → same positional effect, no matter where in the sentence. That is exactly the “relative position” property we wanted. (This holds for general $u, v$ too: $(R_{\theta_i}u)^\top(R_{\theta_j}v) = u^\top R_{\theta_j-\theta_i}v$.)

**Step 2 — make the angle grow with position.**

Token at position $m$ gets rotated by $m\theta$ (token 17 spins 17× more than token 1). The rotation is just the standard 2D formula you know from graphics:

$$
x' = x\cos(m\theta) - y\sin(m\theta),\qquad
y' = x\sin(m\theta) + y\cos(m\theta)
$$

**Step 3 — real vectors are longer than 2D → pair them up.**

Rotation only exists in 2D, so split each head’s Q/K into 2D pairs $(x_1,x_2), (x_3,x_4), \ldots$ (a 128-dim head = 64 little clock hands) and rotate each pair.

**Step 4 — different pairs spin at different speeds (clock analogy).**

- **Fast hand** (large $\theta$): $\cos(m\theta)$ changes quickly — great at telling *nearby* tokens apart, but it wraps around and repeats at long range (aliasing).
- **Slow hand** (small $\theta$): changes slowly — still meaningful for *far* tokens.

One hand can’t cover all distances; a bank of fast+slow hands does (like seconds vs hours on a clock).

**Step 5 — rotate Q and K, not V.**

Position should control *who talks to whom* (the score $q^\top k$), not rewrite *what content is delivered* (V). Everything downstream — softmax, value mixing — is unchanged.

**Why it helps**

- Naturally **relative** → often extrapolates past training length better than absolute embeddings (extreme lengths may still need YaRN / NTK scaling).
- No large learned position table; widely used (LLaMA-family, etc.).

**Mental model:** every Q/K vector carries a bundle of clock hands; token position sets how far each hand has spun; when two tokens meet, only their *spin difference* (their gap) affects the match score.