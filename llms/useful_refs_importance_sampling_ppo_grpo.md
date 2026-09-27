# Useful References: Importance Sampling, PPO and GRPO

## Intro

Training modern LLMs with reinforcement learning (RLHF, RLVR, RL on verifiable rewards) rests on a
small set of core ideas: **importance sampling**, the **clipped PPO objective**, and the **group-relative**
advantage normalization popularized by **GRPO**.

This page distills Nando de Freitas' (@NandoDF) compact handwritten notes on exactly that stack —
from the definition of the RL objective all the way to the full GRPO-style objective with the
unbiased k3 KL estimator. It is a dense but self-contained refresher: notation first, then the
estimators, then the algorithms.

- X post: <https://x.com/NandoDF/status/1919728246821634205>
- Attached X Article: <https://x.com/i/article/1919697329487069184> ("May 6: Importance sampling, PPO and GRPO")

## Notation

- `o` — observation (prompt)
- `a` — action (completion)
- `θ` — policy parameters
- `R = r(a, o)` — the reward
- `D = {o^i, a^i, r(a^i, o^i)}_{i=1}^N` — dataset of N samples

## The objective

```
J(θ) = E[r(a, o)] = ∫ r(a, o) π_θ(a|o) P(o) da do = E_{π_θ}[r(a, o)]
```

## Importance sampling

Sample from a different (old/behavior) policy `μ` instead of `π_θ`:

```
w(θ) = π_θ(a|o) / μ(a|o)
J(θ) ≈ (1/N) Σ_i w(θ^i) r(a^i, o^i),   (o^i, a^i) ~ μ
```

Per-sample importance weights against the old policy:

```
w^i = π_θ(a^i|o^i) / π_old(a^i|o^i)
J(θ) ≈ (1/N) Σ_i w^i r(a^i, o^i)
```

### Variance of the weights

```
Var[w] = E[w²] − (E[w])²,   E[w] = 1   (since ∫ π_θ(a|o) da = 1)
E[w²] = E_{π_old}[(π_θ/π_old)²] = E_{π_θ}[π_θ/π_old]
```

The variance blows up when `π_θ/π_old` is large — i.e. when the policies differ a lot.

### Self-normalized weights

Trades a little bias for lower variance:

```
w̃^i = w^i / Σ_j w^j,   so that Σ_i w̃^i = 1
```

## Baselines and advantage

A baseline `b` that is independent of `a^i` does not change the expectation:

```
A(a, o) = r(a, o) − b
J(θ) ≈ (1/N) Σ_i w^i (r^i − b)
```

**RLOO** (REINFORCE leave-one-out baseline): the baseline is the mean reward of the *other*
samples for the same prompt:

```
b = (1/(N−1)) Σ_{j≠i} r(a^j, o^j)
A^i = r(a^i, o^i) − (1/(N−1)) Σ_{j≠i} r(a^j, o^j)
```

## PPO (clipped surrogate objective — pessimistic update)

```
J(θ) ≈ (1/N) Σ_i min( w^i A^i, clip(w^i, 1−ε, 1+ε) A^i )
```

Clipping: when `A^i > 0` and `w^i > 1+ε` the update is clipped (don't push the policy too
far for good actions); likewise when `A^i < 0` and `w^i < 1−ε`. This limits how much the
new policy can deviate from the old one.

## KL penalty: the k3 estimator

Unbiased and always positive. Per-sample KL term against the SFT reference policy:

```
−log( π_SFT(a^i|o^i) / π_θ(a^i|o^i) ) + π_SFT(a^i|o^i)/π_θ(a^i|o^i) − 1
```

## GRPO (group-relative advantage)

Normalize rewards within the group of samples for a prompt:

```
A^i = ( r^i − mean(r) ) / std(r)
```

Full GRPO-style objective (PPO clipping + KL penalty against the SFT reference policy):

```
J(θ) ≈ (1/N) Σ_i min( [π_θ(a^i|o^i)/π_old(a^i|o^i)] r^i,
                      clip[π_θ(a^i|o^i)/π_old(a^i|o^i), 1−ε, 1+ε] r^i )
     − (β/N) Σ_i [ −log( π_SFT(a^i|o^i)/π_θ(a^i|o^i) )
                   + π_SFT(a^i|o^i)/π_θ(a^i|o^i) − 1 ]
```

## Further reading (mentioned in replies to the post)

- Marc Bellemare's reply references: <https://arxiv.org/html/2503.1428>
