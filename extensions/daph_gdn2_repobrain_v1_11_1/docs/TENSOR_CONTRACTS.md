# Tensor contracts

## DAPH hidden states

- hidden states: `[B, T, D]`
- `B`: batch
- `T`: causal sequence length
- `D`: DAPH hidden size

## GDN2 recurrence reference

- keys `k`: `[B,T,H,K]`
- values `v`: `[B,T,H,V]`
- erase `b`: `[B,T,H,K]`
- write `w`: `[B,T,H,V]`
- log decay `g`: `[B,T,H,K]`
- recurrent state `S`: `[B,H,K,V]`
- recurrence output: `[B,T,H,V]`

The reference function intentionally starts after q/k/v/gate projection. Its purpose is to validate recurrence numerics independent of upstream projection code.

## Repository features

Assuming 1024-dimensional frozen file embeddings:

- chunk embedding: `[C,1024]`
- file embedding after chunk mean: `[1024]`
- all file vectors: `[N,1024]`
- weighted mean: `[1024]`
- max pool: `[1024]`
- repo embedding: `[2048]`

## Repo2LoRA-Lite

Default generator input: `[1,2048]`.

For module shape `(out,in)`, rank `R`, groups `G`:

- generated `A`: `[G,R,in]`
- generated `B`: `[G,out,R]`
- diagnostic dense delta: `[out,in]`

Example for a 1024 -> 1024 projection, rank 8, four groups:

- `A`: `[4,8,1024]`
- `B`: `[4,1024,8]`

Do not materialize dense deltas during ordinary inference.

## Vision

- semantic tokens: `[B,N,D_sem]`
- structural patch tokens: `[B,N,D_struct]` after spatial alignment/resampling
- fused tokens: `[B,N,D_daph]`
