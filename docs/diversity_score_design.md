# Diversity Index Score Design

## Goal

The diversity index, `delta`, controls how broad a user's recommendation feed
should be. Retrieval already uses `delta` to control search breadth:

```text
cluster_budget(delta) = ceil(base + delta * multiplier)
```

With the current constants:

```text
base = DAILY_CLUSTER_BUDGET_BASE = 4
multiplier = DAILY_CLUSTER_BUDGET_DIVERSITY_MULTIPLIER = 8
cluster_budget(delta) = ceil(4 + 8 * delta)
```

This design keeps that retrieval-breadth formula unchanged. The scoring change
uses `delta` during final selection so the feed can cover multiple user research
centroids more smoothly.

The design intentionally does not penalize a candidate paper for being similar
to papers already selected. Diversity is encouraged through user-centroid
coverage, cluster saturation, and the existing hard per-cluster cap.

## Score Formula

For candidate paper `p`, selected feed prefix `S`, user profile centroids
`U = {u_1, ..., u_k}`, and diversity index `delta`:

```text
score(p | S, U, delta) =
    relevance(p, U)
  + alpha * recency(p)
  + delta * beta * centroid_coverage(p | S, U)
  - delta * eta * cluster_saturation(p | S)
```

Expanded:

```text
score(p | S, U, delta) =
    max_i cosine(e_p, u_i)
  + alpha * recency(p)
  + delta * beta * (1 / (1 + thread_count_S(nearest_thread(p))))
  - delta * eta * (cluster_count_S(cluster_id(p)) / max_per_cluster)
```

Current initial weights:

```text
alpha = 0.25
beta  = DAILY_CENTROID_COVERAGE_WEIGHT = 0.10
eta   = DAILY_CLUSTER_SATURATION_WEIGHT = 0.05
```

## Variable Definitions

### Candidate Paper

```text
p
```

A paper returned from retrieval. In code, candidates are tuples:

```text
(sim_score, paper_meta, nearest_centroid_idx)
```

### Selected Prefix

```text
S
```

The papers already selected into the feed. Since coverage and saturation depend
on selected papers, scores are recomputed greedily after each selected paper.

### User Centroids

```text
U = {u_1, ..., u_k}
```

The user's research-interest vectors. `k` is `k_u` in the codebase.

### Paper Embedding

```text
e_p
```

The embedding vector for paper `p`. Paper and user vectors are unit-normalized,
so cosine similarity is a dot product.

### Diversity Index

```text
delta
```

The user's diversity preference:

```text
0.0 <= delta <= 1.0
```

Serving clamps the value before use:

```text
delta_clamped = min(1.0, max(0.0, delta))
```

Interpretation:

```text
delta = 0.0 -> focused ranking, mostly relevance + recency
delta = 0.5 -> moderate centroid coverage and cluster saturation pressure
delta = 1.0 -> strongest coverage and saturation pressure
```

### Relevance

```text
relevance(p, U) = max_i cosine(e_p, u_i)
```

This is the best semantic match between paper `p` and any user centroid. In the
current code, this is `sim_score` from `knn_in_clusters(...)`.

### Nearest User Thread

```text
nearest_thread(p) = argmax_i cosine(e_p, u_i)
```

This identifies which user centroid best matches the paper. In code, this is
`nearest_centroid_idx`.

### Recency

The current recommender uses:

```text
recency_score(update_date)
```

and combines it with semantic relevance:

```text
base_score(p) = relevance(p, U) + alpha * recency(p)
```

### Centroid Coverage

```text
centroid_coverage(p | S, U) =
    1 / (1 + thread_count_S(nearest_thread(p)))
```

Where:

```text
thread_count_S(t) = number of selected papers whose nearest_thread is t
```

Examples:

```text
thread_count = 0 -> coverage = 1.000
thread_count = 1 -> coverage = 0.500
thread_count = 2 -> coverage = 0.333
thread_count = 3 -> coverage = 0.250
```

This rewards candidates from under-covered user interests without requiring a
hard one-paper-per-centroid rule.

### Cluster Saturation

```text
cluster_saturation(p | S) =
    cluster_count_S(cluster_id(p)) / max_per_cluster
```

Where:

```text
cluster_count_S(c) = number of selected papers from k-means cluster c
max_per_cluster = DAILY_MAX_PER_CLUSTER
```

The recommender still enforces:

```text
cluster_count_S(c) < DAILY_MAX_PER_CLUSTER
```

So cluster saturation is a soft penalty before the hard cap is reached.

### No Selected-Paper Similarity Penalty

The formula intentionally excludes:

```text
max_{q in S} cosine(e_p, e_q)
```

That means a paper is not penalized merely because it is similar to an already
selected paper. Diversity comes from breadth, centroid coverage, and cluster
saturation instead.

## Behavior By Delta

At `delta = 0`:

```text
score(p) = relevance(p, U) + alpha * recency(p)
```

At `delta = 0.5`:

```text
score(p | S, U) =
    relevance(p, U)
  + 0.25 * recency(p)
  + 0.05 * centroid_coverage(p | S, U)
  - 0.025 * cluster_saturation(p | S)
```

At `delta = 1`:

```text
score(p | S, U) =
    relevance(p, U)
  + 0.25 * recency(p)
  + 0.10 * centroid_coverage(p | S, U)
  - 0.05 * cluster_saturation(p | S)
```

This removes the old hard threshold behavior where centroid coverage only
activated when `delta > 0.5`.

## Greedy Selection Algorithm

```text
selected = []
thread_counts = Counter()
cluster_counts = Counter()

while len(selected) < n:
    best = None
    best_score = -infinity

    for candidate in candidates:
        if candidate.paper_id in seen_ids:
            continue
        if candidate.paper_id already selected:
            continue
        if cluster_counts[candidate.cluster_id] >= DAILY_MAX_PER_CLUSTER:
            continue

        base_score =
            candidate.raw_similarity
          + alpha * candidate.recency_score

        coverage =
            1 / (1 + thread_counts[candidate.nearest_thread])

        saturation =
            cluster_counts[candidate.cluster_id] / DAILY_MAX_PER_CLUSTER

        adjusted_score =
            base_score
          + delta * beta * coverage
          - delta * eta * saturation

        if adjusted_score > best_score:
            best = candidate
            best_score = adjusted_score

    if best is None:
        break

    selected.append(best)
    thread_counts[best.nearest_thread] += 1
    cluster_counts[best.cluster_id] += 1
```

## Implementation Notes

The current implementation:

1. clamps diversity through `recommender.diversity.clamp_diversity`,
2. preserves the retrieval formula in `recommender.retrieve.find_nearest_clusters`,
3. applies greedy adjusted scoring in `recommender.engine.select_with_relaxation`,
4. keeps `DAILY_MAX_PER_CLUSTER` as a hard cap,
5. adds debug metadata to selected papers:

```text
raw_similarity
recency_score
base_score
diversity_adjusted_score
centroid_coverage_bonus
cluster_saturation_penalty
nearest_centroid_id
```

## Evaluation Metrics

Recommended sweeps:

```text
delta in {0.0, 0.25, 0.5, 0.75, 1.0}
```

Track:

```text
NDCG@20
Precision@20
unique user centroids@20
thread coverage@20
unique clusters@20
cluster entropy@20
category entropy@20
mean raw similarity
Jaccard overlap between adjacent delta feeds
```

Expected behavior:

```text
delta rises -> thread and cluster coverage rise
delta rises -> mean relevance declines mildly, not sharply
adjacent delta values -> feed changes smoothly
```

## Non-Goals

This design does not:

1. change the offline pipeline,
2. change k-means training or corpus embeddings,
3. penalize selected-paper embedding similarity,
4. replace the retrieval breadth formula,
5. learn `delta` automatically from feedback.

