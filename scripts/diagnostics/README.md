# Diagnostic Scripts

This folder contains CLIs for evaluating and visualizing offline recommender artifacts.

These scripts read existing embeddings and metadata from `data/`. They do not run
SPECTER2 encoding. The main production pipeline can call the same implementation
through optional flags, but the scripts here are meant for manual diagnostics.

Commands:

```bash
python scripts/diagnostics/diagnose_kmeans_k.py --sample-size 200000
python scripts/diagnostics/retrain_kmeans_index.py --k 700 --output-prefix data/kmeans_k700
python scripts/diagnostics/visualize_clusters_pca.py --sample-size 50000
python scripts/diagnostics/visualize_clusters_umap.py --sample-size 50000
```
