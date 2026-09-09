# Neural Book Rating Prediction

Two embedding-based models estimate explicit book ratings from user and book identities: a dot-product network and a hybrid network combining dot-product and multilayer-perceptron branches.

## Model comparison

| Model | Interaction mechanism |
|---|---|
| Dot-product network | Normalized dot product of 10-dimensional user/book embeddings, then dense layers |
| Hybrid network | Separate embedding branches for a normalized dot product and an MLP, combined for prediction |

## Corrected experiment

Repeated user-book pairs are collapsed before splitting. Zero-valued implicit interactions are excluded from rating regression. Both notebooks use the same seeded splits and training-only identity mappings. Validation drives early stopping. MAE and RMSE are evaluated only on known users/books, alongside a training-median baseline; coverage is printed explicitly.

## Setup

Use Python 3.10 or 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open the notebook under `notebooks/` and run its cells in order. Data belongs in `data/`; generated results go to `artifacts/`.

Start with `notebooks/dot_product_ratings.ipynb`, then `notebooks/hybrid_ratings.ipynb`. Only `Ratings.csv` is required; see [data instructions](data/README.md).

## Results

New results require rerunning both notebooks. Previous notebook scores are not comparable: the original preprocessing appended rows already present before a random split, permitting duplicate pairs across partitions. The originals remain under `legacy/` with historical outputs cleared.

## Scope

This is rating prediction, not yet a complete top-N recommendation service. Cold-start handling and ranking metrics remain future work. The corrected models have not yet been fully trained during this refresh.
