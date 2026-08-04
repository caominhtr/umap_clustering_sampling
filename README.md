# k-fold cross-validation sampling using UMAP-based dimensionality combined with agglomerative clustering

This implementation is inspired by the work of [Guo et al (2025)](https://link.springer.com/article/10.1186/s13321-025-01039-8). In this paper, the authors demonstrate that UMAP-based splitting creates significantly more challenging and realistic benchmarks for model evaluation than traditional methods (such as random, scaffold-based, or Butina splits). Consequently, UMAP splits are highly recommended for evaluating molecular property prediction and virtual screening tasks.

This code provides an enhanced, end-to-end and ready-to-use UMAP-based splitting that incorporates hyperparameter optimization for `n_clusters`, `n_neighbors`, `min_dist` using [Optuna](https://optuna.org/), followed by the assignment of samples to k-fold cross-validation splits while enforcing predefined maximum Tanimoto similarity constraints between folds.

This optimized UMAP-based k-fold cross-validation sampling was implemented in our [paper](https://link.springer.com/article/10.1186/s13321-026-01262-x):
> **Enhancing virtual screening of cystathionine β-synthase inhibitors: benchmarking target-specific machine-learning scoring functions against state-of-the-art AI docking and co-folding approaches**  
> *Journal of Cheminformatics* (2026). DOI: [10.1186/s13321-026-01262-x](https://link.springer.com/article/10.1186/s13321-026-01262-x)

The repository containing data, source code for this paper can be found in [here](https://github.com/caominhtr/CBS). 

## Workflow
![](Figure_umap.png)

## Installation

```
cd ~/
git clone https://github.com/caominhtr/umap_clustering_sampling.git
cd umap_clustering_sampling
conda env create -f CBS.yaml
conda activate CBS
```

## Execution

The example of dataset in `.smi` format can be found at `example.smi`. Each row corresponds to a SMILES string with its ID. Inputs can also be prepared in `.csv` or `.txt` formats.

```
python umap_cluster.py \
    --input example.smi \
    --output results/ \
    --folds 5 \
    --trials 100 \
    --threshold 0.7 \
    --seed 42
```

where:

- `--input`: Input file containing molecule IDs and SMILES.
- `--output`: Directory where output files will be saved.
- `--folds`: Number of cross-validation folds.
- `--trials`: Number of Optuna optimization trials.
- `--threshold`: Maximum Tanimoto similarity across folds.
- `--seed`: Random seed for reproducibility.

## References

```bibtex
@article{guo2024scaffold,
  title={Scaffold Splits Overestimate Virtual Screening Performance},
  author={Guo, Qianrong and Hernandez-Hernandez, Saiveth and Ballester, Pedro J},
  journal={arXiv preprint arXiv:2406.00873},
  year={2024}
}

@article{guo2024umap,
  title={UMAP-clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines},
  author={Guo, Qianrong and Hernandez-Hernandez, Saiveth and Ballester, Pedro J},
  journal={Journal of Cheminformatics},
  year={2024}
}

@conference{guo2024scaffoldsplits,
    author={Guo, Qianrong and Hernandez-Hernandez, Saiveth and Ballester, Pedro J.},
    editor={Wand, Michael and Malinovsk{\'a}, Krist{\'i}na and Schmidhuber, J{\"u}rgen and Tetko, Igor V.},
    title={Scaffold Splits Overestimate Virtual Screening Performance},
    booktitle={Artificial Neural Networks and Machine Learning -- ICANN 2024},
    year={2024},
    publisher={Springer Nature Switzerland},
    address={Cham},
    pages={58--72},
    isbn={978-3-031-72359-9}
}
```
