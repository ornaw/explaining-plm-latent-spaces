# explaining-plm-latent-spaces

Code and data for clustering protein language model embeddings of phosphatase proteins. Each
algorithm is run on ProtT5 and ESM-2 3B embeddings, and its clusters are compared with the
proteins' EC numbers.

## Layout

```
data/phosphatase/            sequences, annotations and embeddings
src/data.py                  dataset loading, deduplication and EC labels
src/esm2_embeddings.py       checks and publishes a downloaded ESM-2 embedding file
src/esm2_analysis.py         profiles an ESM-2 embedding file
src/algo/                    one folder per clustering algorithm
src/data-split/graphpart/    GraphPart folds
src/notebooks/               Colab notebooks for DipDECK and ESM-2 embedding generation
results/                     input tables, and outputs of local runs
```

## Data

In `data/phosphatase/phosphatase/`:

- `tmp/sequences.fasta`: protein sequences
- `data_standard.parquetbundle`: protein annotations
- `tmp/prot_t5.h5`: ProtT5 embeddings
- `esm2_t36_3B_UR50D_full_length.h5`: ESM-2 3B embeddings

`data/phosphatase/dipdeck_phosphatase_data.zip` holds the first three files for the DipDECK notebooks.

The sequences are UniProt entries ([uniprot.org](https://www.uniprot.org)), used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

In `results/`:

- `ec_audit/proteins.csv`: EC numbers for each protein
- `esm2/esm2_t36_3B_UR50D/embedding_manifest.json`: record of how the ESM-2 3B file was made,
  checked by the ESM-2 scripts
- `graphpart/explore_20260928T005618Z/unlabelled_k5/folds_t0p40_all_records.csv`: the GraphPart
  folds DipDECK uses

## Algorithms

| Folder | Algorithm | Runs on |
|---|---|---|
| `src/algo/kmeans` | k-means | local |
| `src/algo/agglomerative` | agglomerative clustering | local |
| `src/algo/subkmeans` | SubKMeans | local |
| `src/algo/hdbscan` | HDBSCAN | local |
| `src/algo/autonr` | AutoNR | local, Colab or Hugging Face Jobs |
| `src/algo/dipdeck` | DipDECK | Colab (GPU) |

Each folder has a ProtT5 script (`<algorithm>_standalone.py`) and an ESM-2 3B script
(`esm2_<algorithm>_standalone.py`). DipDECK has one script for both embeddings and is run through
the notebooks in `src/notebooks/`.

## Local setup

Python 3.12. From the project root:

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

ProtT5 k-means and the GraphPart script also need these source checkouts:

```
git clone https://github.com/tsenoner/protspace external/protspace
git -C external/protspace checkout 1bcbec518df16c6964df351d3626c3961a23f73e
git clone https://github.com/graph-part/graph-part external/graph-part
git -C external/graph-part checkout f470726216bc7ec4cb60f6424548bfd60c550e42
```

Run every command from the project root. Multi-line commands use bash line continuation (`\`).
Scripts that take `--run-scientific` only check their inputs when run without it.

## Run locally

### Data profile

```
python src/data.py
python src/data.py --dedup
```

### ProtT5

```
python src/algo/kmeans/kmeans_standalone.py --run-scientific
python src/algo/agglomerative/agglomerative_standalone.py --run-scientific
python src/algo/subkmeans/subkmeans_standalone.py --manifest
python src/algo/subkmeans/subkmeans_standalone.py --scientific
python src/algo/subkmeans/subkmeans_standalone.py --score
python src/algo/subkmeans/subkmeans_standalone.py --figures
python src/algo/hdbscan/hdbscan_standalone.py
```

Agglomerative and SubKMeans read the k-means results, so run k-means first. For SubKMeans
`--scientific`, `--jobs` times `--threads` must not exceed the number of CPUs.

AutoNR runs one fit per run ID:

```
python src/algo/autonr/autonr_standalone.py run --run-id raw_full_seed0
python src/algo/autonr/autonr_standalone.py validate --run-id raw_full_seed0
```

Options: `--records full|dedup`, `--seed`, and `--geometry` (`raw`, `l2`, `pca95`, `pca90`,
`pca80`, `pca95_l2`, `pca90_l2`, `pca80_l2`). A `raw` fit takes hours.

### ESM-2 3B

Profile the embedding file first:

```
python -m src.esm2_analysis profile \
  --annotation data/phosphatase/phosphatase/data_standard.parquetbundle \
  --fasta data/phosphatase/phosphatase/tmp/sequences.fasta \
  --reference-h5 data/phosphatase/phosphatase/tmp/prot_t5.h5 \
  --h5 data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5 \
  --output-root results/esm2/esm2_t36_3B_UR50D
```

Then:

```
python src/algo/kmeans/esm2_kmeans_standalone.py --run-scientific
python src/algo/agglomerative/esm2_agglomerative_standalone.py --run-scientific
python src/algo/subkmeans/esm2_subkmeans_standalone.py --run-scientific
python src/algo/subkmeans/esm2_subkmeans_standalone.py --score
python src/algo/autonr/esm2_autonr_standalone.py run
python src/algo/autonr/esm2_autonr_standalone.py validate
```

ESM-2 agglomerative and SubKMeans need ESM-2 k-means first. ESM-2 agglomerative also needs the
ProtT5 agglomerative run. ESM-2 AutoNR takes `--workers` and `--threads`.

ESM-2 HDBSCAN needs the profile and the ProtT5 HDBSCAN run:

```
python src/algo/hdbscan/esm2_hdbscan_standalone.py run-main \
  --model esm2_3b \
  --h5 data/phosphatase/phosphatase/esm2_t36_3B_UR50D_full_length.h5 \
  --model-output-root results/esm2/esm2_t36_3B_UR50D \
  --annotation data/phosphatase/phosphatase/data_standard.parquetbundle \
  --fasta data/phosphatase/phosphatase/tmp/sequences.fasta \
  --reference-h5 data/phosphatase/phosphatase/tmp/prot_t5.h5 \
  --source-runs results/hdbscan/metrics/hdbscan_standalone_runs.csv \
  --output-root results/hdbscan/esm2_t36_3B_UR50D
```

### EC scores

After the ProtT5 and ESM-2 runs of an algorithm:

```
python src/algo/kmeans/kmeans_ec_scores.py --run-scientific
python src/algo/agglomerative/agglomerative_ec_scores.py --run-scientific
python src/algo/subkmeans/subkmeans_ec_scores.py --run-scientific
```

### GraphPart folds

```
python src/data-split/graphpart/graphpart_folds.py
```

This runs GraphPart from `external/graph-part` on the alignment edges included in
`src/data-split/graphpart/` and writes the folds to a new folder in `results/graphpart/`.
Do not `pip install` graph-part in the same environment.

## Run on Colab

The DipDECK notebooks in `src/notebooks/` read the project from Google Drive:

1. Upload this project folder to Google Drive as `MyDrive/explaining-plm-latent-spaces`, or set
   `PROJECT_DIR` in the first code cell to where you put it.
2. Open the notebook in Colab with File > Upload notebook.
3. Select the runtime below, then Runtime > Run all.

Outputs are written to the project folder on Drive.

| Notebook in `src/notebooks/` | Runs | Runtime |
|---|---|---|
| `grouped/01_*.ipynb` to `grouped/10_*.ipynb` | DipDECK grid and reference fits with the library reconstruction loss, one arm per notebook | T4 GPU |
| `reference_k15/dipdeck_reference_k15.ipynb` | DipDECK reference fits at initial K 15 | T4 GPU |
| `dipdeck_colab.ipynb`, `dipdeck_colab_dedup.ipynb` | DipDECK ProtT5 grid and reference, all or deduplicated records | GPU |
| `dipdeck_colab_pca*.ipynb` | DipDECK ProtT5 in PCA spaces | GPU |
| `dipdeck_colab_esm2_3b_seeds0to4.ipynb`, `dipdeck_colab_esm2_3b_seeds5to9.ipynb` | DipDECK ESM-2 3B, half the seeds each | GPU |
| `dipdeck_colab_loss_check.ipynb` | DipDECK reference fits with the library loss, raw ProtT5 and ESM-2 3B | GPU |

To merge one finished grouped notebook locally, download its `<block>_complete.zip` files from
`dipdeck_library_loss_archives/` in the project folder on Drive, extract them into one folder, and run:

```
python src/notebooks/dipdeck_library_loss.py --repo . \
  --output-base <extracted folder> \
  --merge-arm prott5_raw1024_full \
  --merge-output <new folder>
```

`--merge-arm` is the grouped notebook name without its number.

These notebooks ask for a file upload instead of using Drive:

| Notebook | Runs | Upload | Runtime |
|---|---|---|---|
| `src/algo/autonr/autonr_colab.ipynb` | one AutoNR fit, raw | `src/algo/autonr/autonr_colab_inputs.zip` | CPU |
| `src/algo/autonr/autonr_colab_pca90.ipynb` | one AutoNR fit, PCA keeping 90% of the variance | `src/algo/autonr/autonr_colab_pca90_inputs.zip` | CPU |
| `src/data-split/graphpart/graphpart_split_colab.ipynb` | GraphPart train/validation/test split and alignment edges | `data/phosphatase/phosphatase/tmp/sequences.fasta` | CPU |
| `src/notebooks/esm2_colab.ipynb` | ESM-2 embeddings | `data/phosphatase/phosphatase/tmp/sequences.fasta` | GPU (A100 for 3B) |

The ESM-2 3B embeddings are already in `data/`. To make them again, follow the last section of
`esm2_colab.ipynb`, which validates the download locally with `src/esm2_embeddings.py`.

## Run on Hugging Face Jobs

AutoNR on ProtT5, raw geometry. Needs a Hugging Face account with Jobs access and credit.

```
pip install huggingface_hub==1.32.0
hf auth login
mkdir hf_input hf_output
cp src/algo/autonr/autonr_hf_job.py src/algo/autonr/autonr_colab_inputs.zip hf_input/
hf jobs run --flavor cpu-performance --timeout 24h \
  --volume ./hf_input:/input \
  --volume ./hf_output:/output:rw \
  --env OMP_NUM_THREADS=32 --env OPENBLAS_NUM_THREADS=32 --env MKL_NUM_THREADS=32 \
  python:3.12-bookworm -- python /input/autonr_hf_job.py --run
```

Without `--run` the job only checks its inputs. Add `--records dedup` for deduplicated records.
Set `--timeout` to the longest run you accept. Follow the job with `hf jobs logs <job id>` and
download the output with the `hf buckets sync` command printed at launch.

## License

The code in this repository is released under the MIT License (see `LICENSE`). This covers the
code only. The ProtT5 and ESM-2 models used to make the embeddings in `data/`, the UniProt
sequences, and the ProtSpace, GraphPart and ClustPy software have their own licences. Check
those before reusing them.
