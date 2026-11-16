# Foundation vs. Classical Classification Models on Medical Tabular Data

**Built with PriorLabs-TabPFN**

Simualtions comparing logistic regression, Random Forest, XGBoost,
CatBoost, QDA, QUADRO, TabPFN (2.5, 2.6, 3, and 3.5), EXAONE Tabular, and TabICL v1/v2.
Experiments use synthetic scenarios or repeated sampling from three medical datasets.

## 1. Prepare the environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/ghorbani-alireza/medfm
cd medfm
```
 
From the project directory, run:

```bash
uv sync --locked --group notebook
```

This creates the Python 3.13 environment in `.venv` using `uv.lock`.
Use this environment for both terminal commands and the notebook.

Credentials belong in the file `.secrets/token.py`:

```python
HF_TOKEN = "your_huggingface_token"
TABPFN_TOKEN = "your_tabpfn_token"
```

For real-data resampling, we used 3 datasets:

| Data File | Task (positive class) |  DOI |
|----------------------------------------|----------------------------------------|--------|
| `data1_echo_notes.pkl`                 | Abnormal LV systolic report (`-2, -1, 1, 2, 3` vs `0`) | https://doi.org/10.13026/xhrz-ht59 |
| `data2_blood_glucose_management.pkl`   | Next glucose in 70-180 mg/dL after an insulin event | https://doi.org/10.13026/517s-2q57 |
| `data3_blood_gas_oximetry.pkl`         | SaO2 < 88% |  https://doi.org/10.13026/phvt-3277 |

The raw records are controlled-access electronic health records and are not
redistributed here. Access is governed by the PhysioNet data-use agreements for
each DOI. The `io/real_data/*.pkl` files used by the loaders must be obtained
separately and are not part of this repository.

### Troubleshooting

**`uv: command not found`**

If uv is not installed, use the [official installer](https://docs.astral.sh/uv/getting-started/installation/). For Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If uv is installed but the terminal cannot find it:

```bash
source "$HOME/.local/bin/env"
uv --version
```

**`No locks available` during installation or model download**

On servers with this error, from the project directory, run:

```bash
export BENCHMARK_CACHE="$(mktemp -d /tmp/benchmark-cache-XXXXXXXX)"
export UV_CACHE_DIR="$BENCHMARK_CACHE/uv"
export TABPFN_MODEL_CACHE_DIR="$BENCHMARK_CACHE/tabpfn"
export HF_HOME="$BENCHMARK_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_LINK_MODE=copy

uv sync --locked --group notebook
```


## 2. Run the full analyses from the terminal

The pilot includes EXAONE; the full analyses run all other model runners.

Pilot: setting 2, 20 repetitions, using the normal sequential `simulation()` function:

```bash
uv run --locked python scripts/run_analysis.py pilot --output io/sim_results/pilot_setting2_new
```

The pilot also accepts `--repetitions`, `--seed`, and `--settings`. Choose a new
output directory for each run.

Main simulation: all eight baseline settings:

```bash
uv run --locked python scripts/run_analysis.py simulation --repetitions 100 --output io/sim_results/main
```

Sensitivity analysis: the nine sensitivity settings, including setting 7+:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --repetitions 100 --output io/sim_results/sensitivity
```

Real-data resampling: all three datasets:

```bash
uv run --locked python scripts/run_analysis.py resampling --repetitions 500 --output io/real_data_results/main
```

To run a subset, add `--settings 2 5` to simulation/sensitivity or `--datasets echo blood_gas` to
resampling. Each option accepts one or more space-separated values:


**`FileNotFoundError` for a TabPFN checkpoint during a parallel run** First run one repetition with a single worker to download and verify the models:

```bash
uv run --locked python scripts/run_analysis.py sensitivity --settings 1 --repetitions 1 --jobs 1 --output io/sim_results/worker_download
```

After it succeeds, retry.


Main setting IDs are `1` through `8`; sensitivity also accepts `7+`. Dataset
names are `echo`, `glucose`, and `blood_gas`. 

## 3. Tables and plots from saved results

Reporting uses the saved CSV files, without fitting models or reading raw data:

```bash
uv run --locked python scripts/run_analysis.py report --results io/sim_results/main
uv run --locked python scripts/run_analysis.py report --results io/sim_results/sensitivity
uv run --locked python scripts/run_analysis.py report --results io/real_data_results/main
```


Reports save `tables/` and `plots/` inside the selected results directory. The
same command works for a saved pilot directory.

Tables are `measures.csv` (per-method mean ± SD), `best_methods.csv` (best and
second-best per group), and `best_vs_best.csv` (the best tree and best foundation
per measure, their mean values, and the paired foundation-minus-tree effect).
When `--baseline` is given, `sample_size_changes.csv` adds the sensitivity-minus-main
gain per method.



## Foundation models

Default ensemble settings and pretrained checkpoint filenames (check out notebooks/run_analysis.ipynb):

| Model | Ensemble setting | Checkpoint filename |
|--------------------|--------------------|----------------------------------------------------|
| TabPFN 2.5         |    `default = 8`   | `tabpfn-v2.5-classifier-v2.5_default.ckpt`         |
| TabPFN 2.6         |    `default = 8`   | `tabpfn-v2.6-classifier-v2.6_default.ckpt`         |
| TabPFN 3           | `default = "auto"` | `tabpfn-v3-classifier-v3_default.ckpt`             |
| TabPFN 3.5         | `default = "auto"` | `tabpfn-v3.5-20260909.safetensors`                 |
| EXAONE Tabular 1.0 |    `default = 8`   | `exaone-tabular-classifier-v1_default.safetensors` |
| TabICL v1          |    `default = 8`   | `tabicl-classifier-v1-20250208.ckpt`               |
| TabICL v2          |    `default = 8`   | `tabicl-classifier-v2-20260212.ckpt`               |

The four TabPFN source packages are included in `vendor/`. No additional vendor
setup is needed for normal runs.
Their local names (`tabpfn_v25`, `tabpfn_v26`, `tabpfn_v3`, and `tabpfn_v35`)
allow the versions to be imported together for side-by-side benchmarking.


Both TabICL runners leave the ensemble size unset and use the installed package's
default (= 8). Checkpoints select the original v1 and v2 models using
the [TabICL API](https://tabicl.readthedocs.io/en/latest/api.html).

## Files

- `src/medfm/`: data loading, settings, model runners, analysis, and reporting.
- `vendor/`: the four renamed TabPFN source copies.
- `scripts/`: main function to run analysis and get reports.
- `notebooks/run_analysis.ipynb`: model checks, pilot, and reports from saved results.
- `io/`: datasets and saved results; after calling the `report` function (described above), each results directory holds its own `tables/` and `plots/`.

## Licenses

The benchmark's own code uses the MIT license. The TabPFN copies retain their
upstream licenses in their individual vendor folders; those licenses apply to
the copied sources. See the upstream license texts for
[v6.0.0](https://github.com/PriorLabs/TabPFN/blob/v6.0.0/LICENSE),
[v7.1.1](https://github.com/PriorLabs/TabPFN/blob/v7.1.1/LICENSE),
[v8.5.0](https://github.com/PriorLabs/TabPFN/blob/v8.5.0/LICENSE), and
[v9.0.0](https://github.com/PriorLabs/TabPFN/blob/v9.0.0/LICENSE).

