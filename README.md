## Computational environment

**Note:** The majority of analyses — that is, the Python scripts outside the `helpers/` folder—were run in ProQuest TDM Studio using the U.S. Newsstream corpus. Export restrictions therefore apply.

The CPU-based analyses were run on an 8-vCPU Intel Xeon Platinum 8259CL @ 2.50 GHz system (4 physical cores / 8 threads; approximately 30 GB RAM). Transformer-based models were run on a GPU-backed Amazon SageMaker instance provisioned through TDM Studio, using an NVIDIA Tesla T4 GPU with approximately 16 GB VRAM (around 15 GiB usable).

A matched Conda environment was used for the TDM Studio analyses. The local environment used to generate the plots is recorded in `environment_plotting.yml`.

Some intermediate outputs, including sentence embeddings, cannot be included in this repository because of corpus export restrictions. Consequently, some output folders may differ slightly between the TDM Studio and local versions. However, the repository structure and filenames have been kept as consistent as possible.

### Local plotting environment

To create the Conda environment used for the plotting notebook, run the following commands from the repository directory:

```bash
conda env create -f environment_plotting.yml
conda activate sibling-plotting
python -m ipykernel install --user --name sibling-plotting --display-name "Python (SIBling plotting)"
```

Then select **Python (SIBling plotting)** as the Jupyter kernel when running `step4_plot_data.ipynb`.

An exact Windows-specific record of the resolved package builds is provided in `environment_plotting_win64_explicit.txt`. On Windows, this environment can instead be recreated using:

```bash
conda create -n sibling-plotting --file environment_plotting_win64_explicit.txt
```

The YAML file is intended to provide the main reproducible environment specification. The explicit TXT file records the exact Windows package builds used locally.

## Analysis workflow

Run the analyses in the following order:

1. `0_corpus_processing/`
2. `step1_compute-valence_arousal-index.ipynb`
3. `step2_compute-breadth-score.ipynb`
4. `step3_compute-complementary-indices.ipynb`
5. `step4_plot_data.ipynb`
6. Relevant R scripts in `statistical_analyses/`

The files in `helpers/` contain reusable functions called by the notebooks and are not generally intended to be run directly.
