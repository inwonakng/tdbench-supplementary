# Supplementary Materials for TDBench: Tabular Data Distillation Benchmark

## Files

**Data Files**

These files are not included in the repository due to their sizes.

- `data_mode_switch_results_w_reg.csv`
  - Contains the results of every run.
  - Contains information such as the scores, runtime and various parameters of the run.
  - Download link: [https://drive.google.com/file/d/1DPIGMo1_4iwYXMchMZPXBnIujsSfc5I_/view?usp=share_link](https://drive.google.com/file/d/1DPIGMo1_4iwYXMchMZPXBnIujsSfc5I_/view?usp=share_link)
- `ds_stats.csv`
  - Contains dataset statistics.
  - Download link: [https://drive.google.com/file/d/1_0p3gZ47y5gfrwoTDE51eZ_xn3nEC3YH/view?usp=share_link](https://drive.google.com/file/d/1_0p3gZ47y5gfrwoTDE51eZ_xn3nEC3YH/view?usp=share_link)
- `enc_stats.csv`
  - Contains statistics of the encoder models.
  - Download link: [https://drive.google.com/file/d/1u3SsQ9p3OiiOX1AakbnJqQIYx5AfYA7d/view?usp=share_link](https://drive.google.com/file/d/1u3SsQ9p3OiiOX1AakbnJqQIYx5AfYA7d/view?usp=share_link)


**Scripts**

- `analyze_results.py`
  - Driver code that produces most of the results seen in the main paper.
  - Conducts the analysis in the order seen in the manuscript.
- `classifier_performance.py`
  - Helper code to parse/rank the results by different groups.
- `ft_resnet_runtime.py`
  - Script to measure the average runtime of the FTTransformer and ResNet downstream classifiers.


## Python environment

Python version: `3.10.12`

Use of conda is highly encouraged.

The following snippet will create an environment called `tdbench` with all the requirements installed.
Note that the requirements file must be changed for OsX machines.
The code has not been tested on windows.

```bash
conda create -n tdbench python=3.10.12
conda activate tdbench
pip install -r requirements_x86.txt
```

