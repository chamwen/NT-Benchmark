# Benchmark of negative transfer

[![](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)
[![](https://img.shields.io/github/last-commit/chamwen/NT-Benchmark)](https://github.com/chamwen/NT-Benchmark/commits/main)

Negative transfer (NT) represents that introducing source domain data/knowledge decreases the learning performance in the target domain, which is a long-standing and challenging issue in transfer learning. 

This repository contains codes of NT detection experiments with over 20 representative approaches on three NT-specific datasets.



## Datasets

There are three NT-specific datasets with large domain shift, i.e., on one synthetic dataset, and two real-world datasets on object recognition and emotion recognition. Their statistics are summarized below:

<div align="center">
    <img src="presentation/dataset.png", width="750">
</div>


## Running the code

The code is mainly based on Python 3.8 and PyTorch 1.8. 

Code files introduction:

**NT_UDA/** -- NT detection experiments on 12 unsupervised domain adaptation approaches

**NT_SSDA/** -- NT detection experiments on 11 semi-supervised domain adaptation approaches

**NT_Noise/** -- NT detection experiments under source label noise

**NT_UDA/data_synth/** -- the synthetic dataset and code to generate it



Specifically, take NT-UDA for example, the procedure is:

1. Initialize the networks involved in the experiments to ensure fair comparisons by

   ```python
   python save_init_model.py
   ```

2. Run the script of a particular approach on a specific dataset

   ```python
   python demo_img_dnn.py
   ```

   

## Benchmark

For **unsupervised TL** on DomainNet and SEED datasets:

<div align="center">
    <img src="presentation/nt-uda.png", width="800">
</div>



For **semi-supervised TL** on DomainNet and SEED datasets:

<div align="center">
    <img src="presentation/nt-ssda.png", width="800">
</div>



## Citation

This code is corresponding to our IEEE/CAA JAS 2023 paper below:

```
@article{zhang2022ntl,
  title = {A Survey on Negative Transfer},
  author = {Zhang, Wen and Deng, Lingfei and Zhang, Lei and Wu, Dongrui},
  journal = {IEEE/CAA Journal of Automatica Sinica},
  year = {2023},
  volume = {10},
  number = {2},
  pages = {305--329}	
}
```

Please cite our paper if you like or use our work for your research, thank you!
