## Variational Prefix Tuning (VPT) for Code Summarization

### Installation

After cloning/downloading the repository, run the following command to create the conda environment for running.


```
conda env create -f environment.yml
conda activate CSTCVAE
```

### Environment tested
The code is tested in Ubuntu 24.04 environment, please make sure you have JAVA and curl installed in your environment.

### Datasets
Before you train or test a model, follow instructions in data/README.md to download and preprocess the datasets.

### Finetune CodeT5+ for your dataset.

Please use python notebook [finetune_codeT5P.ipynb](finetune_codeT5P.ipynb) to finetune the CodeT5+ model for your dataset.

### Apply Variational Prefix Tuning (VPT) for Code Summarization.

Change backbone model and dataset directory to your finetuned model in [CodeT5+_VPT_bi-criteria.ipynb](CodeT5%2B_VPT_bi-criteria.ipynb).\
This file will train the VPT for your fine-tuned model and run inference and subset selection on the test set you spcify.

### Generated Results
Results generated for the CodeT5+ using VPT method are stored in the [T5_results](T5_results) directory, under file with name of
CodeT5+_VPT_bicriteria_subset_selection_results_lang.txt.

### Results generation of all other decoding methods
Please use the [codeT5P_diverse_decoding.ipynb](codeT5P_diverse_decoding.ipynb) to generate results for all other decoding methods.\
Results will be stored in [diverse_decoding_results](diverse_decoding_results) directory.

### Experiment Results and Weights
Our model weights and inference results can be accessed from https://drive.google.com/file/d/1nyZds1A_cyS5bWaBy4rd7DKr9Ej2vZl2/view?usp=sharing
For the weights included in [CodeT5_VPT_Weight](CodeT5_VPT_Weight), please first load the backbone model CodeT5+ from huggingface, then load our provided weights.
The textual predictions for conducting all the experiments in out paper in [RQ_Textual_Predictions](RQ_Textual_Predictions).
The files are separated based on Research Questions (RQ) and datasets, with name indicating the infernce strategy used.
For all Sampling and VPT results, please use Bi-criteria subset selection from the 100 results provided to obtain the number of summaries you need.

