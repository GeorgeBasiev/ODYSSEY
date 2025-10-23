# Hybrid Graphs for Table-and-Text based Question Answering using LLMs (Unofficial Implementation)

This repository contains an unofficial implementation of the paper ["Hybrid Graphs for Table-and-Text based Question Answering using LLMs"](https://www.arxiv.org/abs/2501.17767).

## Overview

This implementation demonstrates a method for answering questions that require reasoning over both structured (tables) and unstructured (text) data sources using Large Language Models (LLMs) without fine-tuning. It constructs a unified hybrid graph from the data, prunes it based on the question, and uses the pruned graph to provide relevant context to the LLM.

## Setup

1.  **Install Ollama and Pull the Qwen Model:**
    This implementation uses a local Ollama instance. First, [install Ollama](https://ollama.com/download). Then, pull the required model (e.g., `qwen3:8b`):
    ```bash
    ollama pull qwen3:8b
    ```
    *(If you wish to use a different LLM (local or remote), you can modify the `llm` variable definition in `odyssey_single.py` and `odyssey_multiple.py`.)*

2.  **Create and Activate Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate hybrid_graphs
    ```

3.  **Install HybridQA Dataset:**
    Follow the instructions below, adapted from the [official HybridQA repository](https://github.com/wenhuchen/HybridQA):

    ```bash
    git clone https://github.com/wenhuchen/WikiTables-WithLinks
    wget https://hybridqa.s3-us-west-2.amazonaws.com/preprocessed_data.zip
    unzip preprocessed_data.zip
    ```
    *(Ensure the `WikiTables-WithLinks` folder and the unzipped `preprocessed_data` folder are in your working directory.)*

## Usage

After setup, you can run the pipeline either on a single question or a batch of questions.

### Running a Single Question

1.  Choose a question from any file within the `/preprocessed_data` directory.
2.  Run the preprocessing script with the filename and the specific `question_id` as parameters. For example:
    ```bash
    python preprocess_hybridqa.py preprocessed_data/train_step1.json 000a10c2e1cf0fc6
    ```
    This will generate a preprocessed file for the specific question.
3.  Run the main pipeline script on the generated file:
    ```bash
    python odyssey_single.py preprocessed_for_hybrid_graphs/<your_question_id>.json
    ```
    *(Replace `<your_question_id>` with the actual ID used in step 2.)*

### Running Multiple Questions (Batch)

1.  Choose the part of the HybridQA dataset you want to process (e.g., `train_step1.json`, `dev_inputs.json`).
2.  Preprocess the chosen file:
    ```bash
    python preprocess_hybridqa.py preprocessed_data/<your_chosen_part>.json
    ```
    *(Replace `<your_chosen_part>` with the filename like `train_step1.json`)*
3.  Run the main pipeline script on the generated enriched file:
    ```bash
    python odyssey_multiple.py preprocessed_for_hybrid_graphs/<your_chosen_part>_enriched.json
    ```
    *(Replace `<your_chosen_part>` with the same base filename used in step 2.)*

## Authors and Citation

*   **Original Paper Authors:** Ankush Agarwal, Chaitanya Devaguptapu, et al. [[arXiv Link](https://www.arxiv.org/abs/2501.17767)]
*   **HybridQA Dataset Authors:** Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, William Wang. [[GitHub Link](https://github.com/wenhuchen/HybridQA)]

If you use this implementation or the original paper's ideas, please cite:

```bibtex
@article{agarwal2025hybrid,
  title={Hybrid graphs for table-and-text based question answering using llms},
  author={Agarwal, Ankush and Devaguptapu, Chaitanya and others},
  journal={arXiv preprint arXiv:2501.17767},
  year={2025}
}
@article{chen2020hybridqa,
  title={HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data},
  author={Chen, Wenhu and Zha, Hanwen and Chen, Zhiyu and Xiong, Wenhan and Wang, Hong and Wang, William},
  journal={Findings of EMNLP 2020},
  year={2020}
}