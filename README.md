#  Model Merging: Enhancing NLP Task Performance by Integrating Fine-Tuned Small Models with BERT Base


In the rapidly evolving field of Natural Language Processing (NLP), the quest for building more efficient and effective models is continuous. Traditional approaches often involve fine-tuning pre-trained models, like [BERT Base](https://huggingface.co/google-bert/bert-base-uncased), to achieve state-of-the-art performance on various NLP tasks such as sentiment analysis, question answering, etc. However, the computational costs and resource demands associated with fine-tuning these big models can be prohibitive, especially in real-world applications where efficiency is crucial.

Recent advancements have introduced smaller, more efficient models like [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D), which are designed to retain much of the performance of their larger counterparts while significantly reducing computational requirements. These models have opened up new possibilities for creating lightweight NLP solutions. Yet, the question remains: can the strengths of these smaller models be effectively combined with the power of large pre-trained models to further enhance performance?

This project seeks to explore this possibility by evaluating whether merging a fine-tuned small model with a large pre-trained model can improve performance on specific NLP tasks, compared to the traditional approach of fine-tuning the large model alone. By leveraging the [GLUE benchmark](https://gluebenchmark.com/), which is widely recognized for benchmarking various NLP tasks, this study aims to provide insights into the potential benefits and trade-offs of model merging in practical applications.

The primary focus will be on determining whether a hybrid approach can offer a superior balance of accuracy and resource utilization compared to the more classical approach. To do this we will use BERT Base as the large model and TinyBERT as the lightweight model and we will compare their performance on two different datasets: [Internet Movie Database (IMDb)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) in terms of accurancy and FLOPs. 

## Usage

To reproduce this work, first clone the repository 
```
git clone https://github.com/irisdaniaj/model_merging.git
```
move to the repository 
```
cd model_merging
```
and create the conda environment and install the requirements
```
conda create --name myenv python=3.11.9
conda activate myenv
pip install requirements.txt
```
navigate to the scripts folder 
```
cd src
```
and run data_download.py to download tha datasets
```
python data_download.py
```
and to preprocess them run 
```
python prepare_data.py
```
Now, run model_download.py to download the models(BERT Base uncased and TinyBERT uncased)
```
python model_download.py
```
We will now fine-tune TinyBERT on two datasets([Stanford Sentiment Treebank (SST-2)](https://huggingface.co/datasets/stanfordnlp/sst2) and [Recognizing Textual Entailment (RTE)](https://metatext.io/datasets/recognizing-textual-entailment-(rte))) and save the corresponding hyperparameters and model metrics in `training_args.json` and `metrics.json` files, respectively, for each dataset. This process will create three new subfolders within the `models/tinybert` directory, each named after a specific dataset (sst2, rte). Each subfolder will contain the fine-tuned model, the tokenizer, and the associated configuration files, allowing for easy access and reproducibility.
```
python finetune_tinybert.py
```
Now the same will be done for BERT Base.

```
python finetune_bert.py
```
Now that we have the models we can merge them. A weigthed based merging technique will be used where the weights of the two models were combined by optimizing merging coefficients using random search. For more detail about the optimization method please refer to section of `report.pdf`. To merge the models run 
```
python optimization.py
```
This will create a new folder `models/merged_model` and under `models/merged_model/{dataset_name}/best_model` the best merged model will be saved. 

Next we want to compare the performance of the finetuned BERT Base model on the [Internet Movie Database (IMDb)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) datasets. To do this run 
```
python evaluate_bert.py
```
and 
```
python evaluate_merging.py
```
the results will be saved in `.json` format in the `results` folder. 


## Hardware Requirements 

All experiments were conducted on a DGX A100 Architecture, which consists of 8 nodes, each with 256 CPU cores, 1 TB of memory, and 8 NVIDIA A100 GPUs, each providing 40 GB of GPU memory. If your system has less computing power or memory, consider using a dedicated computing cluster or cloud-based resources to ensure efficient and effective fine-tuning.



