#  Model Merging: Enhancing NLP Task Performance by Integrating Fine-Tuned Small Models with BERT Base


In the rapidly evolving field of Natural Language Processing (NLP), the quest for building more efficient and effective models is continuous. Traditional approaches often involve fine-tuning pre-trained models, like [BERT Base](https://huggingface.co/google-bert/bert-base-uncased), to achieve state-of-the-art performance on various NLP tasks such as sentiment analysis, question answering, etc. However, the computational costs and resource demands associated with fine-tuning these big models can be prohibitive, especially in real-world applications where efficiency is crucial.

Recent advancements have introduced smaller, more efficient models like [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D), which are designed to retain much of the performance of their larger counterparts while significantly reducing computational requirements. These models have opened up new possibilities for creating lightweight NLP solutions. Yet, the question remains: can the strengths of these smaller models be effectively combined with the power of large pre-trained models to further enhance performance?

This project seeks to explore this possibility by evaluating whether merging a fine-tuned small model with a large pre-trained model can improve performance on specific NLP tasks, compared to the traditional approach of fine-tuning the large model alone. By leveraging the [GLUE benchmark](https://gluebenchmark.com/), which is widely recognized for benchmarking various NLP tasks, this study aims to provide insights into the potential benefits and trade-offs of model merging in practical applications.

The primary focus will be on determining whether a hybrid approach can offer a superior balance of accuracy and resource utilization compared to the more classical approach. To do this we will use BERT Base as the large model and TinyBERT as the lightweight model and we will compare their performance on three different GLUE datasets: [SST-B](https://paperswithcode.com/dataset/sts-benchmark), [SST-2](https://huggingface.co/datasets/gimmaru/glue-sst2), [RTE](https://paperswithcode.com/dataset/rte) in terms of accurancy and FLOPs. 

## Usage

To reproduce this work, first clone the repository 
```
git clone https://github.com/irisdaniaj/model_merging.git
```
move to the repository 
```
cd model_merging
```
and install the requirements
```
conda create --name myenv --file requirements.txt
```
navigate to the scripts folder 
```
cd scripts
```
and run data_download.py to download tha datasets
```
python data_download.py
```
Now, run model_download.py to download the models(BERT Base uncased and TinyBERT uncased)
```
python model_download.py
```
We will now fine-tune TinyBERT on each dataset and save the corresponding hyperparameters and model metrics in `training_args.json` and `metrics.json` files, respectively, for each dataset. This process will create three new subfolders within the `models/tinybert` directory, each named after a specific dataset (stsb, sst2, rte). Each subfolder will contain the fine-tuned model, the tokenizer, and the associated configuration files, allowing for easy access and reproducibility.
```
python finetune_tinybert.py
```
Now the same will be done for BERT Base.(disclaimer run in on cluster) 

```
python finetune_bert.py
```

## Hardware Requirements 

For fine-tuning TinyBERT, I used an AMD Ryzen 7 7840HS processor with Radeon 780M integrated graphics, featuring 8 cores and 16 threads, along with 32 GB of RAM. Depending on your hardware setup, you may need to adjust the training parameters, such as batch size or learning rate. If your system has less computing power or memory, consider using a dedicated computing cluster or cloud-based resources to ensure efficient and effective fine-tuning.




## TODO

- [ ] fine tune bert base on the data and save it --> save hyperparamters and wieghts
- [ ] merge the finetuned tinybert with pretrained unfinetuned bert base --> save hyperparameters and wieghts(how the hell do i merge them?)
- [ ] run infernce on datasets using bert base
- [ ] run inference on datasets using merge model
- [ ] comprare the two models (comparison in terms of metrics and flops (are flops the most suitable choice here?))
- [ ] remember to create the environment .txt
- [ ] In the report justify why I choose BERT Base and TinyBERT. also table with number of paramters of the BERT models that i can find.
- [ ] justify why I choose those datasets. 

