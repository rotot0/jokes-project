# jokes-project
Humor detection, humor generation
# Overview
This work was aimed to explore the ruGPT3 and ruBERT models in humor detection and humor generation.
# Dependecies
* python3.7
* torchtext==0.8.1
* transformers

# Results
We tested [ruBERT](https://huggingface.co/DeepPavlov/rubert-base-cased)-based and [ruGPT3Large](https://github.com/sberbank-ai/ru-gpts)-based classifiers in the [FUN dataset](https://www.aclweb.org/anthology/P19-1394/). Results compared to the authors' models can be seen below.

| model        | F1-score test | F1-score gold |
|--------------|---------------|---------------|
| SVM baseline | 0.798         | 0.803         |
| ULMFun       | 0.907         | 0.890         |
| ruBERT       | 0.91          | 0.886         |
| ruGPT3Large  | 0.936         | 0.901         |

We also constructed our own dataset (see `data` folder) and did the same classification.

| model        | accuracy | F1      |
|--------------|----------|---------|
| SVM baseline | 0.964    | 0.956   |
| ruBERT       | 0.998    | 0.997   |
| ruGPT3Large  | 0.997    | 0.997   |

You can find some classification models in `notebooks/bert_classification.ipynb`, `notebooks/gpt_classification.ipynb`.  

Then we fine-tuned ruGPT3Small and ruGPT3Large models on our jokes dataset. You can find code for fine-tuning and generation in `notebooks/gpt_tuning.ipynb`. Tuned models: [https://mega.nz/fm/MYEkDBYR]. Overall, the quality of the generated jokes is poor, but sometimes you can obtain interesting and funny examples. After that, we generated some jokes and built discriminative ruBERT classifier that distinguish between generated and real jokes (see `data/jokes_generated.csv`). Although ruBERT performed well in the task of distinguishing jokes, this method is bad for finding good generated jokes.

Finally, we trained GAN with tuned ruGPT3Small generator and [RelGAN](https://arxiv.org/abs/1908.07269)'s discriminator. Also we used Gumbel-Max trick to deal with non-differentiable issue. GAN only decreased the quality of the generator (but we did very few experiments).
