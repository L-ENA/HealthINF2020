# HealthINF2020
Code and supplementary material for the HealthINF conference paper

**Original data sources:**

Sentence data: https://github.com/jind11/PubMed-PICO-Detection

Squad data: https://rajpurkar.github.io/SQuAD-explorer/

Ebm-NLP data: https://github.com/bepnye/EBM-NLP

**Fine-tuning question answering information:**

Fine-tuning can be carried out via the provided Google Colaboraty notebooks (NB enable GPU runtime for full training speed! And a Google accound is needed in order to connect the Drive and save the weights). Please note that all other pre-processing (indicated by .py script), as well as conversion to Squad format, was carried out on a local machine.

The Question answering model was built on a previous version of the Transformers library. This library is constantly changing because its maintainers are very active, which means that errors during training could start showing up at any time. Therefore, all colab scripts have the option to import the library from my local fork. This means that the scripts should run/ be reproducible - just not in the latest version of the Transformers library. 

**Changes after the conference paper was submitted: Question answering**

The amazing people who maintain Transformers give access to many pretrained models: https://huggingface.co/transformers/pretrained_models.html 
In general, this script now produces evaluation results for recall and precision, not only for the testing data as a whole, but also for each individual class (on a token level, it gives extra recall scores for sentences that contained a label, and for sentences that did not contain a PICO, and for the "combined" version: all sentences in this set of testing data).

New best scores for "P(opulation)" class data:


Pretrained model | Under-sampling dominant class | F1 combined | F1 P only | Recall combined | Recall P only
--- | --- | ---| --- | --- | ---
bert-base-uncased (from paper) | na | 87| 74 | na | na
bert-base-uncased | 40% | 86.94| 79.55 | 87.6 | 81.77
bert-large-uncased | 40% | **86.98**| **80.16** | **87.72** | **82.66**


"bert-large-uncased-whole-word-masking-finetuned-squad" performed marginaly worse than bert-large-uncased as base model, possibly because it has already been fine-tuned on too much different data - see colab training file for more details


**Scripts for question answering:**
pytorchSquad.ipynb - fine-tune the question answering model via the Google colab script
predictSquad.ipynb - evaluate your fine-tuned question answering model via the Google colab script

**Scripts for sentence classification:**
MultilangPICObert.ipynb - fine-tune the multilingual or scibert or the Google Bert models. Weights for the pretrained models are available from https://github.com/allenai/scibert and https://github.com/google-research/bert

**Please do get in touch if you have any questions/ suggestions /comments !** 


