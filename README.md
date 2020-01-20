# HealthINF2020
Code and supplementary material for the HealthINF conference paper

**Original data sources:**

Sentence data: https://github.com/jind11/PubMed-PICO-Detection

Squad data: https://rajpurkar.github.io/SQuAD-explorer/

Ebm-NLP data: https://github.com/bepnye/EBM-NLP

**Fine-tuning question answering information:**

Fine-tuning can be carried out via the provided Google Colaboraty notebooks (NB enable GPU runtime for full training speed!). Please note that all other pre-processing (indicated by .py script), as well as conversion to Squad format, was carried out on a local machine.

The Question answering model was built on a previous version of the Transformers library. This library is constantly changing because its maintainers are very active, which means that errors during training could start showing up at any time. Therefore, all colab scripts have the option to import the library from my local fork. This means that the scripts should run - just not in the latest version. 

**Changes after the conference paper was submitted:**

The amazing people who maintain Transformers have just released the "bert-large-uncased-whole-word-masking-finetuned-squad" model - already finetuned on the whole squad dataset. This model can be used instead of "bert-base-uncased", making it possible to reduce the additional training data for any PICO domain. This increased the F1 score for identifying patients by 1 % (without adjusting additional parameters after migrating from base to large).


**Scripts for question answering:**
pytorchSquad.ipynb - fine-tune the question answering model via the Google colab script
predictSquad.ipynb - evaluate your fine-tuned question answering model via the Google colab script

**Scripts for sentence classification:**
MultilangPICObert.ipynb - fine-tune the multilingual or scibert or the Google Bert models. Weights for the pretrained models are available from https://github.com/allenai/scibert and https://github.com/google-research/bert

**Please do get in touch if you have any questions/ suggestions /comments !** 


