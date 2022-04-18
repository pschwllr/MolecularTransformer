# Molecular Transformer

This is the code for the "Molecular Transformer: A Model for Uncertainty-Calibrated Chemical Reaction Prediction" paper published in [ACS Central Science](https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00576).

The preprint presented at the ML for Molecules Materials workshop at Neurips 2018 can be found on [ChemRxiv](https://chemrxiv.org/articles/Molecular_Transformer_for_Chemical_Reaction_Prediction_and_Uncertainty_Estimation/7297379).

To implement our models we were based on [OpenNMT-py (v0.4.1)](http://opennmt.net/OpenNMT-py/).

A trained model was made available online through a graphical user interface on the [IBM RXN for Chemistry](https://rxn.res.ibm.com) platform.


## Install requirements

Create a new conda environment:

```bash
conda create -n mol_transformer python=3.5
source activate mol_transformer
conda install rdkit -c rdkit
conda install future six tqdm pandas
```

The code was tested for pytorch 0.4.1, to install it go on [Pytorch](https://pytorch.org/get-started/locally/).
Select the right operating system and CUDA version and run the command, e.g.:

```bash
conda install pytorch=0.4.1 torchvision -c pytorch
```
Then,
```bash
pip install torchtext==0.3.1
pip install -e . 
```


## Pre-processing 

We us the following regular expression to separate SMILES into tokens:

```python

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

```

During the pre-processing, if the exact same molecule appears on the product, as well as on the reactant side, we remove it from the product side. 
All the molecules are canonicalized using [RDKit](http://www.rdkit.org).

In the experiments we use two open-source datasets (and train/valid/test splits).

* [**USPTO_MIT** dataset](https://github.com/wengong-jin/nips17-rexgen) USPTO/data.zip
* [**USPTO_STEREO** dataset](https://ibm.box.com/v/ReactionSeq2SeqDataset) US_patents_1976-Sep2016_*

Both are subsets from data extracted and originally published by Daniel Lowe (many thanks for that!).

We use two different preprocessing methods:

* **separated** reactants and reagents, where the molecule that contribute atoms to the products (according to the 
atom-mapping) are weakly separated by a `>` token, e.g.: `COc1c(C)c(C)c(OC)c(C(CCCCC\#CCCO)c2ccccc2)c1C>C.CCO.[Pd]`
* **mixed** reactants and reagents, where no distinction is made between molecules that contribute atoms to the products and those that don't. 
Hence, the network has to learn, which would be the reactants for a given input (**more correct, but also more challenging**), e.g.: `C.CCO.COc1c(C)c(C)c(OC)c(C(CCCCC\#CCCO)c2ccccc2)c1C.[Pd]`


The tokenized datasets can be found [here](https://ibm.box.com/v/MolecularTransformerData)

And are best placed into the `data/` folder. 

## Data augmentation

In the datasets ending with `_augm`, the number of training datapoints was doubled.
We did this by adding a copy of every reaction in the training set, where the canoncalized source molecules 
were replaced by a random equivalent SMILES.

By the time of writing the function to generate those random equivalent SMILES was only available in the master branch of RDKit, e.g.:

```python
from rdkit import Chem
smi = ''
random_equivalent_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)
```

## Input file generation

Use the preprocessing.py:

```bash
dataset=MIT_mixed_augm # MIT_mixed_augm / STEREO_mixed_augm
python preprocess.py -train_src data/${dataset}/src-train.txt \
                     -train_tgt data/${dataset}/tgt-train.txt \
                     -valid_src data/${dataset}/src-val.txt \
                     -valid_tgt data/${dataset}/tgt-val.txt \
                     -save_data data/${dataset}/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```

We use a shared vocabulary. The `vocab_size` and `seq_length` are chosen to include the whole datasets.

## Training

Our MIT models were trained for 48 hours on a single GPU (STEREO for 72h), using the following hyperparameters: 

```bash
dataset=MIT_mixed_augm # MIT_mixed_augm / STEREO_mixed_augm

python  train.py -data data/${dataset}/${dataset} \
                   -save_model experiments/checkpoints/${dataset}/${dataset}_model \
                   -seed 42 -gpu_ranks 0 -save_checkpoint_steps 10000 -keep_checkpoint 20 \
                   -train_steps 500000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
                   -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
                   -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
                   -learning_rate 2 -label_smoothing 0.0 -report_every 1000 \
                   -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
                   -dropout 0.1 -position_encoding -share_embeddings \
                   -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
                   -heads 8 -transformer_ff 2048
                                
```

To achieve the best results with single models, we average the last 20 checkpoints. 


## Testing
 
Models and results can be found [here](https://ibm.box.com/v/MolecularTransformerModels).

To generate the predictions use the `translate.py` script: 

```bash

model=${dataset}_model_average_20.pt

python translate.py -model experiments/models/${model} \
                    -src data/${dataset}/src-test.txt \
                    -output experiments/results/predictions_${model}_on_${dataset}_test.txt \
                    -batch_size 64 -replace_unk -max_length 200 -fast
```

## Evaluate predictions

Run the following script to get the top-5 accuracy.

```bash
python score_predictions.py -targets data/${dataset}/tgt-test.txt \
                    -predictions experiments/results/predictions_${model}_on_${dataset}.txt

```


## IBM RXN for Chemistry

The chemical domain on which the Molecular Transformer is able to make accurate predictions on is limited by the training data. 

Since August 2018, a model trained on more diverse chemical reactions (also containing closed-source data) 
is freely available through a graphical user interface on the [IBM RXN for Chemistry](https://rxn.res.ibm.com) platform.
 
So far, it has been used by several thousand organic chemists worldwide for performing more than 10,000 chemical reaction predictions. 
The hope is that by making such AI models easily accessible, we will motivate organic chemists to incorporate them into their daily tasks. 
The overarching goal is not only to accelerate the discovery and synthesis of novel molecules, 
but also to develop more application oriented AI models to assist humans in the real world. 

## Citation

Our paper can be cited as follows: 

```
@article{schwaller2019molecular,
  title={Molecular transformer: A model for uncertainty-calibrated chemical reaction prediction},
  author={Schwaller, Philippe and Laino, Teodoro and Gaudin, Th{\'e}ophile and Bolgar, Peter and Hunter, Christopher A and Bekas, Costas and Lee, Alpha A},
  journal={ACS central science},
  volume={5},
  number={9},
  pages={1572--1583},
  year={2019},
  publisher={ACS Publications}
}
```

If you find the code useful, please do not forget to also cite the underlying framework:

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
