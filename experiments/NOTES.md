# Molecular transformer

Code for the ... paper ().

## Install requirements

We suggest you to create a new conda environment:

```bash
conda create -n mol_transformer python=3.5
source activate mol_transformer
conda install rdkit -c rdkit
conda install future six tqdm 
pip install torchtext=0.3.1
```

The code was tested for pytorch 0.4.1, to install it go on https://pytorch.org/get-started/locally/.
Select the right operating system and CUDA version and run the command, e.g.:

```bash
conda install pytorch torchvision -c pytorch
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
All the molecules are canonicalized using RDKit (link).


In the experiments we use two open-source datasets.

* **MIT** dataset: ... (link)
* **STEREO** dataset: ... (link)

Both are subsets from the data published by Daniel Lowe

We use two different flavors:

* **separated** reactants and reagents, where the molecule that contribute atoms to the products (according to the 
atom-mapping) are weakly separated by a `>` token.
* **mixed** reactants and reagents, where no difference is made between molecules that contribute atoms to the products. 
Hence, the network has to learn, which would be the reactants for a given input. (**challenging**)

What is atom-mapping? 

Atom-mapping basically shows 

Atom-mapping is usually ill-defined and even in the description of the dataset published  

## Data augmentation

In the datasets sets ending with `_augm`, the number of training datapoints was doubled.
We did this by adding a copy of every reaction in the training set, where the canoncialized molecules 
were by a random equivalent SMILES.

By the time of writing the function to generate those random equivalent SMILES was only available in the master branch of RDKit, e.g.:

```python
from rdkit import Chem
smi = ''
random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))
```

## Input file generation

Use the preprocessing.py script in the root folder:

```bash
dataset=MIT # MIT / STEREO
flavor=mixed # mixed / separated / mixed_augm / separated_augm
python preprocess.py -train_src data/rxn/${dataset}/${flavor}/src-train.txt \
                     -train_tgt data/rxn/${dataset}/${flavor}/tgt-train.txt \
                     -valid_src data/rxn/${dataset}/${flavor}/src-val.txt \
                     -valid_tgt data/rxn/nor_trans_random_80k/tgt-val.txt \
                     -save_data data/${dataset}_${flavor} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
```

We use a vocabulary shared between `sources` and `targets`. 
The `vocab_size` and `seq_length` are chosen to include the whole datasets.


## Training

Our baseline models are trained for 230k steps (which corresponded to ~24h on a single GPU):

```bash
python -u train.py -data data/separated/MIT_separated_augm -save_model available_models/separated/MIT_separated_augm_model \
        -layers 4 -rnn_size 256 -word_vec_size 256   \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 230000  -save_checkpoint_steps 10000 -max_generator_batches 32 -dropout 0.1 -keep_checkpoint 2\
        -share_embeddings -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 4 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.0  -seed 42 -report_every 1000 -gpu_ranks 0
```