#!/usr/bin/env python
import argparse
import torch

from onmt.utils.masking import ChemVocabMask

def make_rxn_vocab_mask(vocab, output):

    shared_vocab = torch.load(vocab)[1][1]
    mask = ChemVocabMask(vocab=shared_vocab)
    mask.save_dicts(output)



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-vocab", "-v",  required=True,
                        help="Shared vocab file to analyse")
    parser.add_argument("-output", "-o", required=True,
                        help="Output file")
    opt = parser.parse_args()



    make_rxn_vocab_mask(opt.vocab, opt.output)


if __name__ == "__main__":
    main()