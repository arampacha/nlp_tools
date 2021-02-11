"""
This script can be used to generate backtranslations using pretrained models by Helsinki NLP group from huggingface model hub.
"""

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from fastcore.all import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_models(lang1, lang2, device=None):
    device = ifnone(device, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    tok1 = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang1}-{lang2}")
    tok2 = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{lang2}-{lang1}")
    fwd = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{lang1}-{lang2}").to(device)
    bwd = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{lang2}-{lang1}").to(device)
    return tok1, tok2, fwd, bwd

class Backtranslator:
    """
    Backtranslation generator for text data augmentation
    Uses pretrained models by Helsinki NLP group
    """
    def __init__(self, source_lang, trans_lang, device=None):
        self.device = ifnone(device, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tok1, self.tok2, self.fwd, self.bwd = get_models(source_lang, trans_lang, self.device)

    def do_one(self, texts, num_beams=1):
        gen_kwargs = {
            'do_sample':True,
            'top_p':0.9,
            'repetition_penalty':1.5
        }
        input_ids = self.tok1.batch_encode_plus(texts, return_tensors='pt', padding=True, max_length=512, truncation=True).input_ids
        output_ids = self.fwd.generate(input_ids.to(self.device), num_beams=num_beams, **gen_kwargs)
        res_ids = self.bwd.generate(output_ids, num_beams=num_beams, **gen_kwargs)
        return [self.tok2.decode(ids.detach().cpu(), skip_special_tokens=True).strip() for ids in res_ids]

    def  generate(self, df, text_col=1, bs=32, num_beams=1):
        res = []
        for idx in tqdm(np.array_split(df.index.to_numpy(), int(np.ceil(len(df)/bs)))):
            texts = df.iloc[idx, text_col].to_list()
            res.extend(self.do_one(texts, num_beams=num_beams))
        return pd.DataFrame({'text':res})

def generate_backtranslations(input_fn, lang1:str, lang2:str, output_fn=None, text_col:int=1, num_beams:int=1, bs:int=16, device=None):
    df = pd.read_csv(input_fn)
    btr = Backtranslator(lang1, lang2, device)
    btr_df = btr.generate(df, text_col=text_col, bs=bs, num_beams=num_beams)
    if output_fn: btr_df.to_csv(output_fn)
    return btr_df

@call_parse
def run_backtranslate(
    input_fn:Param(help='Input text file', type=str),
    lang1:Param(help='Source language', type=str),
    lang2:Param(help='Language for backtranslation', type=str),
    output_fn:Param(help='Output text file', type=str, default=''),
    num_beams:Param(help='Number of beams for translation generation', type=int, default=1),
    bs:Param(help='batch size', type=int, default=32),
    use_gpu:Param(help='When True will use GPU if available', type=bool_arg, default=True)):
    
    device = torch.device('cuda' if (torch.cuda.is_available() and use_gpu) else 'cpu')
    if output_fn == '':
        output_fn = f"{Path(input_fn).name.split(sep='.')[0]}_btr_{lang1}-{lang2}.csv"
    generate_backtranslations(input_fn, lang1, lang2, output_fn, num_beams=num_beams, bs=bs, device=device)
