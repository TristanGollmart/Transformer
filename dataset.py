import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')]).type(torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')]).type(torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')]).type(torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.trg_lang]

        # tokenize text
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_trg.encode(trg_text).ids

        # pad to have fixed sequence length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # taking into account also SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # only SOS for decoder

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            #raise ValueError("Sentence too long for sequence length")
            if enc_num_padding_tokens < 0:
                enc_num_padding_tokens = 0
                enc_input_tokens = enc_input_tokens[:self.seq_len]
            if dec_num_padding_tokens < 0:
                dec_num_padding_tokens = 0
                dec_input_tokens = dec_input_tokens[:self.seq_len]

        # create padded sentence with eos and sos
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # label is shifted decoder input one position to the right -> will preserve causality with self attention mask
        label = torch.cat(
            [
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seqlen) & (1, selen, seqlen)
            "label": label, # seq_len
            "src_text": src_text,
            "trg_text": trg_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0