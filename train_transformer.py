import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from pathlib import Path
from tqdm import tqdm


def get_all_sentences(ds, lang):
    ''' dataset consists of pairs of sentences. Get only part that is in language <lang>'''
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    '''

    :param config: dict with configs
    :param ds: dataset
    :param lang: language
    :return: tokenizer for given language dataset and configurations
    '''

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_trg']}", split="train")

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config["lang_trg"])

    # train validation split
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    # build dataset for stream loading and preprocessing sentences to match model requirements
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_trg, config['lang_src'], config['lang_trg'],
                                config['seq_len'])

    max_len_src = 0
    max_len_trg = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        trg_ids = tokenizer_trg.encode(item['translation'][config['lang_trg']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))

    print(f'maximum source sentence length: {max_len_src}')
    print(f'maximum target sentence length: {max_len_trg}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg

def get_model(config, vocab_src_len, vocab_trg_len):
    model = build_transformer(vocab_src_len, vocab_trg_len, config['seq_len'], config['seq_len'], config['d_model'], N=3)
    return model

def train_model(config):
    device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size()).to(device)

    # visualization
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_state']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)   # (B, 1, 1, SL)
            decoder_mask = batch['decoder_mask'].to(device) # (B,1,SL, SL)

            encoder_output = model.encode(encoder_input, encoder_mask) # (B, SL, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, SL, D_Model)
            proj_out = model.proj_layer(decoder_output) # (B, SL, trg_vocab_size)

            label = batch['label'].to(device) #(B, SL)
            # (B, SL, trg_vocab_size) --> (B* SL , trg_vocab_size)
            loss = loss_fn(proj_out.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # Logging
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # save model checkpoint
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)
