import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # Memasukkan semua parameter dari config "cfg"
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block
        shortcut = x      # Menyimpan variabel untuk skip/shortcut connection
        x = self.norm1(x) # Normalisasi
        x = self.att(x)   # Attention
        x = self.drop_shortcut(x) # Dropout
        x = x + shortcut  # Add / skip connection

        # Feed-forward block
        shortcut = x      # Menyimpan variabel untuk skip/shortcut connection
        x = self.norm2(x) # Normalisasi
        x = self.ff(x)    # Feed-forward
        x = self.drop_shortcut(x) # Dropout
        x = x + shortcut  # Add / skip connection

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        '''
        Mengulang transformer block sebanyak yang diinginkan, contohnya n_layers
        sebanyak 12, maka TransformerBlock diduplikasi sebanyak 12 kali kemudian
        disimpan di trf_blocks untuk dipanggil nantinya
        '''
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Embedding kata dan posisi
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Transformer blocks sebanyak n_layers
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        # Neural network dengan output node sebanyak vocab_size
        logits = self.out_head(x)
        return logits