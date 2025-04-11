import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Pastikan d_out (dimensi output) dapat dibagi habis oleh num_heads (jumlah kepala).
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Simpan dimensi output, jumlah kepala, dan dimensi setiap kepala dalam variabel kelas.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Hitung dimensi setiap kepala (d_out dibagi jumlah kepala).

        # Definisikan layer linear untuk membentuk query, key, dan value dari input x.
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Layer linear untuk membentuk query.
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # Layer linear untuk membentuk key.
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Layer linear untuk membentuk value.

        # Layer linear untuk memproyeksikan ulang keluaran gabungan dari semua kepala menjadi dimensi d_out.
        self.out_proj = nn.Linear(d_out, d_out)

        # Layer dropout untuk regularisasi, dengan nilai dropout yang ditentukan oleh parameter.
        self.dropout = nn.Dropout(dropout)

        # Buffer mask untuk menghindari attention pada next token.
        # Menggunakan matriks segitiga atas (upper triangular matrix) untuk masking kausal.
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # Ambil ukuran batch (b), jumlah token (num_tokens), dan dimensi input (d_in) dari bentuk input x.
        b, num_tokens, d_in = x.shape

        # Bentuk key, query, dan value dari input x dengan layer linear yang telah didefinisikan.
        keys = self.W_key(x) # Bentuk tensor: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Mengubah tensor keys, queries, dan values menjadi 4 dimensi:
        # Dari bentuk (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose tensor agar dimensi num_heads dan num_tokens ditukar.
        # Dari (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Hitung skor perhatian (attention scores) menggunakan dot product antara queries dan keys.
        # Hasil dot product memiliki bentuk (b, num_heads, num_tokens, num_tokens).
        attn_scores = queries @ keys.transpose(2, 3)

        # Ambil mask yang hanya mencakup bagian input yang ada (berdasarkan num_tokens).
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Gunakan mask untuk mengisi skor perhatian yang tidak valid dengan -inf.
        # Ini memastikan model tidak memperhatikan token masa depan.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normalisasi skor perhatian menggunakan softmax dan skala dengan dimensi terakhir.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Terapkan dropout pada bobot perhatian (attn_weights) untuk regularisasi.
        attn_weights = self.dropout(attn_weights)

        # Hitung vektor konteks dengan melakukan operasi dot product antara attn_weights dan values.
        # Bentuk tensor setelah perkalian adalah (b, num_heads, num_tokens, head_dim).
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Gabungkan hasil dari setiap kepala menjadi satu tensor dengan dimensi d_out.
        # Bentuk tensor menjadi (b, num_tokens, d_out).
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Terapkan layer linear untuk memproyeksikan ulang vektor konteks gabungan.
        context_vec = self.out_proj(context_vec)

        # Kembalikan vektor konteks yang sudah diproyeksikan sebagai output.
        return context_vec
