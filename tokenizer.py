from transformers import GPT2TokenizerFast
from tokenizers import decoders, pre_tokenizers, processors, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from get_data import load_data

def iterator_dset(dataset):
    for mydataset in [dataset]:
        for i, data in enumerate(mydataset):
            if isinstance(data.get("text", None), str):
                yield data["text"]

def train_tokenizer(vocab_size, iterator):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    tokenizer.train_from_iterator(iterator, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)

    return tokenizer


if __name__=="__main__":
    document = load_data()
    vocab_size = 32000
    trained_tokenizer = train_tokenizer(vocab_size=vocab_size, iterator=iterator_dset(document))
    trained_tokenizer.save_pretrained(f"tokenizer-trained-{vocab_size}")