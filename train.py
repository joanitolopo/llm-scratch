import torch
from config import GPT_CONFIG_124M, OTHER_SETTINGS
from transformer import GPTModel
from transformers import GPT2TokenizerFast
from get_data import load_data
from dataloader import create_dataloader_v1
from utils import calc_loss_batch, evaluate_model, generate_and_print_sample, plot_losses
import os
import argparse


torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, path):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        checkpoint_path = os.path.join(path, "model_epoch_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def train(args):
    gpt_config = {
    "vocab_size": 32000,    # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
    }

    # get data
    dataset = load_data()
    dataset = dataset.train_test_split(test_size=args.val_set_size, seed=42)

    # inisiasi model
    model = GPTModel(gpt_config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_loader = create_dataloader_v1(
        dataset["train"]["text"],
        batch_size=args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
        tokenizer_path=args.tokenizer_path
    )

    val_loader = create_dataloader_v1(
        dataset["test"]["text"],
        batch_size=args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
        tokenizer_path=args.tokenizer_path
    )

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=5, eval_iter=1,
        start_context="Lu ada bikin apa disana?", tokenizer=tokenizer,
        path=args.model_path
    )

    epochs_tensor = torch.linspace(0, args.num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=5e-4, required=True)
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epoch to train', default=1)
    parser.add_argument('--batch_size', type=int, required=True, help='Number of batch size use in dataloader', default=4)
    parser.add_argument('--weight_decay', type=float, required=True, help='Number of weight decay', default=0.1)
    parser.add_argument('--val_set_size', type=float, help='How many data use in validation dataset', default=0.2)
    parser.add_argument('--tokenizer_path', type=str, help='Path to custom tokenizer', default="tokenizer-32000")
    parser.add_argument('--model_path', type=str, help='Path to saved model', default="model")

    args = parser.parse_args()

    train(args)
