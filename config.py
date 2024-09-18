from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12
)

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
