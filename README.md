# How Tokenization Works

This sample demonstrates a core fact about LLMs: **models work with token sequences (numerical IDs), not raw text.**

Before any model can process text, a *tokenizer* splits it into pieces (tokens) and converts each piece into an integer ID. The model itself only ever sees these IDs — it has no direct concept of characters or words.

## Files

- [tokenization_demo.py](tokenization_demo.py) — the runnable demo
- [requirements.txt](requirements.txt) — dependencies (`tiktoken`)

## Running it

```bash
pip install -r requirements.txt
python tokenization_demo.py
```

Example output:

```
Input text:  'Hello world!'
Token IDs:   [9906, 1917, 0]
    9906  ->  'Hello'
    1917  ->  ' world'
       0  ->  '!'

Predicted next token ID: 9907 ('vt')
```

## The flow

The script mirrors these four steps:

1. **Input text or code** — the raw string a user or application provides.
2. **Tokenizer splits content** — the text is broken into sub-word pieces (tokens).
3. **Tokens become IDs** — each piece is mapped to an integer ID from the tokenizer's vocabulary.
4. **Model predicts the next token** — the model consumes the ID sequence and outputs the ID of the next token.

## Code walkthrough

### `Tokenizer` class

```python
class Tokenizer:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)
```

Wraps [`tiktoken`](https://github.com/openai/tiktoken), the tokenizer library used by OpenAI's models. `cl100k_base` is the encoding used by GPT-3.5/GPT-4-era models.

- `encode(text)` — turns a string into a list of integer token IDs. This is the `tokenizer.encode(input_text)` call from the slide.
- `decode(token_ids)` — the reverse operation, turning IDs back into text. It's not on the slide, but it's what lets the demo print a human-readable piece next to each ID.

### `Model` class

```python
class Model:
    def predict(self, token_ids: list[int]) -> int:
        return max(token_ids) + 1
```

A **toy stand-in** for a real language model. Real models predict the next token ID using a trained neural network over the vocabulary; that requires model weights this sample doesn't have. Here, `predict` just returns `max(token_ids) + 1` so the code compiles and runs while preserving the important part of the slide: the method takes a list of IDs in and returns a single ID out — nothing about text ever touches the model.

### `main()`

```python
input_text = "Hello world!"

token_ids = tokenizer.encode(input_text)
```

Step 1 and steps 2–3 from the slide: raw text goes in, a list of token IDs comes out (e.g. `[9906, 1917, 0]` for `"Hello world!"`).

```python
for token_id in token_ids:
    piece = tokenizer.decode([token_id])
    print(f"  {token_id:>6}  ->  {piece!r}")
```

Not part of the original slide snippet — added so you can see which text fragment each ID actually corresponds to (e.g. `9906` is `'Hello'`, `1917` is `' world'` with a leading space, `0` is `'!'`). This makes the abstract "numbers instead of text" idea concrete.

```python
next_token_id = model.predict(token_ids)
next_token_text = tokenizer.decode([next_token_id])
```

Step 4: the model consumes the ID sequence and predicts the next token's ID. The demo decodes it back to text purely for display purposes — the model itself never deals with text, only IDs.

## Key takeaway

Everything a model does — reading a prompt, generating a reply — happens over integer IDs. Text is only ever text at the very edges of the pipeline: when the user types it in, and when the tokenizer decodes IDs back into text for the user to read.
