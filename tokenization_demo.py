"""
Demo: models work with token sequences (numerical IDs), not raw text.

Mirrors the flow:
    text -> tokenizer.encode() -> token_ids -> model.predict() -> next_token_id
"""

import tiktoken


class Tokenizer:
    """Thin wrapper around tiktoken to match the slide's tokenizer.encode() API."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)


class Model:
    """Toy stand-in for an LLM: predicts the next token ID from a sequence of IDs.

    A real model would run the IDs through its neural network; here we just
    pick the highest ID seen so far + 1, to show the shape of the interface
    (IDs in, ID out) without needing model weights.
    """

    def predict(self, token_ids: list[int]) -> int:
        return max(token_ids) + 1


def main() -> None:
    tokenizer = Tokenizer()
    model = Model()

    # 1. Input text or code
    input_text = "Hello world!"

    # 2 & 3. Tokenizer splits content and converts tokens to IDs
    token_ids = tokenizer.encode(input_text)
    print(f"Input text:  {input_text!r}")
    print(f"Token IDs:   {token_ids}")

    # Show each token alongside its ID for clarity
    for token_id in token_ids:
        piece = tokenizer.decode([token_id])
        print(f"  {token_id:>6}  ->  {piece!r}")

    # 4. Model predicts the next token based on the IDs
    next_token_id = model.predict(token_ids)
    next_token_text = tokenizer.decode([next_token_id])
    print(f"\nPredicted next token ID: {next_token_id} ({next_token_text!r})")


if __name__ == "__main__":
    main()
