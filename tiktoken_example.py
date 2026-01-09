import tiktoken

# 1. Define your vocabulary (ranks)
# Keys must be bytes. Values must be integers (ranks).
# In a real scenario, this would be thousands of items.
mergeable_ranks = {
    b"h": 0,
    b"e": 1,
    b"l": 2,
    b"o": 3,
    b"hell": 4,
    b"hello": 5,
    b" world": 6,
}

# 2. Define special tokens
# Ensure these IDs do not clash with the ranks above.
special_tokens = {
    "<|endoftext|>": 1001,
    "<|custom_pad|>": 1002,
}

# 3. Choose a regex pattern
# This is the standard regex used in GPT-4 (cl100k_base).
# It defines how text is split before BPE is applied.
pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# 4. Initialize the custom Encoding
enc = tiktoken.Encoding(
    name="my_custom_encoding",
    pat_str=pat_str,
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens
)

# 5. Test it
text = "hello world<|endoftext|>"
tokens = enc.encode(text, allowed_special="all")

print(f"Tokens: {tokens}")
# Output will depend on your specific ranks, but it will use your mappings.
print(f"Decoded: {enc.decode(tokens)}")