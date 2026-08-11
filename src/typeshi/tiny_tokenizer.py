"""Char-level tokenizer for the tiny motor PoC (spec 2026-08-10, §3).

Grammar tokens are single vocabulary entries, mirroring prepare_tokenizer()
on Qwen; <TARGET> text encodes one char per token, which turns target-copying
into a monotonic char->key attention pattern. Built programmatically -- no
vocab files to maintain."""

from __future__ import annotations

from typeshi.serialize import special_tokens

# Printable ASCII 0x20-0x7E plus newline and tab: the 97 characters that can
# appear in prompt text. Everything else must RAISE at encode time (the build
# gates typed chars but never the target sentence -- spec §3 OOV containment).
TEXT_CHARS = [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t"]

EOS = "<EOS>"
PAD = "<PAD>"


def build_tiny_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(TEXT_CHARS)}
    # No unk_token, deliberately: WordLevel then raises on out-of-vocab input
    # instead of silently mapping it. Wiring an unk would violate spec §3.3.
    inner = Tokenizer(models.WordLevel(vocab))
    # [\s\S] matches any character including newline; every char becomes its own piece.
    inner.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    # The default decode joins word-level tokens with spaces; deserialize()
    # rejects stray whitespace by design, so decode must fuse byte-exactly.
    inner.decoder = decoders.Fuse()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=inner,
        eos_token=EOS,
        pad_token=PAD,
        clean_up_tokenization_spaces=False,
    )
    tok.add_special_tokens({"eos_token": EOS, "pad_token": PAD})
    whole = [t for t in special_tokens() if t.endswith(">")]
    tok.add_tokens(whole, special_tokens=True)
    return tok
