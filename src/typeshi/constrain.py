"""Grammar-constrained decoding for transcription generation.

The 0.8B shakedown showed the dominant failure mode (64% of attempts) was
sampling one token outside the 12.8k event-token island of a 248k-entry
vocabulary and cascading into base-model text, never to return. The design
spec plans constrained decoding regardless; this is the transcription subset.

The v2 stream grammar for transcription is a two-state machine:

    EVENT state: a <c:h> or <BKSP:h> token (or EOS, once something was typed)
    GAP state:   a <DT:k> token

    EVENT -> GAP -> EVENT -> GAP -> ...

Composition additions (<CUR:>, <SELDEL:>) emit plain-text digits and are NOT
representable under this mask -- extend the state machine before using it for
composition generation.
"""

from __future__ import annotations

from typeshi.serialize import special_tokens


def _token_ids(tok, predicate) -> list[int]:
    ids = []
    for t in special_tokens():
        if not t.endswith(">") or not predicate(t):
            continue
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            ids.append(i)
    return ids


class TranscriptionGrammarProcessor:
    """LogitsProcessor: only grammar-legal tokens are sampleable.

    Alternates between event tokens (<c:h>, <BKSP:h>) and gap tokens (<DT:k>);
    EOS becomes legal in event position once at least one event was emitted,
    so the stream can end but never ends on a dangling <DT:>.
    """

    def __init__(self, tok, prompt_len: int) -> None:
        import torch

        self.prompt_len = prompt_len
        is_dt = lambda t: t.startswith("<DT:")  # noqa: E731
        is_event = lambda t: (  # noqa: E731
            not t.startswith(("<DT:", "<MODE:", "<WPM:", "<ECOR:", "<EUNC:",
                              "<REV:", "<TARGET", "<WRITTEN", "<PROCESS"))
        )
        self.dt_ids = torch.tensor(_token_ids(tok, is_dt))
        self.event_ids = torch.tensor(_token_ids(tok, is_event))
        eos = tok.eos_token_id
        self.eos_ids = torch.tensor([eos] if eos is not None else [])

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids.shape[1] - self.prompt_len
        mask = torch.full_like(scores, float("-inf"))
        if generated % 2 == 0:
            # Event position. EOS only after the stream is non-empty.
            allowed = self.event_ids
            if generated > 0 and self.eos_ids.numel():
                allowed = torch.cat([allowed, self.eos_ids])
        else:
            allowed = self.dt_ids
        mask[:, allowed.to(scores.device)] = 0
        return scores + mask


class GumbelSampleProcessor:
    """Exact softmax sampling without torch.multinomial.

    On MPS, torch.multinomial can sample tokens whose probability is exactly
    zero -- caught in the act: a masked token with a verified -inf logit was
    emitted mid-stream (1 violation in ~250 steps corrupts most generations).
    Gumbel-argmax is distributionally identical to temperature sampling and
    uses only argmax: scores/T + Gumbel noise, and -inf + noise = -inf can
    never win. Use with model.generate(do_sample=False); greedy argmax over
    these perturbed scores IS the sample.

    Noise is drawn on CPU from a seeded generator, so results are
    reproducible regardless of device kernels.
    """

    def __init__(self, temperature: float = 1.0, seed: int = 0) -> None:
        import torch

        self.temperature = max(temperature, 1e-5)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, input_ids, scores):
        import torch

        uniform = torch.rand(
            scores.shape, generator=self.generator, dtype=torch.float32
        ).to(scores.device)
        # Explicit steps: method calls bind before unary minus, so the compact
        # one-liner clamps the WRONG intermediate and NaNs the whole vector.
        neg_log_u = -torch.log(uniform.clamp_min(1e-20))   # > 0
        gumbel = -torch.log(neg_log_u.clamp_min(1e-20))
        return scores / self.temperature + gumbel
