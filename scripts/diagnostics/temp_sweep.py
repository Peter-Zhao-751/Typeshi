"""Temperature-validity sweep on 20 held-out targets, using stored prompts."""
import json
import re
import traceback

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, LogitsProcessorList

from typeshi.buffer import replay
from typeshi.constrain import GumbelSampleProcessor, TranscriptionGrammarProcessor
from typeshi.labels import _levenshtein
from typeshi.serialize import deserialize

ck = "checkpoints/motor"
tok = AutoTokenizer.from_pretrained(ck)
model = AutoPeftModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="auto")
model.eval()

seen, cases = set(), []
with open("data/processed/test.jsonl") as fh:
    for line in fh:
        ex = json.loads(line)
        tgt = re.search(r"<TARGET>(.*)<PROCESS>", ex["prompt"], re.S).group(1)
        if tgt in seen:
            continue
        seen.add(tgt)
        cases.append((ex["prompt"], tgt))
        if len(cases) >= 20:
            break
print(f"{len(cases)} cases", flush=True)


def sample(prompt: str, temperature: float, seed: int) -> list:
    """generate() with the stored prompt instead of rebuilding labels."""
    torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    chain = LogitsProcessorList([
        TranscriptionGrammarProcessor(tok, inputs["input_ids"].shape[1]),
        GumbelSampleProcessor(temperature=temperature, seed=seed),
    ])
    out = model.generate(
        **inputs, do_sample=False, max_new_tokens=512,
        pad_token_id=tok.pad_token_id, logits_processor=chain,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    text = text.split(tok.eos_token)[0] if tok.eos_token in text else text
    return deserialize(text)


for temp in (1.0, 0.7, 0.5, 0.3):
    ok, sims, errs = 0, [], 0
    for i, (prompt, tgt) in enumerate(cases):
        try:
            events = sample(prompt, temp, seed=i)
            produced = replay(events)
            sim = 1 - _levenshtein(produced, tgt) / max(len(tgt), 1)
        except Exception:
            if errs == 0:
                traceback.print_exc()
            errs += 1
            sim = 0.0
        sims.append(sim)
        ok += sim >= 0.8
    sims.sort()
    print(
        f"temp {temp}: valid {ok}/20  median sim {sims[len(sims)//2]:.3f}  "
        f"min {sims[0]:.3f}  errors {errs}",
        flush=True,
    )
print("SWEEP DONE", flush=True)
