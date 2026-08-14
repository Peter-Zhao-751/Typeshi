"""After the full gold completion, does the model want to stop?"""
import itertools
import json

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ck = "checkpoints/motor"
tok = AutoTokenizer.from_pretrained(ck)
model = AutoPeftModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="auto")
model.eval()

print("tok.eos_token:", repr(tok.eos_token), tok.eos_token_id)
print("tok.pad_token:", repr(tok.pad_token), tok.pad_token_id)
gc = model.generation_config
print("generation_config eos:", gc.eos_token_id, " pad:", gc.pad_token_id)

with open("data/processed/test.jsonl") as fh:
    examples = [json.loads(l) for l in itertools.islice(fh, 5)]

for n, ex in enumerate(examples):
    full = ex["prompt"] + ex["completion"]
    ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        probs = torch.softmax(model(ids).logits[0, -1].float(), -1)
    top = torch.topk(probs, 5)
    tops = ", ".join(
        f"{tok.convert_ids_to_tokens(int(i))!r}:{float(p):.3f}"
        for p, i in zip(top.values, top.indices)
    )
    print(f"case {n}: p(eos {tok.eos_token_id})={float(probs[tok.eos_token_id]):.4f}  "
          f"p(248044)={float(probs[248044]):.4f}  p(248045)={float(probs[248045]):.4f}")
    print(f"  top5: {tops}")
print("EOS PROBE DONE")
