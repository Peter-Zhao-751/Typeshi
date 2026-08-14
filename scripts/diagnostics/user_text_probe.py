import torch
from typeshi.buffer import replay
from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
from typeshi.generate import generate, generate_windowed
from typeshi.labels import SessionLabels
from typeshi.train_motor import _detect_backend
from pathlib import Path

TARGET = ("Following the end of World War II, the victorious America was a global "
          "leader in a modern world. Men returned home to reoccupy jobs taken by "
          "women who left the workforce to return to a domestic sphere.")
CK = Path("checkpoints/motor-phase2")
tok = load_checkpoint_tokenizer(CK)
model = load_checkpoint_model(CK, _detect_backend()); model.eval()
labels = SessionLabels(55.0, 0.03, 0.005, 0.01)
print(f"target ({len(TARGET)} chars)\n")

ev = generate(model, tok, TARGET, labels, mode="transcription",
              temperature=1.0, max_new_tokens=4*len(TARGET)+64, seed=1)
out = replay(ev)
print(f"A) transcription + grammar mask (NO convergence guarantee):\n   {out!r}\n   exact: {out == TARGET}\n")

ev = generate_windowed(model, tok, TARGET, labels, temperature=1.0, seed=1)
out = replay(ev)
print(f"B) composition + CONVERGENCE decoder:\n   {out!r}\n   exact: {out == TARGET}")
print(f"   events: {len(ev)}, backspaces: {sum(1 for e in ev if e.type.name=='BACKSPACE')}")
