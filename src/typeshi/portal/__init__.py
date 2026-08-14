"""Local web portal for driving a trained motor model by hand.

Split into modules rather than living in scripts/playground.py because the
portal grew past "toy": checkpoint discovery, a job queue, per-sample realism
metrics and corpus lookup are each independently testable without a model,
and a single 900-line script would make none of them so.

Nothing here binds to anything but 127.0.0.1. The phase-2 checkpoint is
KLiCKe-derived and KLiCKe has no license terms, so neither this server nor
anything it generates may be exposed off the machine.
"""
