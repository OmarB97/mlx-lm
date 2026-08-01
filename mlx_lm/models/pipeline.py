# Copyright © 2025 Apple Inc.

import mlx.core as mx


class PipelineMixin:
    def __init__(self):
        super().__init__()
        self.pipeline_rank = 0
        self.pipeline_size = 1
        self.start_idx = 0
        self.end_idx = None

    @property
    def pipeline_layers(self):
        return self.layers[self.start_idx : self.end_idx]

    def pipeline(self, group):
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first
        self.pipeline_rank = group.rank()
        self.pipeline_size = group.size()
        n = len(self.layers)
        base = n // self.pipeline_size
        extra = n % self.pipeline_size
        # Execution order runs from rank size-1 (first layers) down to rank 0
        # (last layers). Hand the remainder to the earliest execution
        # positions so rank 0 — which also holds the output head(s) — gets
        # the smallest share. Computing explicit per-position sizes keeps the
        # partition exact for any n/size (the previous uniform
        # layers-per-rank arithmetic skipped a layer on uneven splits, e.g.
        # 43 layers over 2 ranks left layer 21 unassigned).
        sizes = [base + (1 if i < extra else 0) for i in range(self.pipeline_size)]
        pos = self.pipeline_size - 1 - self.pipeline_rank  # execution position
        self.start_idx = sum(sizes[:pos])
        self.end_idx = self.start_idx + sizes[pos]
        self.layers = self.layers[: self.end_idx]
        # Keep the layer numbers the same for model loading
        self.layers[: self.start_idx] = [None] * self.start_idx
