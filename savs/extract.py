"""
Step 1 of SAVs (Mitra et al., arXiv 2412.00142, Sec. 3.2): record every attention head's output
h_l^m(x_T) at the LAST input token, for every language-model layer.

The per-head outputs are exactly the input to each self-attention's output projection (o_proj):
o_proj(Concat(h^1 ... h^H)). A forward pre-hook on o_proj therefore gives all H heads of a layer
in one tensor of shape [hidden = H * head_dim], which we reshape to [H, head_dim].

Only the prefill pass (the full prompt) is recorded; decode steps during generation are ignored.
"""
import torch

# o_proj modules inside the vision / audio encoders must not be hooked (they are not LM heads).
_EXCLUDE = ("vision", "visual", "audio", "embed_tokens_extend", "image")


class HeadExtractor:
    def __init__(self, vlm):
        self.n_heads = vlm.n_heads
        self.modules = [(n, m) for n, m in vlm.model.named_modules()
                        if n.endswith("self_attn.o_proj") and not any(x in n.lower() for x in _EXCLUDE)]
        if not self.modules:
            raise RuntimeError("No language-model self_attn.o_proj modules found to hook.")
        self.n_layers = len(self.modules)
        self._buf = [None] * self.n_layers
        self._handles = [m.register_forward_pre_hook(self._make_hook(i)) for i, (_, m) in enumerate(self.modules)]

    def _make_hook(self, i):
        def hook(module, args):
            if self._buf[i] is None:                       # first call = prefill over the whole prompt
                self._buf[i] = args[0][0, -1].detach().float().cpu()
        return hook

    def reset(self):
        self._buf = [None] * self.n_layers

    def collect(self):
        """Return the [n_layers, n_heads, head_dim] activations of the last prompt token."""
        if any(b is None for b in self._buf):
            raise RuntimeError("Some layers were not recorded; did the forward pass run?")
        acts = torch.stack(self._buf)                       # [L, H*D]
        acts = acts.view(self.n_layers, self.n_heads, -1)   # [L, H, D]
        if not torch.isfinite(acts).all():
            raise FloatingPointError("Non-finite activations (fp16 overflow?). Reload with dtype='bfloat16' or 'float32'.")
        return acts

    def remove(self):
        for h in self._handles:
            h.remove()

    def describe(self):
        return (f"{self.n_layers} layers x {self.n_heads} heads = {self.n_layers * self.n_heads} heads "
                f"(first hooked module: {self.modules[0][0]})")
