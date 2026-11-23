"""
Uniform wrappers around the open-weight VLMs used in Part 2 (Sparse Attention Vectors).

Every wrapper exposes:
    vlm.run(image: PIL.Image, prompt: str, max_new_tokens: int) -> str
    vlm.model      (the underlying HF model, used by savs.extract to hook attention heads)
    vlm.n_heads    (attention heads per language-model layer)

Model keys (see MODELS): qwen25vl_7b, phi4_mm, llama32v_11b, gemma3_12b, smolvlm_256m (local smoke test only).
"""
import torch
import transformers
from packaging import version

_TV = version.parse(transformers.__version__)
# transformers renamed torch_dtype -> dtype in 4.56; Phi-4 needs the old 4.48 pin.
_DTYPE_KW = "dtype" if _TV >= version.parse("4.56") else "torch_dtype"

MODELS = {
    # key            HF id                                         loader   quant    compute dtype      owner
    "qwen25vl_7b":  dict(hf_id="Qwen/Qwen2.5-VL-7B-Instruct",             kind="chat", quant="4bit", dtype="float16",  owner="Ujjwal",  gated=False),
    "phi4_mm":      dict(hf_id="microsoft/Phi-4-multimodal-instruct",     kind="phi4", quant=None,   dtype="float16",  owner="Ujjwal",  gated=False),
    "llama32v_11b": dict(hf_id="meta-llama/Llama-3.2-11B-Vision-Instruct", kind="chat", quant="4bit", dtype="float16",  owner="Anvesh",  gated=True),
    "gemma3_12b":   dict(hf_id="google/gemma-3-12b-it",                   kind="chat", quant="4bit", dtype="bfloat16", owner="Pratham", gated=True),
    # tiny model that fits a 4 GB laptop GPU; used only to smoke-test the pipeline
    "smolvlm_256m": dict(hf_id="HuggingFaceTB/SmolVLM-256M-Instruct",     kind="chat", quant=None,   dtype="float16",  owner="local",   gated=False),
}
# Gemma-3 activations overflow float16 (NaNs/inf), hence bfloat16 for it. If a model produces NaNs
# or garbage on the T4, override with load_model(key, dtype="float32") (slower, more memory).


def _text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


class _Base:
    def __init__(self, key, dtype=None):
        spec = MODELS[key]
        self.key, self.hf_id, self.spec = key, spec["hf_id"], spec
        self.compute_dtype = getattr(torch, dtype or spec["dtype"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_device(self, inputs):
        out = {}
        for k, v in dict(inputs).items():
            if torch.is_tensor(v):
                v = v.to(self.model.device)
                if v.is_floating_point():
                    v = v.to(self.compute_dtype)
            out[k] = v
        return out

    @property
    def n_heads(self):
        return _text_config(self.model.config).num_attention_heads

    @torch.inference_mode()
    def run(self, image, prompt, max_new_tokens=8):
        inputs = self._to_device(self.build_inputs(image, prompt))
        n_in = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  **self._gen_kwargs())
        return self._decode(out[:, n_in:])

    def _gen_kwargs(self):
        return {}


class ChatVLM(_Base):
    """Qwen2.5-VL, Llama-3.2-Vision, Gemma-3, SmolVLM: all go through AutoModelForImageTextToText + chat template."""

    def __init__(self, key, dtype=None):
        super().__init__(key, dtype)
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        kw = {_DTYPE_KW: self.compute_dtype, "device_map": "auto" if self.device == "cuda" else None}
        if self.spec["quant"] == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype, bnb_4bit_use_double_quant=True)
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, **kw).eval()
        offloaded = [d for d in getattr(self.model, "hf_device_map", {}).values() if d in ("cpu", "disk")]
        if offloaded:
            print(f"WARNING: {len(offloaded)} modules offloaded to CPU/disk -> very slow. Restart runtime and retry.")

    def build_inputs(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt")

    def _decode(self, ids):
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class Phi4MM(_Base):
    """microsoft/Phi-4-multimodal-instruct (remote code; needs transformers==4.48.2, peft==0.13.2)."""

    def __init__(self, key, dtype=None):
        super().__init__(key, dtype)
        if _TV != version.parse("4.48.2"):
            print(f"WARNING: Phi-4-multimodal expects transformers==4.48.2, found {transformers.__version__}.")
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        self.processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, trust_remote_code=True, _attn_implementation="eager",  # no flash-attn on T4
            device_map="cuda" if self.device == "cuda" else None,
            **{_DTYPE_KW: self.compute_dtype}).eval()
        self.gen_cfg = GenerationConfig.from_pretrained(self.hf_id)

    def build_inputs(self, image, prompt):
        text = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
        return self.processor(text=text, images=[image], return_tensors="pt")

    def _gen_kwargs(self):
        return {"generation_config": self.gen_cfg}

    def _decode(self, ids):
        return self.processor.batch_decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()


def load_model(key, dtype=None):
    """Load a model by key. dtype overrides the default compute dtype ('float16'|'bfloat16'|'float32')."""
    if key not in MODELS:
        raise KeyError(f"Unknown model {key!r}; choose from {list(MODELS)}")
    cls = Phi4MM if MODELS[key]["kind"] == "phi4" else ChatVLM
    return cls(key, dtype)
