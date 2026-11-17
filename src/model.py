"""
Generator model module of the RAG system.
Does:
- Define the model structure/configuration;
- Download the weights from the Hugging Face repository (once) and cache them locally;
- Load them into the created model;
- Expose a `generate(prompt)` method, used by `main.py` and the retrievers.

Model selection: the Qwen2.5-Instruct family (3B / 7B / 14B).
- Already instruction-tuned, therefore focused on providing responses;
- Apache 2.0 license: no authentication or license agreement required for download (unlike Llama);
- The 3B runs comfortably on a single RTX GPU (via WSL); the 7B/14B need more VRAM
  (see MODEL_SPECS below) and are best used in 4-bit, or they fall back to CPU.

Author: Alessandro Diana
Version: November 17, 2026
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------- constants -----------------------

# Local folder where models are saved, so they are downloaded only once.
# Kept out of git via .gitignore.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Approximate sizes for each supported model, used both to inform the user
# (see main.py) and to pick a sensible default precision.
#   download_gb : size of the fp16/bf16 weights pulled from Hugging Face (disk).
#   vram_fp16_gb: VRAM needed to load the weights in full precision.
#   vram_4bit_gb: VRAM needed to load the weights quantized to 4-bit (Q4).
# Figures are approximate and exclude the extra VRAM used at inference time
# (activations, KV cache), so leave some headroom.
MODEL_SPECS = {
    "Qwen/Qwen2.5-3B-Instruct":  {"download_gb": 6.2,  "vram_fp16_gb": 7.0,  "vram_4bit_gb": 2.8},
    "Qwen/Qwen2.5-7B-Instruct":  {"download_gb": 15.2, "vram_fp16_gb": 16.0, "vram_4bit_gb": 5.8},
    "Qwen/Qwen2.5-14B-Instruct": {"download_gb": 29.5, "vram_fp16_gb": 30.0, "vram_4bit_gb": 12.4},
}


# ----------------------- helper methods -----------------------

def print_used_vram(model=None):
    """
    Get and print the current VRAM allocation using CUDA.

    Args:
        model: unused, kept for backward compatibility.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()                # synchronize the GPU to obtain accurate data
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[model] VRAM allocated: {allocated_gb:.2f} GB (Peak: {max_allocated_gb:.2f} GB)")
    else:
        print("[model] CUDA not available, running on CPU.")


def available_vram_gb() -> float | None:
    """Return the total VRAM of the current CUDA device in GB, or None on CPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return None


@dataclass
class LLMConfig:
    """Model and generation parameters, gathered in one place."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device_map: str = "auto"                        # use GPU if available, otherwise CPU
    dtype: str = "auto"                             # bfloat16 on new GPU, float32 on CPU
    # Load weights in 4-bit to reduce VRAM usage (requires bitsandbytes).
    load_in_4bit: bool = True
    # Local cache: models are saved here and reused instead of re-downloading.
    cache_dir: str = str(MODELS_DIR)
    # Generation (next-token selection) parameters.
    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    # Default system prompt: encourages the model to rely on the context.
    system_prompt: str = (
        "You are a precise assistant. Answer the user's question using only the "
        "information provided in the context. If the context does not contain the "
        "answer, say that you don't have that information."
    )


class LocalLLM:
    """
    Encapsulates the model and tokenizer behind a simple interface.

    On creation it downloads the weights (only the first time, then reuses the
    local cache in `cfg.cache_dir`) and loads them. The rest of the program
    interacts solely via `generate()`.
    """

    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self.tokenizer = None
        self.model = None
        self._load()

    # ------------------------------------------------------------------ #
    # Download + load
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """
        Download (first run) and load the tokenizer and model weights.

        `from_pretrained(..., cache_dir=...)` downloads the files into the given
        local folder if they are not already present, and reuses them on
        subsequent runs, so the model is downloaded only once.
        """
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[model] Loading '{self.cfg.model_name}' model ...")
        print(f"[model] Local cache: {self.cfg.cache_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, cache_dir=self.cfg.cache_dir
        )

        load_kwargs = {
            "device_map": self.cfg.device_map,
            "torch_dtype": self.cfg.dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        # Decide whether 4-bit quantization is usable. bitsandbytes requires the
        # whole quantized model to fit in VRAM: if it does not, loading fails with
        # "Some modules are dispatched on the CPU or the disk". We check the GPU
        # VRAM against the model's 4-bit footprint and, if it does not fit, we
        # disable quantization and let the model run (slowly) on CPU/fp16 instead
        # of crashing.
        use_4bit = self.cfg.load_in_4bit and torch.cuda.is_available()
        if use_4bit:
            spec = MODEL_SPECS.get(self.cfg.model_name)
            vram = available_vram_gb()
            if spec and vram is not None and vram < spec["vram_4bit_gb"] + 1.0:
                print(f"[model] WARNING: ~{spec['vram_4bit_gb']:.1f} GB needed in 4-bit but only "
                      f"~{vram:.1f} GB VRAM available. Disabling 4-bit to avoid a crash; "
                      f"the model will use CPU offload and run slowly.")
                use_4bit = False

        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif not torch.cuda.is_available():
            # On CPU, 4-bit (bitsandbytes) is not supported: force full precision.
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        self.model.eval()                       # inference mode: disable dropout, etc.
        print_used_vram()
        print("[model] Model ready.")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a response based on a user prompt.

        Args:
            prompt: user prompt (query plus context), inserted into the model's
                    chat template with the 'user' role.
            system_prompt: optional system prompt placed before the user prompt.

        Returns:
            the newly generated text (excluding the input prompt).
        """
        messages = [
            {"role": "system", "content": system_prompt or self.cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)

    def _chat(self, messages: list[dict]) -> str:
        """Apply the chat template, run generation, and decode the output."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=self.cfg.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # discard the prompt tokens and decode only the new response.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    """
    Quick test. Without context, the model is unaware of a made-up definition:
    useful for seeing the starting point before adding RAG.
    """
    llm = LocalLLM()
    answer = llm.generate("What is a Florbometer?")
    print("\n[TEST] Question is 'What is a Florbometer?'")
    print("Answer without context:\n", answer)

"""
MODEL NOTE:
    Qwen2.5-3B-Instruct:  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
    Qwen2.5-7B-Instruct:  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
    Qwen2.5-14B-Instruct: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
    All support up to 128K context tokens and generate up to 8K tokens,
    with multilingual support for 29+ languages (English, Italian, ...).
"""
