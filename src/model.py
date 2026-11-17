"""
Generator model module of the RAG system.
Does:
- Define the model structure/configuration;
- Download the weights from the Hugging Face repository;
- Load them into the created model;
- Expose a `generate(prompt)` method, used by `main.py` and the retrievers.

Model selection: Qwen2.5-3B-Instruct.
- It is already instruction-tuned, and therefore focused on providing responses;
- Apache 2.0 license: no authentication or license agreement required for download (unlike Llama);
- Small size: it runs on a single RTX GPU (via WSL) and, in 4-bit, requires less than ~3 GB of VRAM.

Author: Alessandro Diana
Version: November 15, 2026
"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------- costant -----------------------


# ----------------------- methods -----------------------

def print_used_vram(model):
    """
    Method to get and print the current VRAM allocation using cuda.
    
    Args:
        model: the model to analyse
    """
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()                # Synchronize the GPU to obtain accurate data
        
        allocated_bytes = torch.cuda.memory_allocated()
        allocated_gb = allocated_bytes / (1024 ** 3)
        
        max_allocated_bytes = torch.cuda.max_memory_allocated()
        max_allocated_gb = max_allocated_bytes / (1024 ** 3)
        
        print(f"[model] VRAM allocated: {allocated_gb:.2f} GB (Peak: {max_allocated_gb:.2f} GB)")
    else:
        print("[model] CUDA not available, running on CPU.")


@dataclass
class LLMConfig:
    """Model and generation parameters, gathered in one place."""
    
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device_map: str = "auto"                        # use GPU if available, otherwise CPU    
    dtype: str = "auto"                             # bfloat16 on new GPU, float32 on CPU
    # Load weights in 4-bit to reduce VRAM usage (requires bitsandbytes)
    load_in_4bit: bool = True
    # method for output generation (selection of the next token)
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

    The instance downloads the weights (if not already cached) and loads them upon creation.
    The rest of the program interacts solely via `generate()`.
    """


    def __init__(self, config: LLMConfig | None = None):
        self.cfg = config or LLMConfig()
        self.tokenizer = None
        self.model = None
        self._load()


    def _load(self) -> None:
        """
        Download (upon first launch) and load the tokenizer and model weights.
        `from_pretrained` automatically downloads the files into the Hugging Face cache if they are not already present; for subsequent runs, it reuses the cache.
        """
        
        print(f"[model] Loading '{self.cfg.model_name}' model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        load_kwargs = {
            "device_map": self.cfg.device_map,
            "torch_dtype": self.cfg.dtype,
        }
        # 4-bit quantization
        if self.cfg.load_in_4bit:
            # get quantization configuration
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)
        self.model.eval()                       # Inference mode: disable dropout, etc.
        print_used_vram(self.model)
        print("[model] Model ready.")


    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate a response based on a user prompt.
        
        Args:
            prompt: user prompt, query plus context. The prompt is inserted into the model's chat template (with the 'user' role) 
            system_prompt: before the user prompt, the prompt for the system.
        
        Returns: 
            the model response to the user questions (newly generated text, excluding the input prompt).
        """
        messages = [
            {"role": "system", "content": system_prompt or self.cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self._chat(messages)

    def _chat(self, messages: list[dict]) -> str:
        """
        Applies the chat template, performs generation, and decodes the output.
        
        Args:
            messagges: list of dictionaries
        
        Returns:
            output string
        """
        # Format the messages according to the model's template and tokenize them.
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    Rapid and simple test.
    Without context, the model is unaware of a made-up definition: useful for seeing the starting point before adding RAG.
    """
    
    llm = LocalLLM()
    answer = llm.generate("What is a Florbometer?")
    print("\n[TEST] Question is 'What is a Florbometer?'")
    print("Answer without context:\n", answer)

"""
MODEL NOTE:
    Qwen2.5-3B-Instruct:
        Site specs: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
        Long-context Support up to 128K tokens and can generate up to 8K tokens.
        Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.
    Qwen2.5-7B-Instruct:
        Site specs: https://huggingface.co/ThomasBaruzier/Qwen2.5-7B-Instruct-GGUF
    Qwen2.5-14B-Instruct:
        Site specs: https://huggingface.co/ThomasBaruzier/Qwen2.5-14B-Instruct-GGUF
"""