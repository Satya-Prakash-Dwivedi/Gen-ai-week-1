import argparse
import sys
import json
from typing import Dict, List, Tuple, Optional, Union
import tiktoken
from transformers import AutoTokenizer
import anthropic

class TokenizerApp:

     SUPPORTED_MODELS = {
          # OpenAI models
          "gpt-3.5-turbo": "cl100k_base",
          "gpt-4": "cl100k_base",
          "gpt-4-turbo":"cl100k_base",

          # Anthropic Models
          "claude-3-opus": "claude",
          "claude-3-sonnet": "claude",
          "claude-3-haiku": "claude",

          # Hugging face models
          "llama-3": "meta-llama/Llama-3-8b-hf",
          "mistral": "mistralai/Mistral-7B-v0.1",
          "grok-1": "xai-org/grok-1"
     }

     ROLE_TOKENS = {
           "gpt": {
            "system": 4,
            "user": 4,
            "assistant": 4
        },
        "claude": {
            "system": 8,
            "user": 8,
            "assistant": 8
        },
        "llama": {
            "system": 4,
            "user": 4,
            "assistant": 4
        },
        "mistral": {
            "system": 4,
            "user": 4,
            "assistant": 4
        },
        "grok": {
            "system": 4,
            "user": 4,
            "assistant": 4
        }
     }

     def __init__(self):
          self.tokenizers = {}
          self.current_model = None
          self.role = "user" # Default role

     def _get_tokenizer_type(self, model: str) -> str:
        if model.startswith("gpt"):
            return "tiktoken"
        elif model.startswith("claude"):
            return "anthropic"
        else: 
            return "transformers"
    
     def load_tokenizer(self, model: str) -> None: 
         self.current_model = model
         
         if model not in self.SUPPORTED_MODELS: 
             print(F"Error: Model '{model}' is not supported")
             print(f"Supported Models: {', '.join(self.SUPPORTED_MODELS.keys())}")
             sys.exit(1)

         if model in self.tokenizers:
             return # tokenizer already loaded
         
         tokenizer_type = self._get_tokenizer_type(model)

         try: 
             if tokenizer_type == "tiktoken":
                 encoding_name = self.SUPPORTED_MODELS[model]
                 self.tokenizers[model] = tiktoken.get_encoding(encoding_name)

             elif tokenizer_type == "anthropic":
                 self.tokenizers[model] = anthropic.Anthropic().get_tokenizer()

             elif tokenizer_type == "transformers":
                 model_name = self.SUPPORTED_MODELS[model]
                 self.tokenizers[model] = AutoTokenizer.from_pretrained(model_name)

             print(f"Successfully loaded tokenizer for {model}")

         except Exception as e:
          print(f"Error loading tokenizer for     {model} : {str(e)}")
          sys.exit(1)

     def set_role(self, role: str) -> None:
        valid_roles = ["system", "user", "assistant"]
        if role not in valid_roles:
            print(f"Error: Invalid role. Choose from {', '.join(valid_roles)}")
            return
        
        self.role = role
        print(f"Role set to: {role}")

     def get_role_token_count(self) -> int:

        model_family = self._get_model_family()
        return self.ROLE_TOKENS.get(model_family, {}).get(self.role, 0)

     def _get_model_family(self) -> str:
        
        if self.current_model.startswith("gpt"):
            return "gpt"
        elif self.current_model.startswith("claude"):
            return "claude"
        elif self.current_model.startswith("llama"):
            return "llama"
        elif self.current_model.startswith("mistral"):
            return "mistral"
        elif self.current_model.startswith("grok"):
            return "grok"
        return "unknown"