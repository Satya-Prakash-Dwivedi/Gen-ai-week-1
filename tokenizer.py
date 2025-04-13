#!/usr/bin/env python3
"""
Tokenizer CLI - A command-line tool for tokenizing text for various LLM models
"""


import argparse
import sys
import json
from typing import Dict, List, Tuple, Optional, Union
import tiktoken
from transformers import AutoTokenizer
import anthropic


class TokenizerApp:
    """Main application class for the tokenizer tool"""

    SUPPORTED_MODELS = {
        # OpenAI models
        "gpt-3.5-turbo": "cl100k_base",  # Uses tiktoken
        "gpt-4": "cl100k_base",          # Uses tiktoken
        "gpt-4-turbo": "cl100k_base",    # Uses tiktoken
        
        # Anthropic models
        "claude-3-opus": "claude",       # Uses Anthropic's tokenizer
        "claude-3-sonnet": "claude",     # Uses Anthropic's tokenizer
        "claude-3-haiku": "claude",      # Uses Anthropic's tokenizer
        
        # Hugging Face models
        "llama-3": "meta-llama/Llama-3-8b-hf",  # Uses transformers
        "mistral": "mistralai/Mistral-7B-v0.1",  # Uses transformers
        "grok-1": "xai-org/grok-1",      # Uses transformers
    }

    ROLE_TOKENS = {
        # Approximate token costs for different roles in different models
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
        self.role = "user"  # Default role

    def _get_tokenizer_type(self, model: str) -> str:
        """Determine which tokenizer to use based on the model name"""
        if model.startswith("gpt"):
            return "tiktoken"
        elif model.startswith("claude"):
            return "anthropic"
        else:
            return "transformers"

    def load_tokenizer(self, model: str) -> None:
        """Load the appropriate tokenizer for the specified model"""
        self.current_model = model

        if model not in self.SUPPORTED_MODELS:
            print(f"Error: Model '{model}' is not supported")
            print(f"Supported models: {', '.join(self.SUPPORTED_MODELS.keys())}")
            sys.exit(1)

        if model in self.tokenizers:
            return  # Tokenizer already loaded

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
            print(f"Error loading tokenizer for {model}: {str(e)}")
            sys.exit(1)

    def set_role(self, role: str) -> None:
        """Set the message role (system, user, assistant)"""
        valid_roles = ["system", "user", "assistant"]
        if role not in valid_roles:
            print(f"Error: Invalid role. Choose from {', '.join(valid_roles)}")
            return
        
        self.role = role
        print(f"Role set to: {role}")

    def get_role_token_count(self) -> int:
        """Get the number of tokens used by the role formatting"""
        model_family = self._get_model_family()
        return self.ROLE_TOKENS.get(model_family, {}).get(self.role, 0)

    def _get_model_family(self) -> str:
        """Get the model family for the current model"""
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

    def tokenize_text(self, text: str) -> Tuple[List[int], int]:
        """Tokenize the given text using the current model's tokenizer"""
        if not self.current_model or self.current_model not in self.tokenizers:
            print("Error: No tokenizer loaded. Please select a model first.")
            return [], 0

        tokenizer = self.tokenizers[self.current_model]
        tokenizer_type = self._get_tokenizer_type(self.current_model)
        
        try:
            if tokenizer_type == "tiktoken":
                tokens = tokenizer.encode(text)
                return tokens, len(tokens)
            
            elif tokenizer_type == "anthropic":
                tokens = tokenizer.encode(text).tokens
                return list(range(len(tokens))), len(tokens)  # Anthropic doesn't expose token IDs, using indices
            
            elif tokenizer_type == "transformers":
                result = tokenizer(text, return_tensors="pt")
                token_ids = result.input_ids[0].tolist()
                return token_ids, len(token_ids)
        
        except Exception as e:
            print(f"Error during tokenization: {str(e)}")
            return [], 0

    def display_tokens(self, tokens: List[int], token_count: int) -> None:
        """Display the tokens and related information"""
        model_family = self._get_model_family()
        role_tokens = self.get_role_token_count()
        
        print(f"\n--- Tokenization Results for {self.current_model} ({self.role} role) ---")
        print(f"Raw tokens: {tokens}")
        print(f"Token count: {token_count}")
        print(f"Role formatting tokens: ~{role_tokens}")
        print(f"Total tokens (text + role): ~{token_count + role_tokens}")
        
        if model_family == "gpt":
            pricing_info = {
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
                "gpt-4": {"input": 0.01, "output": 0.03},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03}
            }
            if self.current_model in pricing_info:
                price_info = pricing_info[self.current_model]
                estimated_cost = (token_count + role_tokens) * price_info["input"] / 1000
                print(f"Estimated cost (as input): ${estimated_cost:.6f} USD")

    def decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens back to text when possible"""
        if not self.current_model or self.current_model not in self.tokenizers:
            return "No tokenizer loaded"

        tokenizer = self.tokenizers[self.current_model]
        tokenizer_type = self._get_tokenizer_type(self.current_model)
        
        try:
            if tokenizer_type == "tiktoken":
                return tokenizer.decode(tokens)
            elif tokenizer_type == "transformers":
                return tokenizer.decode(tokens)
            elif tokenizer_type == "anthropic":
                return "Token decoding not supported for Anthropic models"
        except Exception as e:
            return f"Error decoding tokens: {str(e)}"


def main():
    """Main function to run the tokenizer CLI"""
    parser = argparse.ArgumentParser(description="Tokenizer CLI - A tool for tokenizing text for various LLM models")
    
    parser.add_argument("--model", "-m", type=str, 
                        help="Select the model to use for tokenization")
    
    parser.add_argument("--role", "-r", type=str, choices=["system", "user", "assistant"],
                        help="Select the role for the message")
    
    parser.add_argument("--file", "-f", type=str,
                        help="Path to a file containing text to tokenize")
    
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List all supported models")
    
    parser.add_argument("--text", "-t", type=str,
                        help="Text to tokenize (alternative to interactive mode)")
    
    args = parser.parse_args()
    
    app = TokenizerApp()
    
    # Handle listing models
    if args.list_models:
        print("Supported models:")
        for model in app.SUPPORTED_MODELS.keys():
            print(f"  - {model}")
        return
    
    # Set model if provided
    if args.model:
        app.load_tokenizer(args.model)
    else:
        # Default model if none provided
        app.load_tokenizer("gpt-3.5-turbo")
    
    # Set role if provided
    if args.role:
        app.set_role(args.role)
    
    # Process text from file if provided
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens, token_count = app.tokenize_text(text)
            app.display_tokens(tokens, token_count)
            return
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            sys.exit(1)
    
    # Process text from command line argument if provided
    if args.text:
        tokens, token_count = app.tokenize_text(args.text)
        app.display_tokens(tokens, token_count)
        return
    
    # Interactive mode
    print(f"""
======= Tokenizer CLI =======
Current model: {app.current_model}
Current role: {app.role}
Type 'help' for commands or start typing text to tokenize.
Type 'exit' or Ctrl+D to quit.
=============================""")
    
    while True:
        try:
            user_input = input("\n> ")
            
            if user_input.lower() == "exit":
                break
            
            elif user_input.lower() == "help":
                print("""
Commands:
  model <model_name>    - Change the current model
  role <role_name>      - Change the current role (system, user, assistant)
  list models           - List all supported models
  clear                 - Clear the screen
  exit                  - Exit the program
  <text>                - Tokenize the entered text
                """)
            
            elif user_input.lower().startswith("model "):
                model_name = user_input[6:].strip()
                app.load_tokenizer(model_name)
            
            elif user_input.lower().startswith("role "):
                role_name = user_input[5:].strip()
                app.set_role(role_name)
            
            elif user_input.lower() == "list models":
                print("Supported models:")
                for model in app.SUPPORTED_MODELS.keys():
                    print(f"  - {model}")
            
            elif user_input.lower() == "clear":
                # Clear screen - works on most terminals
                print("\033c", end="")
            
            elif user_input.strip():
                tokens, token_count = app.tokenize_text(user_input)
                app.display_tokens(tokens, token_count)
                
                # Show decoded tokens when possible
                if tokens:
                    decoded = app.decode_tokens(tokens)
                    if decoded and decoded != user_input:
                        print(f"\nDecoded text: {decoded}")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        
        except EOFError:
            print("\nExiting...")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)