#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_template.py  —  Check chat template for HF folders and GGUF files, incl. template name guess.

Usage:
  python check_template.py [<model_path_or_gguf>] [--messages <json>] [--max-new 64] [--verbose]

Examples:
  # HF folder:
  python check_template.py "C:\\path\\to\\merged_model_for_gguf_convert"

  # GGUF file:
  python check_template.py "C:\\path\\to\\model-unsloth-Q4_K_M.gguf"

  # Custom messages + more tokens:
  python check_template.py ./model.gguf --messages "[{\\"role\\":\\"user\\",\\"content\\":\\"Sag Hallo!\\"}]" --max-new 128

Exit codes:
  0 = OK
  2 = Load / argument error
  3 = Rendering / chat error
  4 = GGUF loaded but no chat template metadata found
"""

import argparse
import json
import os
import sys
import traceback

# --- Defaults (optional): set to avoid passing the path every time ---
DEFAULT_MODEL_PATH = r""  # e.g. r"C:\path\to\model.gguf" or r"C:\path\to\hf_folder"
DEFAULT_MESSAGES_JSON = None  # e.g. '[{"role":"user","content":"Sag Hallo!"}]'
DEFAULT_MAX_NEW = 64

def is_dir(p: str) -> bool:
    try:
        return os.path.isdir(p)
    except Exception:
        return False

def is_gguf(p: str) -> bool:
    return os.path.isfile(p) and p.lower().endswith(".gguf")

DEFAULT_MESSAGES = [
    {"role": "user", "content": "Nenne eine Zahl zwischen 1 und 3."},
    {"role": "assistant", "content": "2"},
]

def summarize(s: str, n: int = 300) -> str:
    s = (s or "").strip().replace("\r", "").replace("\t", " ")
    return (s[:n] + " …") if len(s) > n else s

def guess_template_name(tpl: str) -> str:
    """Very light-weight heuristic to label common templates."""
    t = (tpl or "").lower()
    # strong signals first
    if "### instruction" in t and "### response" in t:
        return "alpaca-like"
    if "<|im_start|>" in t or "<|im_end|>" in t:
        return "chatml"
    if "<|start_header_id|>" in t or "<|end_header_id|>" in t or "assistant\n" in t and "user\n" in t and "system\n" in t:
        return "llama-3 style"
    if "[inst]" in t and "[/inst]" in t:
        return "llama-2 style"
    if "<start_of_turn>" in t or "<end_of_turn>" in t:
        return "llama-3.* (alt) / unsloth"
    if "gemma" in t or "google" in t and "assistant" in t and "user" in t:
        return "gemma-like"
    if "role" in t and "messages" in t and "eos_token" in t:
        return "hf-jinja (generic)"
    return "unknown"

def check_hf_folder(path: str, messages, verbose=False) -> int:
    print("=== Hugging Face (folder) ===")
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("ERROR: transformers not installed. pip install transformers", file=sys.stderr)
        return 2

    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as e:
        print("ERROR: could not load tokenizer from:", path, "\n", e, file=sys.stderr)
        return 2

    print("Loaded tokenizer from:", path)
    tpl = getattr(tok, "chat_template", None)
    print("Has chat_template?:", bool(tpl))
    if tpl:
        print("Guessed template name:", guess_template_name(tpl))
    print("chat_template snippet:\n", summarize(tpl))

    print("Tokens — BOS/EOS/PAD ids:", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id)
    print("Tokens — BOS/EOS/PAD str:", tok.bos_token, tok.eos_token, tok.pad_token)

    # Render test
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        print("\nRendered prompt preview:\n", summarize(rendered, 600))
        # quick tokenize check
        toks = tok(rendered + (tok.eos_token or "")).input_ids[:20]
        print("Tokenize OK — first IDs:", toks)
        print("\nRESULT: HF template appears **OK**.")
        return 0
    except Exception:
        print("ERROR during apply_chat_template:\n", traceback.format_exc(), file=sys.stderr)
        return 3

def check_gguf_file(path: str, messages, max_new_tokens=64, verbose=False) -> int:
    print("=== GGUF (llama-cpp-python) ===")
    try:
        from llama_cpp import Llama
    except Exception:
        print("ERROR: llama-cpp-python not installed. pip install llama-cpp-python", file=sys.stderr)
        return 2

    try:
        # Force gguf chat handler if present; verbose logs show chosen template.
        llm = Llama(model_path=path, chat_format="gguf", verbose=bool(verbose or True))
    except Exception as e:
        print("ERROR: could not load GGUF:", path, "\n", e, file=sys.stderr)
        return 2

    meta = getattr(llm, "metadata", {}) or {}
    has_tpl = ("tokenizer.chat_template" in meta) or ("tokenizer.chat_templates" in meta)
    print("Loaded GGUF:", path)
    print("Metadata keys present:", len(meta))
    print("Has tokenizer.chat_template?:", "tokenizer.chat_template" in meta)
    if "tokenizer.chat_template" in meta:
        tpl = meta["tokenizer.chat_template"]
        print("Guessed template name:", guess_template_name(tpl))
        print("chat_template snippet:\n", summarize(tpl))

    if "tokenizer.chat_templates" in meta:
        print("\nFound tokenizer.chat_templates (multiple). Showing first entry snippet:")
        try:
            tpls = meta["tokenizer.chat_templates"]
            s = str(tpls) if isinstance(tpls, str) else json.dumps(tpls)
            print(summarize(s, 600))
        except Exception:
            pass

    # Minimal chat completion test — if llama-cpp sees a template, it will try to use it
    try:
        msgs = messages if messages else [{"role":"user","content":"Sag Hallo!"}]
        out = llm.create_chat_completion(messages=msgs, max_tokens=max_new_tokens)
        text = out["choices"][0]["message"]["content"]
        print("\nChat completion OK — preview:\n", summarize(text, 600))
        if has_tpl:
            print("\nRESULT: GGUF has template metadata and chat works — looks **OK**.")
            return 0
        else:
            print("\nWARN: No template metadata found — your frontend may fallback to a default (e.g., Alpaca).")
            return 4
    except Exception:
        print("ERROR during chat completion:\n", traceback.format_exc(), file=sys.stderr)
        return 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="HF folder OR .gguf path (optional if DEFAULT_MODEL_PATH/MODEL_PATH is set)")
    ap.add_argument("--messages", help="JSON list of {role,content}", default=None)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Resolve path priority: CLI > ENV > DEFAULT
    if not args.path:
        env_path = os.getenv("MODEL_PATH", "").strip()
        if env_path:
            args.path = env_path
        elif DEFAULT_MODEL_PATH:
            args.path = DEFAULT_MODEL_PATH
        else:
            print("ERROR: No path provided and DEFAULT_MODEL_PATH/MODEL_PATH not set.", file=sys.stderr)
            sys.exit(2)

    # messages
    if args.messages:
        try:
            messages = json.loads(args.messages)
            assert isinstance(messages, list) and all(isinstance(m, dict) for m in messages)
        except Exception:
            print("ERROR: --messages must be a JSON list of {role,content}", file=sys.stderr)
            sys.exit(2)
    else:
        messages = None
        if DEFAULT_MESSAGES_JSON:
            try:
                messages = json.loads(DEFAULT_MESSAGES_JSON)
            except Exception:
                messages = None
        if messages is None:
            messages = DEFAULT_MESSAGES

    # default max-new override
    if args.max_new == 64 and DEFAULT_MAX_NEW != 64:
        args.max_new = DEFAULT_MAX_NEW

    p = args.path
    if is_dir(p):
        code = check_hf_folder(p, messages, verbose=args.verbose)
    elif is_gguf(p):
        code = check_gguf_file(p, messages, max_new_tokens=args.max_new, verbose=args.verbose)
    else:
        print("ERROR: path is neither a directory nor a .gguf file:", p, file=sys.stderr)
        code = 2
    sys.exit(code)

if __name__ == "__main__":
    main()
