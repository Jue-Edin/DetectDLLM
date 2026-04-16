# Local model files

This repository does not ship large model weights.

Expected local path for the DUO checkpoint:

```text
models/
  duo-distilled/
    config.json
    config.py
    merges.txt
    model.py
    model.safetensors
    README.md
    tokenizer.json
    tokenizer_config.json
    tokenizer_metadata.json
    vocab.json
```

The detector scripts assume that `models/duo-distilled/` already exists on the local machine.
