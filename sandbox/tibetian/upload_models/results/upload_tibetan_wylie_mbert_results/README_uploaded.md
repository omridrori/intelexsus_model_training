
# mbert-tibetan-continual-wylie-final

This repository is public.

## Overview
**This is a BERT model continually trained from `bert-base-multilingual-cased` on Tibetan data represented in Wylie transliteration.**

It was trained as part of the Intelexsus project on a mixed Tibetan corpus that includes:
- Tibetan text converted from the original Tibetan script (Unicode) into Wylie transliteration
- Data that was originally authored in Wylie transliteration

The goal is to improve Tibetan representations for downstream tasks where Tibetan content is available or normalized in Wylie.

## Model Details
- **Base model**: `bert-base-multilingual-cased`
- **Language/Script**: Tibetan via Wylie transliteration (bo)
- **Training objective**: Masked Language Modeling (MLM)
- **Architecture**: 12-layer, 768-hidden, 12-heads
- **Tokenizer**: WordPiece tokenizer compatible with mBERT

## How to Use
You can use this model directly with the `transformers` library for the fill-mask task.

```python
from transformers import pipeline

model_name = "OMRIDRORI/mbert-tibetan-continual-wylie-final"
unmasker = pipeline("fill-mask", model=model_name)

# Example sentence in Wylie transliteration (demonstrative only)
result = unmasker("bod yig la [MASK] yod do")
print(result)
```

You can also load the model and tokenizer directly for more control:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "OMRIDRORI/mbert-tibetan-continual-wylie-final"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
# You can now use the model for your own fine-tuning and inference tasks.
```

## Training Data
The continual training used a Tibetan corpus consisting of:
- Tibetan Unicode-script text converted to Wylie transliteration
- Native Wylie transliterated sources

This combination aims to support tasks where Tibetan content is handled in transliterated form.

## Intended Use and Limitations
This model is intended for research and downstream tasks involving Tibetan in Wylie transliteration. It may contain biases present in the training data and may not perform well outside this domain.

## Citation
If you use this model, please cite the Intelexsus project or link to the model page: `https://huggingface.co/OMRIDRORI/mbert-tibetan-continual-wylie-final`
