
# mbert-tibetan-continual-unicode-240k

This repository is public.

## Overview
**This is a BERT model continually trained from `bert-base-multilingual-cased` on Tibetan data.**

It was trained as part of the Intelexsus project on a mixed Tibetan corpus that includes:
- Tibetan text written in the original Tibetan script (Unicode)
- Data originally in Wylie transliteration that was converted into Tibetan script

The aim is to improve Tibetan representations for downstream tasks while preserving compatibility with multilingual BERT.

## Model Details
- **Base model**: `bert-base-multilingual-cased`
- **Language**: Tibetan (bo)
- **Training objective**: Masked Language Modeling (MLM)
- **Architecture**: 12-layer, 768-hidden, 12-heads
- **Tokenizer**: WordPiece tokenizer compatible with mBERT (includes Tibetan Unicode support)

## How to Use
You can use this model directly with the `transformers` library for the fill-mask task.

```python
from transformers import pipeline

model_name = "OMRIDRORI/mbert-tibetan-continual-unicode-240k"
unmasker = pipeline("fill-mask", model=model_name)

# Example sentence in Tibetan (demonstrative only)
result = unmasker("བོད་ཡིག་ [MASK] ཡིན་པ་རེད།")
print(result)
```

You can also load the model and tokenizer directly for more control:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "OMRIDRORI/mbert-tibetan-continual-unicode-240k"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
# You can now use the model for your own fine-tuning and inference tasks.
```

## Training Data
The continual training used a Tibetan corpus consisting of:
- Native Tibetan text in Unicode (U+0F00–U+0FFF block)
- Wylie transliterated data converted into Tibetan script prior to training

This combination aims to cover both native-script Tibetan and content originally prepared in transliteration that has been normalized to Unicode Tibetan.

## Intended Use and Limitations
This model is intended for research and downstream tasks involving Tibetan. It may contain biases present in the training data and may not perform well outside the Tibetan domain.

## Citation
If you use this model, please cite the Intelexsus project or link to the model page: `https://huggingface.co/OMRIDRORI/mbert-tibetan-continual-unicode-240k`
