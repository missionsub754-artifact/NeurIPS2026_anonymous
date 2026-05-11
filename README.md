# X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding
---
This is anonymized code for "X-CoSD: Communication-Efficient Cross-Vocabulary Collaborative Speculative Decoding."  Parts of the code are adapted from [MCSD](https://github.com/NJUNLP/MCSD/tree/main/MCSD). 

---
## Environments
---
- Python version: 3.9.18
- Pytorch version: 11.8
- transformers version: 4.57.0
## Run code
---
We provide a script for run code:
```
python evaluation.py \
--draft-model <PATH_TO_DRAFT_MODEL> \
--target-model <PATH_TO_TARGET_MODEL> \
--fp16 \
--n-config <SET_CANDIDATE_LENGTH> \
--datapath <PATH_TO_DATA> \
--verification-method <SET_METHOD>
```

Note:
- If you want to run SLM or LLM only, please activate --run-baseline option.
