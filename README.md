# xlsr-fine-tuning-week

Cod i cyd fynd a wythnos mireinio model XLSR Facebook gyda HuggingFace ar gyfer y Gymraeg // *Code for fine tuning Facebook XLSR model with HuggingFace for Welsh.*

Gweler https://huggingface.co/DewiBrynJones/wav2vec2-large-xlsr-welsh ar gyfer model wedi'i hyfforddi eisoes. // *See https://huggingface.co/DewiBrynJones/wav2vec2-large-xlsr-welsh for a pre-trained model.*

# Sut i'w ddefnyddio...  // *How to use...*

`$ make`

`$ make run `

`root@bff0be8425ea:/usr/src/xlsr-finetune# python3 run.py`

Yn dibynnu ar y cerdyn graffics, bydd yn gymryd rhai oriau i hyfforddi. // *Depending on your graphics card, it will take some hours to train.* 

Ar GeForce RTX 2080, mae'n cymryd hyd at 13 awr // *On a GeForce RTX 2080 it takes up to 13 hours.* 



# Gwerthuso // *Evaluation*

Yn ol set profi CommonVoice Cymraeg, mae gan y model Word Error Rate (WER) o **25.31%**. // *According to the Welsh CommonVoice test set, the model has a Word Error Rate (WER) of **25.31%***.

```
root@26215fd43562:/usr/src/xlsr-finetune# python3 evaluate.py
Reusing dataset common_voice (/root/.cache/huggingface/datasets/common_voice/cy/6.1.0/0041e06ab061b91d0a23234a2221e87970a19cf3a81b20901474cffffeb7869f)
Loading the LM will be faster if you build a binary file.
Reading /models/kenlm/lm_filtered.arpa
----5---10---15---20---25---30---35---40---45---50---55---60---65---70---75---80---85---90---95--100
****************************************************************************************************
found 1gram
found 2gram
Loading cached processed dataset at /root/.cache/huggingface/datasets/common_voice/cy/6.1.0/0041e06ab061b91d0a23234a2221e87970a19cf3a81b20901474cffffeb7869f/cache-7bc763ec5b45ddb8.arrow
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 603/603 [08:35<00:00,  1.17ba/s]
WER: 25.315298
WER with LM: 22.783297
```

```
root@26215fd43562:/usr/src/xlsr-finetune# python3 decode.py --wav speech.wav
mae ganddynt ddau o blant mab a merch
```
