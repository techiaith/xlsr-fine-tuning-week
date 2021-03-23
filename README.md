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

Yn ol set profi CommonVoice Cymraeg, mae gan y model Word Error Rate (WER) o **27.21%**. // *According to the Welsh CommonVoice test set, the model has a Word Error Rate (WER) of **27.21%***

```
root@bff0be8425ea:/usr/src/xlsr-finetune# python3 evaluate.py                                                                                                         /usr/local/lib/python3.6/dist-packages/torchaudio/backend/utils.py:54: UserWarning: "sox" backend is being deprecated. The default backend will be changed to "sox_io" backend in 0.8.0 and "sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. Please refer to https://github.com/pytorch/audio/issues/903 for the detail.
  '"sox" backend is being deprecated. '
Reusing dataset common_voice (/root/.cache/huggingface/datasets/common_voice/cy/6.1.0/0041e06ab061b91d0a23234a2221e87970a19cf3a81b20901474cffffeb7869f)
Special tokens have been added in the vocabulary, make sure the associated word embedding are fine-tuned or trained.
WER: 27.21%
```

```
root@bff0be8425ea:/usr/src/xlsr-finetune# python3 decode.py                                                                                                           /usr/local/lib/python3.6/dist-packages/torchaudio/backend/utils.py:54: UserWarning: "sox" backend is being deprecated. The default backend will be changed to "sox_io" backend in 0.8.0 and "sox" backend will be removed in 0.9.0. Please migrate to "sox_io" backend. Please refer to https://github.com/pytorch/audio/issues/903 for the detail.
  '"sox" backend is being deprecated. '
Reusing dataset common_voice (/root/.cache/huggingface/datasets/common_voice/cy/6.1.0/0041e06ab061b91d0a23234a2221e87970a19cf3a81b20901474cffffeb7869f)
Downloading: 100%|| 158/158 [00:00<00:00, 298kB/s]Downloading: 100%|| 481/481 [00:00<00:00, 921kB/s]Downloading: 100%|| 138/138 [00:00<00:00, 261kB/s]Downloading: 100%|| 85.0/85.0 [00:00<00:00, 154kB/s]Special tokens have been added in the vocabulary, make sure the associated word embedding are fine-tuned or trained.
Downloading: 100%|| 1.56k/1.56k [00:00<00:00, 2.96MB/s]Downloading: 100%|| 1.26G/1.26G [01:49<00:00, 11.5MB/s]100%|| 96/96 [00:00<00:00, 139.20ex/s]

Prediction: ['rhedais i ffwrdd heb ddweud dim wrthi ym beth digwyddodd', 'ac yr oedd y ferch yn ofnus d']
Reference: ['Rhedais i ffwrdd heb ddweud dim wrthi am beth ddigwyddodd.', 'Ac yr oedd y ferch yn ofnus.']

```
