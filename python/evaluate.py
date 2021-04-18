import torch
import torchaudio
import json
import numpy as np

from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import text_preprocess

import ctcdecode

test_dataset = load_dataset("common_voice", "cy", split="test")
wer = load_metric("wer")

#processor = Wav2Vec2Processor.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
#model = Wav2Vec2ForCTC.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
processor = Wav2Vec2Processor.from_pretrained("/models/published/wav2vec2-large-xlsr-welsh")
model = Wav2Vec2ForCTC.from_pretrained("/models/published/wav2vec2-large-xlsr-welsh")
model.to("cuda")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

vocab=list()
vocab_file_path=processor.tokenizer.save_vocabulary('/tmp')[0]
with open(vocab_file_path, 'r', encoding='utf-8') as vocab_file:
    vocabjson = json.load(vocab_file)
    ids=range(0,  len(vocabjson))
    vocab = processor.tokenizer.convert_ids_to_tokens(ids)        
                                                   
    space_ix = vocab.index('|')
    padding_ix = vocab.index('[PAD]')

    vocab[space_ix]=' '
    vocab[padding_ix]='_'

kenlm_scorer=ctcdecode.WordKenLMScorer('/models/kenlm/lm_filtered.arpa', alpha=2.5, beta=0)
ctcdecoder=ctcdecode.BeamSearchDecoder(vocab,
                        num_workers=4,
                        beam_width=12,
                        scorers=[kenlm_scorer],
                        cutoff_prob=np.log(0.0001),
                        cutoff_top_n=40)



# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = text_preprocess.cleanup(batch["sentence"]) + " "
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
       logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_strings"] = processor.batch_decode(pred_ids)
    return batch


def evaluate_with_lm(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
       logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    c,x,y = logits.size()

    decoded=list()
    for i in range(0,c):
        ctc = torch.softmax(logits[i], dim=-1).cpu().detach().numpy()
        ctc = np.log(ctc)
        decoded.append(ctcdecoder.decode(ctc))
    
    batch["pred_strings"] = decoded
    return batch




result = test_dataset.map(evaluate, batched=True, batch_size=8)
print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))


lm_result = test_dataset.map(evaluate_with_lm, batched=True, batch_size=8)
print("WER with LM: {:2f}".format(100 * wer.compute(predictions=lm_result["pred_strings"], references=lm_result["sentence"])))


