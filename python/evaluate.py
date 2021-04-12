import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

test_dataset = load_dataset("common_voice", "cy", split="test")
wer = load_metric("wer")

#processor = Wav2Vec2Processor.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
#model = Wav2Vec2ForCTC.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
processor = Wav2Vec2Processor.from_pretrained("/models/published/wav2vec2-large-xlsr-welsh")
model = Wav2Vec2ForCTC.from_pretrained("/models/published/wav2vec2-large-xlsr-welsh")
model.to("cuda")

chars_to_ignore_regex = '[\,\?\.\!\-\u2013\u2014\u00AC\;\:\"\\%\\\]'
resampler = torchaudio.transforms.Resample(48_000, 16_000)


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    batch["sentence"] = batch["sentence"].replace('\u2018',"'")
    batch["sentence"] = batch["sentence"].replace('ñ',"n")
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

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))

