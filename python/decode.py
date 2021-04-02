import torch
import torchaudio

from datasets import load_dataset

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

#
test_dataset = load_dataset("common_voice", "cy", split="test[:1]")

processor = Wav2Vec2Processor.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
model = Wav2Vec2ForCTC.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    print (batch["path"])
    print (batch["sentence"])
    print (len(batch))
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:1], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

print ("logits")
print(logits)

predicted_ids = torch.argmax(logits, dim=-1)

print ("predicted_ids")
print (predicted_ids)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Prediction:", processor.decode(predicted_ids[0]))
print("Reference:", test_dataset["sentence"][:1])

print (" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))

