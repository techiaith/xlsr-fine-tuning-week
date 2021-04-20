import torch
import librosa

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from argparse import ArgumentParser, RawTextHelpFormatter

DESCRIPTION = """

 Prifysgol Bangor University

"""

processor = Wav2Vec2Processor.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")
model = Wav2Vec2ForCTC.from_pretrained("DewiBrynJones/wav2vec2-large-xlsr-welsh")


def greedy_decode(logits):
    predicted_ids=torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def main(audio_file, **args):
    audio, rate = librosa.load(audio_file, sr=16000)
    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    output=greedy_decode(logits)
    print (output)


if __name__ == "__main__":

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)

    parser.add_argument("--wav", dest="audio_file", required=True)
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
