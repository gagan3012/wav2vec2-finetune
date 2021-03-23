# Wav2Vec2-Large-XLSR-53-Nepali 

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on Nepali using the [Common Voice](https://huggingface.co/datasets/common_voice), and [OpenSLR ne](http://www.openslr.org/43/). 

When using this model, make sure that your speech input is sampled at 16kHz.

## Usage

The model can be used directly (without a language model) as follows:

```python
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

!wget https://www.openslr.org/resources/43/ne_np_female.zip
!unzip ne_np_female.zip
!ls ne_np_female

colnames=['path','sentence'] 
df  = pd.read_csv('/content/ne_np_female/line_index.tsv',sep='\\t',header=None,names = colnames)
df['path'] = '/content/ne_np_female/wavs/'+df['path'] +'.wav'

train, test = train_test_split(df, test_size=0.1)

test.to_csv('/content/ne_np_female/line_index_test.csv')

test_dataset = load_dataset('csv', data_files='/content/ne_np_female/line_index_test.csv',split = 'train')

processor = Wav2Vec2Processor.from_pretrained("gagan3012/wav2vec2-xlsr-nepali")
model = Wav2Vec2ForCTC.from_pretrained("gagan3012/wav2vec2-xlsr-nepali") 

resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
\tspeech_array, sampling_rate = torchaudio.load(batch["path"])
\tbatch["speech"] = resampler(speech_array).squeeze().numpy()
\treturn batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"][:2], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
\tlogits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)

print("Prediction:", processor.batch_decode(predicted_ids))
print("Reference:", test_dataset["sentence"][:2])

```
#### Result 

Prediction: ['पारानाको ब्राजिली राज्यमा रहेको राजधानी', 'देवराज जोशी त्रिभुवन विश्वविद्यालयबाट शिक्षाशास्त्रमा स्नातक हुनुहुन्छ']

Reference: ['पारानाको ब्राजिली राज्यमा रहेको राजधानी', 'देवराज जोशी त्रिभुवन विश्वविद्यालयबाट शिक्षाशास्त्रमा स्नातक हुनुहुन्छ']

## Evaluation

The model can be evaluated as follows on the {language} test data of Common Voice.  # TODO: replace #TODO: replace language with your {language}, *e.g.* French


```python
import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re

!wget https://www.openslr.org/resources/43/ne_np_female.zip
!unzip ne_np_female.zip
!ls ne_np_female

colnames=['path','sentence'] 
df  = pd.read_csv('/content/ne_np_female/line_index.tsv',sep='\\t',header=None,names = colnames)
df['path'] = '/content/ne_np_female/wavs/'+df['path'] +'.wav'

train, test = train_test_split(df, test_size=0.1)

test.to_csv('/content/ne_np_female/line_index_test.csv')

test_dataset = load_dataset('csv', data_files='/content/ne_np_female/line_index_test.csv',split = 'train')
wer = load_metric("wer")

processor = Wav2Vec2Processor.from_pretrained("gagan3012/wav2vec2-xlsr-nepali")
model = Wav2Vec2ForCTC.from_pretrained("gagan3012/wav2vec2-xlsr-nepali") 
model.to("cuda")

chars_to_ignore_regex = '[\\,\\?\\.\\!\\-\\;\\:\\"\\“]'  
resampler = torchaudio.transforms.Resample(48_000, 16_000)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
\tbatch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
\tspeech_array, sampling_rate = torchaudio.load(batch["path"])
\tbatch["speech"] = resampler(speech_array).squeeze().numpy()
\treturn batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

# Preprocessing the datasets.
# We need to read the aduio files as arrays
def evaluate(batch):
\tinputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

\twith torch.no_grad():
\t\tlogits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

\tpred_ids = torch.argmax(logits, dim=-1)
\tbatch["pred_strings"] = processor.batch_decode(pred_ids)
\treturn batch

result = test_dataset.map(evaluate, batched=True, batch_size=8)

print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))

```

**Test Result**: 5.970952 %  

## Training

The script used for training can be found [here](https://colab.research.google.com/drive/1AHnYWXb5cwfMEa2o4O3TSdasAR3iVBFP?usp=sharing) 
