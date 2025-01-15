'''
    !pip install transformers
    !pip install datasets
    !pip install soundfile
    !pip install librosa
'''

from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset, load_from_disk

# This dataset is a collection of different sounds of 5 seconds
dataset = load_from_disk("./models/ashraq/esc50/train")


from transformers import pipeline

zero_shot_classifier = pipeline(task="zero-shot-audio-classification" , model="./models/laion/clap-htsat-unfused")

print(zero_shot_classifier.feature_extractor.sampling_rate)

print(audio_sample["audio"]["sampling_rate"])

from datasets import Audio

dataset = dataset.cast_column("audio" , Audio(sampling_rate=48_000))

audio_sample = dataset[0]
print(audio_sample)

# labels for Zero-Shot classification need to be provided by us
candidate_labels = ["Sound of a dog" ,  "Sound of vacuum cleaner"]

result = zero_shot_classifier(audio_sample["audio"]["array"] , candidate_labels=candidate_labels)

print(result)
'''
[{'score': 0.9985589385032654, 'label': 'Sound of a dog'},
 {'score': 0.0014411123702302575, 'label': 'Sound of vacuum cleaner'}]
'''

# new unrelated labels
candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]

print( zero_shot_classifier(audio_sample["audio"]["array"], candidate_labels=candidate_labels) )

'''
Limitation: model tries to fit in the given candidate labels, even if its not the sound of a bird singing
[{'score': 0.6172530055046082, 'label': 'Sound of a bird singing'},
 {'score': 0.21602635085582733, 'label': 'Sound of vacuum cleaner'},
 {'score': 0.12547191977500916, 'label': 'Sound of an airplane'},
 {'score': 0.04124866798520088, 'label': 'Sound of a child crying'}]
'''