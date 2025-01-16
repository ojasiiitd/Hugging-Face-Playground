'''
!pip install transformers
!pip install gradio
!pip install timm
!pip install timm
!pip install inflect
!pip install phonemizer
'''

from transformers.utils import logging

logging.set_verbosity_error()

from transformers import pipeline

narrator = pipeline("text-to-speech", model="./models/kakao-enterprise/vits-ljs")

text = """
Researchers at the Allen Institute for AI, \
HuggingFace, Microsoft, the University of Washington, \
Carnegie Mellon University, and the Hebrew University of \
Jerusalem developed a tool that measures atmospheric \
carbon emitted by cloud servers while training machine \
learning models. After a model’s size, the biggest variables \
were the server’s location and time of day it was active.
"""

narrated_text = narrator(text)

from IPython.display import Audio as IPythonAudio

IPythonAudio(narrated_text["audio"][0], rate=narrated_text["sampling_rate"])
