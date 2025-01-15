from transformers import logging
logging.set_verbosity_error()

from transformers import pipeline
import torch

translator = pipeline(task="translation" , model="facebook/nllb-200-distilled-600M" , torch_dtype=torch.bfloat16)

text = """\
I went to the train station and found a hideous red coat there.
"""

text_nihon = translator(text , scr_lang="eng_Latn" , tgt_lang="jpn_Jpan")

print(translator)

# garbage collector cleanup
import gc
del translator
gc.collect()


summarizer = pipeline(task="summarization", model="./models/facebook/bart-large-cnn", torch_dtype=torch.bfloat16) # bfloat 16 to compress the model

text = """Upon India's independence on 15 August 1947, the new prime minister Jawaharlal Nehru invited Ambedkar to serve as the Dominion of India's Law Minister; two weeks later, he was appointed Chairman of the Drafting Committee of the Constitution for the future Republic of India.

On 25 November 1949, Ambedkar in his concluding speech in constituent assembly said:

    "The credit that is given to me does not really belong to me. It belongs partly to Sir B.N. Rau the Constitutional Advisor to the Constituent Assembly who prepared a rough draft of the Constitution for the consideration of the Drafting Committee." 

Indian constitution guarantees and protections for a wide range of civil liberties for individual citizens, including freedom of religion, the abolition of untouchability, and the outlawing of all forms of discrimination. Ambedkar was one of the ministers who argued for extensive economic and social rights for women, and won the Assembly's support for introducing a system of reservations of jobs in the civil services, schools and colleges for members of scheduled castes and scheduled tribes and Other Backward Class, a system akin to affirmative action. India's lawmakers hoped to eradicate the socio-economic inequalities and lack of opportunities for India's depressed classes through these measures. The Constitution was adopted on 26 November 1949 by the Constituent Assembly.

Ambedkar expressed his disapproval for the constitution in 1953 during a parliament session and said "People always keep on saying to me "Oh you are the maker of the constitution". My answer is I was a hack. What I was asked to do, I did much against my will." Ambedkar added that, "I am quite prepared to say that I shall be the first person to burn it out. I do not want it. It does not suit anybody." 
"""

summary = summarizer(text , min_length=100 , max_length=300)

print(summary)

'''
[{'summary_text': "Ambedkar was appointed as the Dominion of India's Law Minister on 15 August 1947. Two weeks later, he was appointed Chairman of the Drafting Committee of the Constitution for the future Republic of India. The Constitution was adopted on 26 November 1949 by the Constituent Assembly. Ambedkar was one of the ministers who argued for extensive economic and social rights for women. He won the Assembly's support for introducing a system of reservations of jobs in the civil services, schools and colleges for members of scheduled castes and scheduled tribes and Other Backward Class."}]
'''