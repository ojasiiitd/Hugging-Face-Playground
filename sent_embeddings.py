from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = [
    'I am working',
    'My brother is in the hospital',
    'The leaves are glistening with dew'
    ]
sentences2 = [
    'I am at work',
    'The doctors are doing their best',
    'Even the birds have started chirping late'
    ]

embeddings1 = model.encode(sentences1 , convert_to_tensor = True)
embeddings2 = model.encode(sentences2 , convert_to_tensor = True)

from sentence_transformers import util

cosine_scores = util.cos_sim(embeddings1 , embeddings2)

for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],sentences2[i],cosine_scores[i][i]))

'''
I am working 		 I am at work 		 Score: 0.8367
My brother is in the hospital 		 The doctors are doing their best 		 Score: 0.4257
The leaves are glistening with dew 		 Even the birds have started chirping late 		 Score: 0.1377
'''

# garbage collector cleanup
import gc
del model , embeddings1 , embeddings2
gc.collect()
