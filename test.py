# -*- coding: utf-8 -*-

from pprint import pprint
import tfidf_similarity as sim

test = ["the dogs are nice", "cats are nice", "Who is this dog?", 
        "I don't like cats", "dogs", "I like dogs", "the tree is nice", "things are nice"]
test = list(map(lambda x:sim.preprocess(x), test))
query = sim.preprocess("cats, dogs and things are nice")
cos_sim, tfidf_vectorizer, test = sim.vectorize_sim_search(query, test)
res = sim.get_most_similar(test, cos_sim, len(test))
pprint(res)
