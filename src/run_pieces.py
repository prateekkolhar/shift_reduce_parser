'''
Created on Oct 7, 2017

@author: Prateek Kolhar
'''

import os
os.chdir("D:\\Programming\\eclipse-workspace-java\\nlp2_shift_reduce_parser\\src")

import models2 as m2

from random import shuffle

train = m2.read_data("./data/train.conllx")
shuffle(train)
dev = m2.read_data("data/dev.conllx")
(feature_indexer, label_indexer, decision_state_cache, feature_cache) = m2.generate_features(train, True)

########### all train
trained_model = m2.train_greedy_model(train, 10, feature_indexer, label_indexer, decision_state_cache, feature_cache)
print "Parsing dev"

dev_decoded = [trained_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)


############# smaller size train greedy train. beam parse

train_1000 = train[0:1000]
greedy_model = m2.train_greedy_model(train, 10, feature_indexer, label_indexer, decision_state_cache, feature_cache)
trained_model = m2.BeamedModel(greedy_model.feature_indexer,greedy_model.feature_weights,3)
print "Parsing dev"

dev_decoded = [trained_model.parse(sentence) for sentence in [dev[1]]]
m2.print_evaluation(dev, dev_decoded)


dev_decoded = [greedy_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)

############ smaller size train beam train. beam parse
train_1000 = train[0:1000]
trained_model = m2.train_beamed_model(train_1000, feature_indexer, decision_state_cache, 10)

print "Parsing dev"

dev_decoded = [trained_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)

reload(m2)