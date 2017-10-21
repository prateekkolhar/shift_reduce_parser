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
(feature_indexer, label_indexer, decision_state_cache, feature_cache) = m2.generate_features(train, True, False)

########### all train
trained_model = m2.train_greedy_model(train, 10, feature_indexer, label_indexer, decision_state_cache, feature_cache)
print "Parsing dev"

dev_decoded = [trained_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)


############# smaller size train greedy train. beam parse
(feature_indexer, label_indexer, decision_state_cache, feature_cache) = m2.generate_features(train, True)
train_1000 = train[0:1000]
greedy_model = m2.train_greedy_model(train, 10, feature_indexer, label_indexer, decision_state_cache, feature_cache)

dev_decoded = [greedy_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)


beamed_model = m2.BeamedModel(greedy_model.feature_indexer,greedy_model.feature_weights,3)
print "Parsing dev"

dev_decoded = [beamed_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)




############ smaller size train beam train. beam parse
train_1000 = train[0:1000]
print ""

i=4
print "beam size:"+str(i)
beamed_model = m2.train_beamed_model(train_1000, feature_indexer, decision_state_cache, 5,i+1)
print "Parsing dev"
dev_decoded = [beamed_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)
greedy_model = m2.GreedyModel(beamed_model.feature_indexer,beamed_model.feature_weights)
dev_decoded = [greedy_model.parse(sentence) for sentence in dev]
m2.print_evaluation(dev, dev_decoded)




reload(m2)