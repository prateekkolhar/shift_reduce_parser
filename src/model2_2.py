'''
Created on Oct 8, 2017

@author: Prateek Kolhar
'''
# models.py

import sys
from utils import *
from adagrad_trainer import *
from treedata import *
import numpy as np
import timeit


# Greedy parsing model. This model treats shift/reduce decisions as a multiclass classification problem.
class GreedyModel(object):
    def __init__(self, feature_indexer, feature_weights):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):

        label_indexer = get_label_indexer()
        parser = initial_parser_state(len(sentence))
        while not parser.is_finished():
            if len(parser.stack) < 2:
                result = "S"
            else:
                pred = np.zeros(len(label_indexer))
                feature_cache = [[] for k in xrange(0, len(label_indexer))]
                
                for label_idx in xrange(0,len(label_indexer)):
                    feature_cache= extract_features(
                      self.feature_indexer,sentence, parser, label_indexer.get_object(label_idx), False)
                    pred[label_idx] = self.feature_weights[feature_cache].sum()
                
                result_idx = pred.argmax();
                
                if parser.stack_two_back() !=-1 and label_indexer.get_object(result_idx)=="L":
                    result = "L"
                elif parser.buffer_len()!=0 and label_indexer.get_object(result_idx)=="S":
                    result = "S"
                else:
                    result = "R"
                
            parser = parser.take_action(result)
        
        return ParsedSentence(sentence.tokens, parser.get_dep_objs(len(sentence)))
        
#         raise Exception("IMPLEMENT ME")


# Beam-search-based global parsing model. Shift/reduce decisions are still modeled with local features, but scores are
# accumulated over the whole sequence of decisions to give a "global" decision.


class BeamedModel(object):
    def __init__(self, feature_indexer, feature_weights, beam_size=1):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.beam_size = beam_size
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def compute_next_state(self, sentence, prev_state, prev_score, action, add_to_indexer=False):
        feature_vec = extract_features(self.feature_indexer,sentence, prev_state, action, add_to_indexer)
        new_state= prev_state.take_action(action);
        new_score = self.feature_weights[feature_vec].sum()+prev_score;
        
        return (feature_vec,new_state,new_score);
        
        
        
    def parse(self, sentence):
        
        label_indexer = get_label_indexer()
        init_state = initial_parser_state(len(sentence))
        b_c = Beam(self.beam_size)
        b_p = Beam(self.beam_size)
        
        b_p.add(init_state, 0)
        
        for i in xrange(len(sentence)*2):
            for j in xrange(len(b_p.elts)):
                prev_state = b_p.elts[j];
                prev_score = b_p.scores[j];
                if len(prev_state.stack) < 2:
                    action = "S"
                    (feature_vec,new_state,new_score) = self.compute_next_state(sentence, prev_state, prev_score, action)
                    b_c.add(new_state, new_score)
                else:
                    if prev_state.stack_two_back() !=-1:
                        action = "L"
                        (feature_vec,new_state,new_score) = self.compute_next_state(sentence, prev_state, prev_score, action)
                        b_c.add(new_state, new_score)
                    if prev_state.buffer_len()!=0:
                        action = "S"
                        (feature_vec,new_state,new_score) = self.compute_next_state(sentence, prev_state, prev_score, action)
                        b_c.add(new_state, new_score) 
                    if True:
                        action = "R"
                        (feature_vec,new_state,new_score) = self.compute_next_state(sentence, prev_state, prev_score, action)
                        b_c.add(new_state, new_score) 
            b_p = b_c
            b_c = Beam(self.beam_size)
#             print len(b_p.elts[0].deps)
            
        pred = b_p.elts[0]
        return ParsedSentence(sentence.tokens, pred.get_dep_objs(len(sentence)))
# Stores state of a shift-reduce parser, namely the stack, buffer, and the set of dependencies that have
# already been assigned. Supports various accessors as well as the ability to create new ParserStates
# from left_arc, right_arc, and shift.
class ParserState(object):
    # stack and buffer are lists of indices
    # The stack is a list with the top of the stack being the end
    # The buffer is a list with the first item being the front of the buffer (next word)
    # deps is a dictionary mapping *child* indices to *parent* indices
    # (this is the one-to-many map; parent-to-child doesn't work in map-like data structures
    # without having the values be lists)
    def __init__(self, stack, buffer, deps):
        self.stack = stack
        self.buffer = buffer
        self.deps = deps

    def __repr__(self):
        return repr(self.stack) + " " + repr(self.buffer) + " " + repr(self.deps)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.stack == other.stack and self.buffer == other.buffer and self.deps == other.deps
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def stack_len(self):
        return len(self.stack)

    def buffer_len(self):
        return len(self.buffer)

    def is_legal(self):
        return self.stack[0] == -1

    def is_finished(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def buffer_head(self):
        return self.get_buffer_word_idx(0)

    # Returns the buffer word at the given index
    def get_buffer_word_idx(self, index):
        if index >= len(self.buffer):
            raise Exception("Can't take the " + repr(index) + " word from the buffer of length " + repr(len(self.buffer)) + ": " + repr(self))
        return self.buffer[index]

    # Returns True if idx has all of its children attached already, False otherwise
    def is_complete(self, idx, parsed_sentence):
        _is_complete = True
        for child in xrange(0, len(parsed_sentence)):
            if parsed_sentence.get_parent_idx(child) == idx and (child not in self.deps.keys() or self.deps[child] != idx):
                _is_complete = False
        return _is_complete

    def stack_head(self):
        if len(self.stack) < 1:
            raise Exception("Can't go one back in the stack if there are no elements: " + repr(self))
        return self.stack[-1]

    def stack_two_back(self):
        if len(self.stack) < 2:
            raise Exception("Can't go two back in the stack if there aren't two elements: " + repr(self))
        return self.stack[-2]

    # Returns a new ParserState that is the result of taking the given action.
    # action is a string, either "L", "R", or "S"
    def take_action(self, action):
        if action == "L":
            return self.left_arc()
        elif action == "R":
            return self.right_arc()
        elif action == "S":
            return self.shift()
        else:
            raise Exception("No implementation for action " + action)

    # Returns a new ParserState that is the result of applying left arc to the current state. May crash if the
    # preconditions for left arc aren't met.
    def left_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_two_back(): self.stack_head()})
        new_stack = list(self.stack[0:-2])
        new_stack.append(self.stack_head())
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying right arc to the current state. May crash if the
    # preconditions for right arc aren't met.
    def right_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_head(): self.stack_two_back()})
        new_stack = list(self.stack[0:-1])
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying shift to the current state. May crash if the
    # preconditions for right arc aren't met.
    def shift(self):
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], self.deps)

    # Return the Dependency objects corresponding to the dependencies added so far to this ParserState
    def get_dep_objs(self, sent_len):
        dep_objs = []
        for i in xrange(0, sent_len):
            dep_objs.append(Dependency(self.deps[i], "?"))
        return dep_objs


# Returns an initial ParserState for a sentence of the given length. Note that because the stack and buffer
# are maintained as indices, knowing the words isn't necessary.
def initial_parser_state(sent_len):
    return ParserState([-1], range(0, sent_len), {})


# Returns an indexer for the three actions so you can iterate over them easily.
def get_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("S")
    label_indexer.get_index("L")
    label_indexer.get_index("R")
    return label_indexer


# Returns a GreedyModel trained over the given treebank.

def generate_features(parsed_sentences, add_to_indexer, feature_indexer=-1):
    
    if feature_indexer==-1: feature_indexer = Indexer();
    label_indexer = get_label_indexer();
    
    print "decision state cache generation:"
    start = timeit.default_timer()
    decision_state_cache=[]
    for sentence in parsed_sentences:
        dec_seq = get_decision_sequence(sentence)
        decision_state_cache.append((dec_seq[0],dec_seq[1][:-1]))
    stop = timeit.default_timer()
    print "decision_state_cache extracted:" + str(stop-start)
    
    
    feature_cache = [[[[] for k in xrange(0, len(label_indexer))] for j in xrange(0, len(decision_state_cache[i][0]))] for i in xrange(0, len(parsed_sentences))]
    
    if add_to_indexer: print "Train feature extraction"
    else: print "Test feature extraction"
    start = timeit.default_timer()
    for sentence_idx in xrange(0, len(parsed_sentences)):
        if sentence_idx%100 == 0: print str(sentence_idx)+"/"+ str(len(parsed_sentences))
        for state_idx in xrange(0,len(decision_state_cache[sentence_idx][0])):
            for label_idx in xrange(0,len(label_indexer)):
                feature_cache[sentence_idx][state_idx][label_idx] = extract_features(
                    feature_indexer,parsed_sentences[sentence_idx], decision_state_cache[sentence_idx][1][state_idx], label_indexer.get_object(label_idx), add_to_indexer)
    stop = timeit.default_timer()
    print "features extracted:" + str(stop-start)
    return (feature_indexer, label_indexer, decision_state_cache, feature_cache)

def train_greedy_model(parsed_sentences, epoch_num, feature_indexer=-1, label_indexer=-1, decision_state_cache=-1, feature_cache=-1 ):
    
    if feature_indexer==-1:
        (feature_indexer, label_indexer, decision_state_cache, feature_cache)= generate_features(parsed_sentences,True)
    wt = np.random.rand(len(feature_indexer))
    wt_a = np.random.rand(len(feature_indexer))
    c=1
    lr=1
    print "Train"
    
    start = timeit.default_timer()
    for epochs in xrange(epoch_num):
        print "epochs:" +str(epochs)
        for sentence_idx in xrange(0, len(parsed_sentences)):
            for state_idx in xrange(0,len(decision_state_cache[sentence_idx][1])):
                #####Train#####
                pred = np.zeros(len(label_indexer))
                
                for label_idx  in xrange(0,len(label_indexer)):
                    pred[label_idx] = wt[feature_cache[sentence_idx][state_idx][label_idx]].sum()
                    
#                 pred[label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]+=1
                pred_label_idx = pred.argmax();
                wt[feature_cache[sentence_idx][state_idx][pred_label_idx]] -= 1*lr 
                wt[feature_cache[sentence_idx][state_idx][label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]] += 1*lr
                
                wt_a[feature_cache[sentence_idx][state_idx][pred_label_idx]] -= 1*c*lr 
                wt_a[feature_cache[sentence_idx][state_idx][label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]] += 1*c*lr
            c+=1
    stop = timeit.default_timer()
    print "train complete:" + str(stop-start)
    return GreedyModel(feature_indexer, wt-wt_a/c);


# Returns a BeamedModel trained over the given treebank.

def compute_next_state(sentence, gold_action, prev_state, prev_score, prev_cum_f,prev_avg_cum_f, prev_is_gold, c,feature_weights, action, feature_indexer, add_to_indexer=False):
    feature_vec = extract_features(feature_indexer,sentence, prev_state, action, add_to_indexer)
    new_state= prev_state.take_action(action);
    new_score = feature_weights[feature_vec].sum()+prev_score;
    new_cum_f = Counter()
    new_cum_f.add(prev_cum_f)
    new_cum_f.increment_all(feature_vec,1)
    
    new_avg_cum_f = Counter()
    new_avg_cum_f.add(prev_avg_cum_f)
    new_avg_cum_f.increment_all(feature_vec,c)
    
    new_is_gold = (gold_action==action) & prev_is_gold
    return (new_state,new_score,new_cum_f,new_avg_cum_f, new_is_gold)


def train_beamed_model(parsed_sentences, feature_indexer, gold_state_cache, epoch_num=10, beam_size=3):
    label_indexer = get_label_indexer()

    wt = np.random.rand(len(feature_indexer))
    wt_a = np.random.rand(len(feature_indexer))
    c=1
    lr=1
    
    start = timeit.default_timer()
    
    print "train start"
    
    for epochs in xrange(epoch_num):
        print "epochs:" +str(epochs)
        for sentence_idx in xrange(0, len(parsed_sentences)):
            
            sentence = parsed_sentences[sentence_idx]
            b_c = Beam(beam_size)
            b_p = Beam(beam_size)
            is_gold = True
            init_state = initial_parser_state(len(sentence))
            cum_f = Counter()
            avg_cum_f = Counter()
            gold_cum_f= Counter()
            gold_avg_cum_f = Counter()
            b_p.add((init_state,cum_f,avg_cum_f,is_gold), 0, is_gold)
            gold_in_beam = True
            for state_idx in xrange(0,len(sentence)*2):
                gold_cum_f.increment_all(gold_state_cache[sentence_idx][1][state_idx], 1) 
                gold_cum_f.increment_all(gold_state_cache[sentence_idx][1][state_idx], c)
                for prev_elt_idx in xrange(len(b_p.elts)):
                    
                    prev_state = b_p.elts[prev_elt_idx][0]
                    prev_cum_f = b_p.elts[prev_elt_idx][1]
                    prev_avg_cum_f = b_p.elts[prev_elt_idx][2]
                    prev_is_gold = b_p.elts[prev_elt_idx][3]
                    prev_score = b_p.scores[prev_elt_idx]
                    
                    if len(prev_state.stack) < 2:
                        action = "S"
                        (new_state,new_score,new_cum_f,new_avg_cum_f, new_is_gold) = compute_next_state(
                            sentence, gold_state_cache[sentence_idx][0][state_idx], prev_state, prev_score, prev_cum_f,
                            prev_avg_cum_f, prev_is_gold, c, wt, action, feature_indexer, True)
                        
                        gold_in_beam = b_c.add((new_state,new_cum_f,new_avg_cum_f), new_score, new_is_gold)
                    else:
                        if prev_state.stack_two_back() !=-1:
                            action = "L"
                            (new_state,new_score,new_cum_f,new_avg_cum_f, new_is_gold) = compute_next_state(
                                sentence, gold_state_cache[sentence_idx][0][state_idx], prev_state, prev_score, prev_cum_f,
                                prev_avg_cum_f, prev_is_gold, c, wt, action, feature_indexer, True)
                            
                            gold_in_beam = b_c.add((new_state,new_cum_f,new_avg_cum_f), new_score, new_is_gold)
                        if prev_state.buffer_len()!=0:
                            action = "S"
                            (new_state,new_score,new_cum_f,new_avg_cum_f, new_is_gold) = compute_next_state(
                                sentence, gold_state_cache[sentence_idx][0][state_idx], prev_state, prev_score, prev_cum_f,
                                prev_avg_cum_f, prev_is_gold, c, wt, action, feature_indexer, True)
                            
                            gold_in_beam = b_c.add((new_state,new_cum_f,new_avg_cum_f), new_score, new_is_gold)
                        if True:
                            action = "R"
                            (new_state,new_score,new_cum_f,new_avg_cum_f, new_is_gold) = compute_next_state(
                                sentence, gold_state_cache[sentence_idx][0][state_idx], prev_state, prev_score, prev_cum_f,
                                prev_avg_cum_f, prev_is_gold, c, wt, action, feature_indexer, True)
                            
                            gold_in_beam = b_c.add((new_state,new_cum_f,new_avg_cum_f), new_score, new_is_gold)
                    
                  
                    
                
                b_p = b_c
                b_c = Beam(beam_size)
                if(not gold_in_beam): 
                    # early update
                    break   
            
            pred =b_p.elts[0][1]
            
            for idx in pred.keys():
                wt[idx]-=pred.get_count(idx)
                wt_a[idx]-=c*wt[idx]
            for idx in gold_cum_f.keys():
                wt[idx]+=gold_cum_f.get_count(idx)
                wt_a[idx]+=c*wt[idx]
            c=c+1
                #global update
                    
                    
                #####Train#####
                
#                 pred = np.zeros(len(label_indexer))
#                 
#                 for label_idx  in xrange(0,len(label_indexer)):
#                     pred[label_idx] = wt[feature_cache[sentence_idx][state_idx][label_idx]].sum()
#                     
# #                 pred[label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]+=1
#                 pred_label_idx = pred.argmax();
#                 wt[feature_cache[sentence_idx][state_idx][pred_label_idx]] -= 1*lr 
#                 wt[feature_cache[sentence_idx][state_idx][label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]] += 1*lr
#                 
#                 wt_a[feature_cache[sentence_idx][state_idx][pred_label_idx]] -= 1*c*lr 
#                 wt_a[feature_cache[sentence_idx][state_idx][label_indexer.get_index(decision_state_cache[sentence_idx][0][state_idx])]] += 1*c*lr
            c+=1
    stop = timeit.default_timer()
    print "train complete:" + str(stop-start)
    return GreedyModel(feature_indexer, wt-wt_a/c);
    


# Extract features for the given decision in the given parser state. Features look at the top of the
# stack and the start of the buffer. Note that this isn't in any way a complete feature set -- play around with
# more of your own!
def extract_features(feat_indexer, sentence, parser_state, decision, add_to_indexer):
    feats = []
    sos_tok = Token("<s>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>")
    if parser_state.stack_len() >= 1:
        head_idx = parser_state.stack_head()
        stack_head_tok = sentence.tokens[head_idx] if head_idx != -1 else root_tok
        if parser_state.stack_len() >= 2:
            two_back_idx = parser_state.stack_two_back()
            stack_two_back_tok = sentence.tokens[two_back_idx] if two_back_idx != -1 else root_tok
        else:
            stack_two_back_tok = sos_tok
    else:
        stack_head_tok = sos_tok
        stack_two_back_tok = sos_tok
    buffer_first_tok = sentence.tokens[parser_state.get_buffer_word_idx(0)] if parser_state.buffer_len() >= 1 else eos_tok
    buffer_second_tok = sentence.tokens[parser_state.get_buffer_word_idx(1)] if parser_state.buffer_len() >= 2 else eos_tok
    # Shortcut for adding features
    def add_feat(feat):
        maybe_add_feature(feats, feat_indexer, add_to_indexer, feat)
    add_feat(decision + ":S0Word=" + stack_head_tok.word)
    add_feat(decision + ":S0Pos=" + stack_head_tok.pos)
    add_feat(decision + ":S0CPos=" + stack_head_tok.cpos)
    add_feat(decision + ":S1Word=" + stack_two_back_tok.word)
    add_feat(decision + ":S1Pos=" + stack_two_back_tok.pos)
    add_feat(decision + ":S1CPos=" + stack_two_back_tok.cpos)
    add_feat(decision + ":B0Word=" + buffer_first_tok.word)
    add_feat(decision + ":B0Pos=" + buffer_first_tok.pos)
    add_feat(decision + ":B0CPos=" + buffer_first_tok.cpos)
    add_feat(decision + ":B1Word=" + buffer_second_tok.word)
    add_feat(decision + ":B1Pos=" + buffer_second_tok.pos)
    add_feat(decision + ":B1CPos=" + buffer_second_tok.cpos)
    add_feat(decision + ":S1S0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos)
    add_feat(decision + ":S0B0Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1B0Pos=" + stack_two_back_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B1Pos=" + stack_head_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":B0B1Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":S0B0WordPos=" + stack_head_tok.word + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0PosWord=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1S0WordPos=" + stack_two_back_tok.word + "&" + stack_head_tok.pos)
    add_feat(decision + ":S1S0PosWord=" + stack_two_back_tok.pos + "&" + stack_head_tok.word)
    add_feat(decision + ":S1S0B0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0B1Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    return feats


# Computes the sequence of decisions and ParserStates for a gold-standard sentence using the arc-standard
# transition framework. We use the minimum stack-depth heuristic, namely that
# Invariant: states[0] is the initial state. Applying decisions[i] to states[i] yields states[i+1].
def get_decision_sequence(parsed_sentence):
    decisions = []
    states = []
    state = initial_parser_state(len(parsed_sentence))
    while not state.is_finished():
        if not state.is_legal():
            raise Exception(repr(decisions) + " " + repr(state))
        # Look at whether left-arc or right-arc would add correct arcs
        if len(state.stack) < 2:
            result = "S"
        else:
            # Stack and buffer must both contain at least one thing
            one_back = state.stack_head()
            two_back = state.stack_two_back()
            # -1 is the ROOT symbol, so this forbids attaching the ROOT as a child of anything
            # (passing -1 as an index around causes crazy things to happen so we check explicitly)
            if two_back != -1 and parsed_sentence.get_parent_idx(two_back) == one_back and state.is_complete(two_back, parsed_sentence):
                result = "L"
            # The first condition should never be true, but doesn't hurt to check
            elif one_back != -1 and parsed_sentence.get_parent_idx(one_back) == two_back and state.is_complete(one_back, parsed_sentence):
                result = "R"
            elif len(state.buffer) > 0:
                result = "S"
            else:
                result = "R" # something went wrong, buffer is empty, just do right arcs to finish the tree
        decisions.append(result)
        states.append(state)
        if result == "L":
            state = state.left_arc()
        elif result == "R":
            state = state.right_arc()
        else:
            state = state.shift()
    states.append(state)
    return (decisions, states)
