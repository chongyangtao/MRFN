import os
import pickle
from collections import defaultdict
import logging
import gensim
import time
import numpy as np
from random import shuffle
import codecs
import concurrent.futures
from datetime import datetime
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def load_file(input_file, word2idx_file, char2idx_file, isshuffle = True):
    word2idx = pickle.load(open(word2idx_file, 'rb'))
    char2idx = pickle.load(open(char2idx_file, 'rb'))
    
    revs = []
    response_set = []
    with codecs.open(input_file, 'r', 'utf-8') as f: 
        for k, line in enumerate(f):
            parts = line.strip().split("\t")
            label = parts[0]
            context = parts[1:-1] # multi-turn
            #context = " ".join(parts[1:-1])  # single-turn
            response = parts[-1]

            data = {"y": label, "c": context, "r": response}
            #print(context, response, label)
            revs.append(data)
            response_set.append(response)
    print("processed dataset with %d context-response pairs " % (len(revs)))
    if isshuffle == True:
        shuffle(revs)

    return revs, response_set, word2idx, char2idx


def get_char_word_idx_from_sent(sent, word_idx_map, char_idx_map, max_word_len=50, max_char_len=16):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    token_ids = [word_idx_map.get(word.encode("utf-8"), 0) for word in sent.split()]
    x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
    x_mask = pad_sequences([len(token_ids)*[1]], padding='post', maxlen=max_word_len)[0]
    x_len = min(len(token_ids), max_word_len)

    x_char = np.zeros([max_word_len, max_char_len], dtype=np.int32) 
    x_char_mask = np.zeros([max_word_len, max_char_len], dtype=np.int32) 
    x_char_len = np.zeros([max_word_len], dtype=np.int32) 

    # get char index
    for i, word in enumerate(sent.split()):
        if i >= max_word_len: continue
        char_ids = [char_idx_map.get(c.encode("utf-8"), 0) for c in word]
        x_char[i] = pad_sequences([char_ids], padding='post', maxlen=max_char_len)[0]
        x_char_mask[i] = pad_sequences([len(char_ids)*[1]], padding='post', maxlen=max_char_len)[0]
        x_char_len[i] = len(char_ids)


    return x, x_mask, x_len, x_char, x_char_mask, x_char_len

def get_char_word_idx_from_sent_msg(sents, word_idx_map, char_idx_map, max_turn=10, max_word_len=50, max_char_len=16):
    word_turns = []
    word_masks = []
    word_lens = []

    char_turns = []
    char_masks = []
    char_lens = []

    for sent in sents:
        words = sent.split()
        token_ids = [word_idx_map.get(word.encode("utf-8"), 0) for word in words]
        x = pad_sequences([token_ids], padding='post', maxlen=max_word_len)[0]
        x_mask = pad_sequences([len(token_ids)*[1]], padding='post', maxlen=max_word_len)[0]
        
        word_turns.append(x)
        word_masks.append(x_mask)        
        word_lens.append(min(len(words), max_word_len))

        x_char = np.zeros([max_word_len, max_char_len], dtype=np.int32) 
        x_char_mask = np.zeros([max_word_len, max_char_len], dtype=np.int32) 
        x_char_len = np.zeros([max_word_len], dtype=np.int32) 
 
        for i, word in enumerate(words):
            if i >= max_word_len: continue
            char_ids = [char_idx_map.get(c.encode("utf-8"), 0) for c in word]
            x_char[i] = pad_sequences([char_ids], padding='post', maxlen=max_char_len)[0]
            x_char_mask[i] = pad_sequences([len(char_ids)*[1]], padding='post', maxlen=max_char_len)[0]
            x_char_len[i] = len(char_ids)

        char_turns.append(x_char)
        char_masks.append(x_char_mask)
        char_lens.append(x_char_len)

    word_turns_new = np.zeros([max_turn, max_word_len], dtype=np.int32) 
    word_masks_new = np.zeros([max_turn, max_word_len], dtype=np.int32) 
    word_lens_new = np.zeros([max_turn], dtype=np.int32) 

    char_turns_new = np.zeros([max_turn, max_word_len, max_char_len], dtype=np.int32) 
    char_masks_new = np.zeros([max_turn, max_word_len, max_char_len], dtype=np.int32) 
    char_lens_new = np.zeros([max_turn, max_word_len], dtype=np.int32) 

    if len(word_turns) <= max_turn:
        word_turns_new[-len(word_turns):]= word_turns
        word_masks_new[-len(word_turns):] = word_masks
        word_lens_new[-len(word_turns):] = word_lens
       
        char_turns_new[-len(word_turns):]= char_turns
        char_masks_new[-len(word_turns):] = char_masks
        char_lens_new[-len(word_turns):] = char_lens
      
        
    if len(word_turns) > max_turn:
        word_turns_new[:] = word_turns[len(word_turns)-max_turn:len(word_turns)]
        word_masks_new[:] = word_masks[len(word_turns)-max_turn:len(word_turns)]
        word_lens_new[:] = word_lens[len(word_turns)-max_turn:len(word_turns)]
        
        char_turns_new[:] = char_turns[len(word_turns)-max_turn:len(word_turns)]
        char_masks_new[:] = char_masks[len(word_turns) - max_turn:len(word_turns)]
        char_lens_new[:] = char_lens[len(word_turns) - max_turn:len(word_turns)]

    # print("sents: ", sents)
    # print("word_turns_new: ", word_turns_new)
    # print("word_masks_new: ", word_masks_new)
    # print("word_lens_new: ", word_lens_new)
    # print("char_turns_new: ", char_turns_new)
    # print("char_masks_new: ", char_masks_new)
    # print("char_lens_new: ", char_lens_new)
    # print("\n")
    # time.sleep(5)
    
    return word_turns_new, word_masks_new, word_lens_new, char_turns_new, char_masks_new, char_lens_new




def build_records(data_file, word2idx_file, char2idx_file, records_name, max_turn=10, max_utterance_len=50, max_word_len=16, isshuffle=False, max_mum=100000000):
    revs, response_set, word2idx, char2idx= load_file(data_file, word2idx_file, char2idx_file, isshuffle)
    print("load data done ...")
    writer = tf.python_io.TFRecordWriter(records_name)
    for k, rev in enumerate(revs):
        context, content_mask, context_len, char_context, char_content_mask, char_context_len = \
                    get_char_word_idx_from_sent_msg(rev["c"], word2idx, char2idx, max_turn, max_utterance_len, max_word_len)
        response, response_mask, response_len, char_response, char_response_mask, char_response_len = \
                    get_char_word_idx_from_sent(rev['r'], word2idx, char2idx, max_utterance_len, max_word_len)
        y_label = int(rev["y"]) 
        features = {
            'context': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context.tostring()])),
            'content_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[content_mask.tostring()])),
            'context_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_len.tostring()])),
            'response': tf.train.Feature(bytes_list=tf.train.BytesList(value=[response.tostring()])),
            'response_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[response_mask.tostring()])),
            'response_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[response_len])),

            'char_context': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_context.tostring()])),
            'char_content_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_content_mask.tostring()])),
            'char_context_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_context_len.tostring()])),
            'char_response': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_response.tostring()])),
            'char_response_mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_response_mask.tostring()])),
            'char_response_len': tf.train.Feature(bytes_list=tf.train.BytesList(value=[char_response_len.tostring()])),
            
            'y_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_label]))      
        }

        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writer.write(tf_serialized)
        if((k+1)%10000==0):
            print('Write {} examples to {}'.format(k+1, records_name))
        if (k+1)>=max_mum:
            break
    writer.close()


def get_record_parser(FLAGS):
    def _parser(example_proto):
        dics = {
            'context': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'content_mask': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'context_len': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response_mask': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'response_len': tf.FixedLenFeature(shape=[], dtype=tf.int64),

            'char_context': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'char_content_mask': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'char_context_len': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'char_response': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'char_response_mask': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'char_response_len': tf.FixedLenFeature(shape=[], dtype=tf.string),

            'y_label': tf.FixedLenFeature(shape=[], dtype=tf.int64)
        }

        parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)

        context = tf.reshape(tf.decode_raw(parsed_example["context"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len])
        content_mask = tf.reshape(tf.decode_raw(parsed_example["content_mask"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len])
        context_len = tf.reshape(tf.decode_raw(parsed_example["context_len"], tf.int32), [FLAGS.max_turn])
        response = tf.reshape(tf.decode_raw(parsed_example["response"], tf.int32), [FLAGS.max_utterance_len])
        response_mask = tf.reshape(tf.decode_raw(parsed_example["response_mask"], tf.int32), [FLAGS.max_utterance_len])
        response_len = parsed_example["response_len"]

        char_context = tf.reshape(tf.decode_raw(parsed_example["char_context"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len, FLAGS.max_word_len])
        char_content_mask = tf.reshape(tf.decode_raw(parsed_example["char_content_mask"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len, FLAGS.max_word_len])
        char_context_len = tf.reshape(tf.decode_raw(parsed_example["char_context_len"], tf.int32), [FLAGS.max_turn, FLAGS.max_utterance_len])
        char_response = tf.reshape(tf.decode_raw(parsed_example["char_response"], tf.int32), [FLAGS.max_utterance_len, FLAGS.max_word_len])
        char_response_mask = tf.reshape(tf.decode_raw(parsed_example["char_response_mask"], tf.int32), [FLAGS.max_utterance_len, FLAGS.max_word_len])
        char_response_len = tf.reshape(tf.decode_raw(parsed_example["char_response_len"], tf.int32), [FLAGS.max_utterance_len])

        y_label = parsed_example["y_label"]

        return context, content_mask, context_len, response, response_mask, response_len, \
                char_context, char_content_mask, char_context_len, char_response, char_response_mask, char_response_len, y_label
    return _parser


def get_batch_dataset(record_file, parser, batch_size, num_threads, capacity, is_test=False):
    num_threads = tf.constant(num_threads, dtype=tf.int32)
    
    if is_test:
        dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).repeat(1).batch(batch_size) 
    else:
        dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).shuffle(capacity).repeat().batch(batch_size)
    return dataset

def process_word_embeddings(embedding_file, total_words, word_embedding_size,  outfile):
    word_dict = dict()
    vectors = [list(np.zeros(word_embedding_size))]
    with open(embedding_file,'r') as f:
        lines = f.readlines()  # there exits an useless line in word2vec 
        for i, line in enumerate(lines):
            line = line.strip().split(' ') 
            word_dict[line[0]] = i + 1
            vectors.append(list(map(float, line[1:])))
            if i > total_words:
                break
        
    with open(os.path.join(outfile, 'char_emb_matrix.pkl'), 'wb') as f:
        pickle.dump(vectors, f) # 
    with open(os.path.join(outfile, 'char_dict.pkl'), 'wb') as f:
        pickle.dump(word_dict, f)



if __name__ == "__main__":
    if 0:
        build_records('ubuntu/train.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/train.char.small.tfrecords', isshuffle=True, max_mum=20000)
        build_records('ubuntu/valid.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/valid.char.small.tfrecords', max_mum=10000)
        build_records('ubuntu/test.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/test.char.small.tfrecords', max_mum=10000)
    else:
        build_records('ubuntu/train.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/train.char.tfrecords', isshuffle=True)
        build_records('ubuntu/valid.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/valid.char.tfrecords')
        build_records('ubuntu/test.txt', 'ubuntu/word_dict.pkl', 'ubuntu/char_dict.pkl', 'ubuntu/test.char.tfrecords')


