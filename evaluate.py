import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from metrics import recall_2at1, recall_at_k, precision_at_k, MRR, MAP
from data_utils_record import get_record_parser, get_batch_dataset

from nvidia_helper import get_available_gpu
from config import config
from model_FLS import model 


# python evaluate.py --log_root=./logs_fls/ --batch_size=400 

data_path='ubuntu'

if __name__=="__main__":
    FLAGS = config(data_path)
    if FLAGS.auto_gpu:
        # Selecting which GPU to use
        index_of_gpu = get_available_gpu()
        FLAGS.gpu = 'gpu:' + str(index_of_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu.split(':')[1]

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Checkpoint directory. 
    out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.log_root))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints/"))

    with tf.device("/%s" % FLAGS.gpu):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        # session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)


        FLAGS.vocab_size = 439759
        FLAGS.char_vocab_size = 173 # 326  173


        with sess.as_default():

            parser = get_record_parser(FLAGS)
            test_dataset = get_batch_dataset(FLAGS.test_record_file, parser, FLAGS.batch_size, FLAGS.num_threads, FLAGS.capacity, True)
            test_iterator = test_dataset.make_initializable_iterator()
            sess.run(test_iterator.initializer)

            test_handle = sess.run(test_iterator.string_handle())

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, test_dataset.output_types, test_dataset.output_shapes)

            model = model(iterator, FLAGS, FLAGS.embed_dim, FLAGS.vocab_size, FLAGS.char_embed_dim, FLAGS.char_vocab_size, FLAGS.rnn_dim, FLAGS.max_turn, 
                            FLAGS.max_utterance_len, FLAGS.max_word_len)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

            # Restore all variables
            saver = tf.train.Saver()
            print(tf.train.latest_checkpoint(checkpoint_dir))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

            def dev_step():
                acc = []
                losses = []
                pred_scores = []
                ture_scores = []
                count = 0
                while True:
                    try:
                        feed_dict = {
                            handle: test_handle, 
                            model.dropout_keep_prob: 1.0
                        }
                        step, loss, accuracy, y_pred, target = sess.run(
                            [global_step, model.loss, model.accuracy, model.y_pred, model.target], feed_dict)

                        acc.append(accuracy)
                        losses.append(loss)
                        pred_scores += list(y_pred[:, 1])
                        ture_scores += list(target)

                        count +=1
                        if count % 1000 == 0:
                            print("Processing {} samples".format(count))
                    except tf.errors.OutOfRangeError:
                        break

                MeanAcc = sum(acc) / len(acc)
                MeanLoss = sum(losses) / len(losses)
                
                with open(os.path.join(out_dir, 'predScores-iter-%s-test.txt'%(step)), 'w') as f:
                    for score1, score2 in zip(pred_scores, ture_scores):
                        f.writelines(str(score1) + '\t' + str(score2) + '\n')

            
                num_sample = int(len(pred_scores) / 10)
                score_list = np.split(np.array(pred_scores), num_sample, axis=0)
                recall_2_1 = recall_2at1(score_list, k=1)

                recall_at_1 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 1) 
                recall_at_2 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 2)
                recall_at_5 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 5)
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
                print("**********************************")
                print('pred_scores: ', len(pred_scores))
                print("recall_2_1:  %.3f" % (recall_2_1))
                print("recall_at_1: %.3f" % (recall_at_1))
                print("recall_at_2: %.3f" % (recall_at_2))
                print("recall_at_5: %.3f" % (recall_at_5))
                print("**********************************")


            dev_step()







