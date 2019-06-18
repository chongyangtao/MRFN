import pickle
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from nvidia_helper import get_available_gpu
from metrics import recall_2at1, recall_at_k, precision_at_k, MRR, MAP
from data_utils_record import get_record_parser, get_batch_dataset
from model_FLS import model 
from config import config

data_path='ubuntu'
if __name__=="__main__":
    FLAGS = config(data_path)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    if FLAGS.auto_gpu:
        index_of_gpu = get_available_gpu()
        FLAGS.gpu = 'gpu:' + str(index_of_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu.split(':')[1]
    else:
    	os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.log_root))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. 
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with tf.device("/%s" % FLAGS.gpu):
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        if FLAGS.init_dict:
            # Load pretrained word embeddings
            print("Loading pretrained word embeddings ...")
            with open(FLAGS.init_embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            pretrained_word_embeddings = np.array(embeddings)
            print(pretrained_word_embeddings.shape)
            FLAGS.vocab_size = pretrained_word_embeddings.shape[0]
        else:
            FLAGS.vocab_size = 439759
            pretrained_word_embeddings = None
            print(FLAGS.vocab_size)


        if FLAGS.init_char_dict:
            # Load pretrained char embeddings
            print("Loading pretrained char embeddings ...")
            with open(FLAGS.init_char_embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            pretrained_char_embeddings = np.array(embeddings)
            print(pretrained_char_embeddings.shape)
            FLAGS.char_vocab_size = pretrained_char_embeddings.shape[0]
        else:
            FLAGS.char_vocab_size = 173
            pretrained_char_embeddings = None
            print(FLAGS.char_vocab_size)

        with sess.as_default():
            parser = get_record_parser(FLAGS)
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Create training dataset begain... | %s " % time_str)
            train_dataset = get_batch_dataset(FLAGS.train_record_file, parser, FLAGS.batch_size, FLAGS.num_threads, FLAGS.capacity, False)
            valid_dataset = get_batch_dataset(FLAGS.valid_record_file, parser, FLAGS.batch_size, FLAGS.num_threads, FLAGS.capacity, True)
            test_dataset = get_batch_dataset(FLAGS.test_record_file, parser, FLAGS.batch_size, FLAGS.num_threads, FLAGS.capacity, True)
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("Create training dataset end... | %s " % time_str)

            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

            train_iterator = train_dataset.make_one_shot_iterator()
            valid_iterator = valid_dataset.make_initializable_iterator()
            test_iterator = test_dataset.make_initializable_iterator()

            train_handle = sess.run(train_iterator.string_handle())

            model = model(iterator, FLAGS, FLAGS.embed_dim, FLAGS.vocab_size, FLAGS.char_embed_dim, FLAGS.char_vocab_size, FLAGS.rnn_dim, FLAGS.max_turn, 
                            FLAGS.max_utterance_len, FLAGS.max_word_len, pretrained_word_embeddings, pretrained_char_embeddings)

            # print("Total number of parameters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.placeholder(tf.float32, shape=[])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(model.loss, global_step=global_step)

            # Initialize all variables and word embedding
            saver = tf.train.Saver(max_to_keep=1)
            if not FLAGS.reload_model:
                sess.run(tf.global_variables_initializer())
                if FLAGS.init_dict:
                    sess.run(model.embedding_init)
            else:
                print("Reload model ...")
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

            # for item in tf.trainable_variables():
            #     print(item.name)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("train/loss", model.loss)
            acc_summary = tf.summary.scalar("train/accuracy", model.accuracy)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            # Dev summaries
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir)
            # Dev summaries
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir)


            def train_step():
                """
                A single training step
                """
                train_step = tf.train.global_step(sess, global_step)
                if FLAGS.lr_decay:
                    current_lr = max(FLAGS.lr * np.power(FLAGS.decay_rate, (train_step/FLAGS.decay_steps)), FLAGS.lr_minimal)
                    # current_lr = max(FLAGS.lr * np.power(FLAGS.decay_rate, int(train_step/FLAGS.decay_steps)), FLAGS.lr_minimal)
                else:
                    current_lr = FLAGS.lr

                feed_dict = {
                    learning_rate: current_lr,
                    handle: train_handle, 
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy= sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)


                if step % FLAGS.print_every == 0:
                    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("Step: %d \t| loss: %.3f \t| acc: %.3f \t| lr: %.5f \t| %s" % (step, loss, accuracy, current_lr, time_str))
                    train_summary_writer.add_summary(summaries, step)


            def dev_step(flag, writer):
                if flag=='test':
                    sess.run(test_iterator.initializer)
                    test_handle = sess.run(test_iterator.string_handle())
                else:
                    sess.run(valid_iterator.initializer)
                    valid_handle = sess.run(valid_iterator.string_handle())
                acc = []
                losses = []
                pred_scores = []
                ture_scores = []
                count = 0
                while True:
                    try:
                        feed_dict = {
                            handle: test_handle if flag=='test' else valid_handle, 
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
                            print(count)
                    except tf.errors.OutOfRangeError:
                        break

                MeanAcc = sum(acc) / len(acc)
                MeanLoss = sum(losses) / len(losses)
                
                with open(os.path.join(out_dir, 'predScores-iter-%s.txt'%(step)), 'w') as f:
                    for score1, score2 in zip(pred_scores, ture_scores):
                        f.writelines(str(score1) + '\t' + str(score2) + '\n')

                summary_MeanLoss = tf.Summary(value=[tf.Summary.Value(tag='%s/MeanLoss'%(flag), simple_value=MeanLoss)])
                summary_MeanAcc = tf.Summary(value=[tf.Summary.Value(tag='%s/MeanAcc'%(flag), simple_value=MeanAcc)])
                writer.add_summary(summary_MeanLoss, step)
                writer.add_summary(summary_MeanAcc, step)

                num_sample = int(len(pred_scores) / 10)
                score_list = np.split(np.array(pred_scores), num_sample, axis=0)
                recall_2_1 = recall_2at1(score_list, k=1)

                recall_at_1 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 1) 
                recall_at_2 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 2)
                recall_at_5 = recall_at_k(np.array(ture_scores),  np.array(pred_scores), 5)
                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("**********************************")
                print("%s results.........."%(flag.title()))
                print('pred_scores: ', len(pred_scores))
                print("Step: %d \t| loss: %.3f \t| acc: %.3f \t|  %s" %(step, MeanLoss, MeanAcc, time_str))
                print("recall_2_1:  %.3f" % (recall_2_1))
                print("recall_at_1: %.3f" % (recall_at_1))
                print("recall_at_2: %.3f" % (recall_at_2))
                print("recall_at_5: %.3f" % (recall_at_5))
                print("**********************************")

                summary_recall_2_1 = tf.Summary(value=[tf.Summary.Value(tag='%s/recall_2_1'%(flag), simple_value=recall_2_1)])
                summary_recall_at_1 = tf.Summary(value=[tf.Summary.Value(tag='%s/recall_at_1'%(flag), simple_value=recall_at_1)])
                summary_recall_at_2 = tf.Summary(value=[tf.Summary.Value(tag='%s/recall_at_2'%(flag), simple_value=recall_at_2)])
                summary_recall_at_5 = tf.Summary(value=[tf.Summary.Value(tag='%s/recall_at_5'%(flag), simple_value=recall_at_5)])                    

                writer.add_summary(summary_recall_2_1, step)
                writer.add_summary(summary_recall_at_1, step)
                writer.add_summary(summary_recall_at_2, step)
                writer.add_summary(summary_recall_at_5, step)
                return MeanLoss, recall_2_1+recall_at_1


            optimal_metrics = 0.0
            optimal_step = 0
            
            for i in range(FLAGS.num_epochs):
                train_step()
                current_step = tf.train.global_step(sess, global_step)
                
                if current_step % FLAGS.valid_every == 0:
                    meanLoss, metrics = dev_step('dev', dev_summary_writer)
                    if metrics>optimal_metrics:
                        optimal_metrics = metrics
                        optimal_step = current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                    print("opt_step: %d \t| opt_metric: %.3f" %(optimal_step, optimal_metrics))

                sys.stdout.flush()




