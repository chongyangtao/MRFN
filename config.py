import tensorflow as tf

def config(data_path='ubuntu'): 
	tf.flags.DEFINE_boolean("auto_gpu", False, "Allow detect gpu automatically")
	tf.flags.DEFINE_string('train_record_file', './%s/train.char.tfrecords'%(data_path), 'Path expression to training datafiles. ')
	tf.flags.DEFINE_string('valid_record_file', './%s/valid.char.tfrecords'%(data_path), 'Path expression to valid datafiles. ')
	tf.flags.DEFINE_string('test_record_file', './%s/test.char.tfrecords'%(data_path), 'Path expression to test datafiles. ')
	
	tf.flags.DEFINE_integer("num_threads", 4, "Thread of parsing data")
	tf.flags.DEFINE_integer("capacity", 15000, "Capacity of parsing data")

	tf.flags.DEFINE_integer("embed_dim", 200, "Dimensionality of embedding")
	tf.flags.DEFINE_integer("rnn_dim", 200, "Dimensionality of rnn")

	tf.flags.DEFINE_boolean("init_dict", True, "Allow initialize word2vec")
	tf.flags.DEFINE_string('init_embeddings_path', './%s/word_emb_matrix.pkl'%(data_path), 'Path expression to initial embedding file. ')
	tf.flags.DEFINE_string('word_dict_path', './%s/word_dict.pkl'%(data_path), 'Path expression to word dict. ')

	tf.flags.DEFINE_boolean("init_char_dict", True, "Allow initialize char2vec")


	tf.flags.DEFINE_string('init_char_embeddings_path', './%s/char_emb_matrix.pkl'%(data_path), 'Path expression to initial char embedding file. ')
	tf.flags.DEFINE_string('char_dict_path', './%s/char_dict.pkl'%(data_path), 'Path expression to char dict. ')

	tf.flags.DEFINE_integer("char_vocab_size", 100, "Size of char vocabulary")
	tf.flags.DEFINE_integer("char_embed_dim", 64, "Dimensionality of char embedding")
	tf.flags.DEFINE_integer("char_hid", 100, "Dimensionality of char embedding") # 100

	tf.flags.DEFINE_integer("vocab_size", 5000000, "Size of vocabulary")
	tf.flags.DEFINE_integer("max_turn", 10, "Max length of turn")
	tf.flags.DEFINE_integer("max_utterance_len", 50, "Max length of word") 
	tf.flags.DEFINE_integer("max_word_len", 16, "Max length of characters") 

	tf.flags.DEFINE_boolean("use_char", True, "Allow use char rep")
	tf.flags.DEFINE_boolean("use_word", True, "Allow use word rep")
	tf.flags.DEFINE_boolean("use_seq", True, "Allow use seq rep")
	tf.flags.DEFINE_boolean("use_conv", True, "Allow use conv rep")
	tf.flags.DEFINE_boolean("use_self", True, "Allow use self rep")
	tf.flags.DEFINE_boolean("use_cross", True, "Allow use cross rep")

	tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability") 

	tf.flags.DEFINE_float('lr', 1e-3, 'Initial learning rate')  # 0.15
	tf.flags.DEFINE_boolean("lr_decay", False, "Allow use decay for learning rate")
	tf.flags.DEFINE_float('decay_rate', 0.9, 'decay rate')  
	tf.flags.DEFINE_integer('decay_steps', 2000, 'decay steps')  
	tf.flags.DEFINE_float('lr_minimal', 0.00005, 'minimal learning rate') # 0.00002

	 
	# Training parameters
	tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
	tf.flags.DEFINE_integer("num_epochs", 80000, "Number of training epochs")
	tf.flags.DEFINE_integer("print_every", 50, "Print the results after this many steps")
	tf.flags.DEFINE_integer("valid_every", 1000, "Valid model on dev set after this many steps")
	tf.flags.DEFINE_integer("checkpoint_every", 4000, "Save model after this many step")
	tf.flags.DEFINE_integer("test_every", 1000, "Testing model after this many step")

	tf.flags.DEFINE_boolean("reload_model", False, "Allow reload the model")
	tf.flags.DEFINE_string('log_root', 'logs_tmp/', 'Root directory for all logging.')

	# Misc Parameters
	tf.flags.DEFINE_string('gpu', 'gpu:0', 'Which GPU to use')
	tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
	tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()

	return FLAGS