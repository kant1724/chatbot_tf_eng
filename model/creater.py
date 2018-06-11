import os
from model import seq2seq_model
import tensorflow as tf

enc_vocab_size = 1000
dec_vocab_size = 1000
max_train_data_size = 0

def create_model(session, forward_only, repo, _buckets, working_directory, layer_size=256, num_layers=3, batch_size=64, learning_rate=0.03, learning_rate_decay_factor=0.99, max_gradient_norm=5.0):
    print("Creating %d layers of %d units." % (num_layers, layer_size))
    model = seq2seq_model.Seq2SeqModel(enc_vocab_size, dec_vocab_size, _buckets, layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor, forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(os.path.join(repo, working_directory))
    checkpoint_suffix = ".index"
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    print ("Reading development and training data (limit: %d)." % max_train_data_size)
    
    return model
