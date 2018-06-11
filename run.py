import os
import tensorflow as tf
import numpy as np
import properties as p
from six.moves import xrange
from model import creater

np.set_printoptions(threshold=np.nan)

class runner():
    sess = None
    model = None
    is_ready = False
    root = p.get_root()
    working_directory = p.get_working_directory()
    _buckets = p.get_buckets()
    EOS_ID = 2
    
    def init_session(self, user, project):
        repo = os.path.join(self.root, user, project)
        layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor = p.get_training_config(user, project)
        self.sess = tf.Session()
        with tf.variable_scope("chatbot", reuse=tf.AUTO_REUSE):
            self.model = creater.create_model(self.sess, True, repo, self._buckets, self.working_directory
                                            , layer_size=layer_size, num_layers=num_layers
                                            , max_gradient_norm=max_gradient_norm, batch_size=batch_size
                                            , learning_rate=learning_rate, learning_rate_decay_factor=learning_rate_decay_factor)
            self.model.batch_size = 1
        
        self.is_ready = True
    
    def run_session(self, token_ids):
        bucket_id = max(min([b for b in xrange(len(self._buckets)) if self._buckets[b][0] > len(token_ids)]) - 1, 0)
        print("bucket_id : " + str(bucket_id))
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
        _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        percent = []
        for i in range(len(output_logits)):
            percent.append(output_logits[i][0][outputs[i]])
        if self.EOS_ID in outputs:
            outputs = outputs[:outputs.index(self.EOS_ID)]

        return outputs
    
    def get_is_ready(self):
        return self.is_ready
    