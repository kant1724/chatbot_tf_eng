import tensorflow as tf
import re
import os
import time
import numpy as np
import properties as p
from six.moves import xrange
from model import creater

np.set_printoptions(threshold=np.nan)

class trainer():
    _PAD = "_PAD"
    _GO = "_GO"
    _EOS = "_EOS"
    _UNK = "_UNK"
    _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
    PAD_ID = 0
    GO_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
    _DIGIT_RE = re.compile(br"\d")
    root = p.get_root()
    working_directory = p.get_working_directory()
    _buckets = p.get_buckets()
    steps_per_checkpoint = 300
    stop_yn = False
    saving_step = 3000
    '''
        model, session
    '''
    model = None
    model_for_test = None
    sess = None
    
    def read_data(self, train_enc_ids, train_dec_ids):
        data_set = [[] for _ in self._buckets]
        for i in range(len(train_enc_ids)):
            source, target = train_enc_ids[i], train_dec_ids[i]
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            target_ids.append(self.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(self._buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[max(bucket_id - 1, 0)].append([source_ids, target_ids])
                    break
        return data_set
    
    def train(self, user, project, saving_step, train_enc_ids, train_dec_ids):
        saving_step = int(saving_step)
        self.clear_training_message(user, project, saving_step)
        print("Preparing data in %s" % self.working_directory)
        repo = os.path.join(self.root, user, project)        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allocator_type = 'BFC'
        layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor = p.get_training_config(user, project)
        
        # new session
        self.sess = tf.Session() 
        with tf.variable_scope("chatbot", reuse=tf.AUTO_REUSE):
            self.model = creater.create_model(self.sess, False, repo, self._buckets, self.working_directory
                                       , layer_size=layer_size, num_layers=num_layers
                                       , max_gradient_norm=max_gradient_norm, batch_size=batch_size
                                       , learning_rate=learning_rate, learning_rate_decay_factor=learning_rate_decay_factor)
            
            self.model_for_test = creater.create_model(self.sess, True, repo, self._buckets, self.working_directory
                                            , layer_size=layer_size, num_layers=num_layers
                                            , max_gradient_norm=max_gradient_norm, batch_size=batch_size
                                            , learning_rate=learning_rate, learning_rate_decay_factor=learning_rate_decay_factor)
            self.model_for_test.batch_size = 1
            
            train_set = self.read_data(train_enc_ids, train_dec_ids)
            train_bucket_sizes = [len(train_set[b]) for b in xrange(len(self._buckets))]
            train_total_size = float(sum(train_bucket_sizes))
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
            step_time, loss = 0.0, 0.0            
            previous_losses = []
            while True:
                if self.stop_yn:
                    self.stop_yn = False
                    self.sess.close()
                    self.sess = None
                    self.model = None
                    self.model_for_test = None
                    print("Stop Training!")
                    break
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(train_set, bucket_id)
                _, step_loss, _ = self.model.step(self.sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
                step_time = time.time() - start_time
                loss += step_loss / self.steps_per_checkpoint
                self.make_training_message(user, project, self.model.global_step.eval(session=self.sess), loss, step_time, saving_step)
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    self.sess.run(self.model.learning_rate_decay_op)
                    previous_losses.append(loss)
                if self.model.global_step.eval(session=self.sess) % saving_step == 0:
                    self.saving_training_info(user, project)
                    checkpoint_path = os.path.join(repo, self.working_directory, "seq2seq.ckpt")
                    self.model_for_test.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
                step_time, loss = 0.0, 0.0
    
    def training_test(self, token_ids):
        bucket_id = max(min([b for b in xrange(len(self._buckets)) if self._buckets[b][0] > len(token_ids)]) - 1, 0)
        print("bucket_id : " + str(bucket_id))
        encoder_inputs, decoder_inputs, target_weights = self.model_for_test.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
        w, _, output_logits = self.model_for_test.step(self.sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        percent = []
        for i in range(len(output_logits)):
            percent.append(output_logits[i][0][outputs[i]])
        if self.EOS_ID in outputs:
            outputs = outputs[:outputs.index(self.EOS_ID)]
        print(outputs)
        return outputs
    
    def saving_training_info(self, user, project):
        f = open(os.path.join(self.root, user, project, 'working_dir', 'training_info'), 'w', encoding='utf8')
        f.write('Saving model...')
        f.close()
    
    def clear_training_message(self, user, project, saving_step):
        f1 = open(os.path.join(self.root, user, project, 'working_dir', 'training_info'), 'w', encoding='utf8')
        f1.write('')
        f1.close()
        f2 = open(os.path.join(self.root, user, project, 'working_dir', 'saving_step'), 'w', encoding='utf8')
        f2.write(str(saving_step))
        f2.close()
        
    def make_training_message(self, user, project, global_step, loss, step_time, saving_step):
        f1 = open(os.path.join(self.root, user, project, 'working_dir', 'training_info'), 'w', encoding='utf8')
        f1.write('Current step: ' + str(global_step) + ', Cost: ' + str(round(loss, 5) * 100) + ', Period per step: ' + str(round(step_time, 3)))
        f1.close()
        f2 = open(os.path.join(self.root, user, project, 'working_dir', 'saving_step'), 'w', encoding='utf8')
        f2.write(str(saving_step))
        f2.close()
    
    def stop(self):
        self.stop_yn = True
        
    def loop(self):
        while True:
            if self.stop_yn:
                self.stop_yn = False
                print("Stop Loop!")
                break
    
    def reset_tf(self):
        tf.reset_default_graph()
    