from threading import Thread
import train
trainer_thread = []

def start_training_thread(user, project, saving_step, train_enc_ids, train_dec_ids):
    trainer = train.trainer()
    thread = Thread(target = trainer.train, args = (user, project, saving_step, train_enc_ids, train_dec_ids))
    thread.start()
    trainer_thread.append({"user" : user, "project" : project, "trainer" : trainer})

def stop_training_thread(user, project):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project:
            trainer = trainer_thread[i]['trainer']
            thread = Thread(target = trainer.stop)
            thread.start()
            trainer_thread.remove(trainer_thread[i])
            break
    
def get_trainer_thread(user, project):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project:
            return trainer_thread[i]['trainer']
    return None

def training_test(user, project, token_ids):
    for i in range(len(trainer_thread)):
        if trainer_thread[i]['user'] == user and trainer_thread[i]['project'] == project:
            trainer = trainer_thread[i]['trainer']
            res = trainer.training_test(token_ids)
            return res
