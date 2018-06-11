import glob
import os
import properties as p

def delete_ckpt(user, project, root=p.get_root()):
    seq2seq = os.path.join(root, user, project, p.get_working_directory(), 'seq2seq')
    checkpoint = os.path.join(root, user, project, p.get_working_directory(), 'checkpoint')
    
    filelist = glob.glob(seq2seq + '*')
    for file in filelist:
        os.remove(file)
        
    filelist = glob.glob(checkpoint + '*')
    for file in filelist:
        os.remove(file)
