from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS
import run
import os
import properties as p
import training_thread as wt
import file

ip_addr = open('./ip_addr', encoding="utf8").readlines()[0].replace('\n', '')

app = Flask(__name__, static_url_path="/static") 
CORS(app)

trainer = None
runner = None

@app.route("/update_training_config", methods=['POST'])
def update_training_config():
    user = request.form['user']
    project = request.form['project']
    layer_size = request.form['LAYER_SIZE']
    num_layers = request.form['NUM_LAYERS']
    batch_size = request.form['BATCH_SIZE']
    learning_rate = request.form['LEARNING_RATE']
    learning_rate_decay_factor = request.form['LEARNING_RATE_DECAY_FACTOR']
    max_gradient_norm = request.form['MAX_GRADIENT_NORM']
    with open('./user/' + user + '/' + project + '/config/training_config', 'w', encoding='utf8') as f:
        f.write("layer_size=" + layer_size + "\n")
        f.write("num_layers=" + num_layers + "\n")
        f.write("batch_size=" + batch_size + "\n")
        f.write("learning_rate=" + learning_rate + "\n")
        f.write("learning_rate_decay_factor=" + learning_rate_decay_factor + "\n")
        f.write("max_gradient_norm=" + max_gradient_norm + "\n")

    return ''

@app.route("/init_chatbot", methods=['POST'])
def init_chatbot():
    user = request.form['user']
    project = request.form['project']
    global runner
    if runner == None:
        runner = run.runner()
        runner.init_session(user, project)
    return jsonify({'ok' : 'ok'})
    
@app.route("/run_chatbot", methods=['POST'])
def run_chatbot():
    token_ids = eval(request.form['token_ids'])
    reply = runner.run_session(token_ids)
    return jsonify({'reply' : str(reply)})

@app.route("/is_chatbot_ready", methods=['POST'])
def is_chatbot_ready():
    is_ready = runner.get_is_ready()
    if is_ready == True:
        is_ready = 'Y'
    else:
        is_ready = 'N'
        
    return jsonify({'is_ready' : is_ready})

@app.route("/start_training", methods=['POST'])
def start_training():
    user = request.form['user']
    project = request.form['project']
    saving_step = request.form['saving_step']
    train_enc_ids = eval(request.form['train_enc_ids'])
    train_dec_ids = eval(request.form['train_dec_ids'])
    global runner
    runner = None
    wt.start_training_thread(user, project, saving_step, train_enc_ids, train_dec_ids)    
    return ''

@app.route("/stop_training", methods=['POST'])
def stop_training(): 
    user = request.form['user']
    project = request.form['project']
    wt.stop_training_thread(user, project)
    return ''

@app.route("/get_training_info", methods=['POST'])
def get_training_info():
    root = p.get_root()
    user = request.form['user']
    project = request.form['project']
    f1 = open(os.path.join(root, user, project, 'working_dir', 'training_info'), 'r', encoding='utf8')
    f2 = open(os.path.join(root, user, project, 'working_dir', 'saving_step'), 'r', encoding='utf8')
    training_info = f1.readline()
    saving_step = f2.readline()
    return jsonify({'training_info' : training_info, 'saving_step' : saving_step})

@app.route("/is_training", methods=['POST'])
def is_training(): 
    user = request.form['user']
    project = request.form['project']
    res = wt.get_trainer_thread(user, project)
    is_training = 'N'
    if res != None:
        is_training = 'Y'
    return jsonify({'is_training' : is_training})

@app.route("/is_running", methods=['POST'])
def is_running(): 
    is_running = 'N'
    if runner != None:
        is_running = 'Y'
    return jsonify({'is_running' : is_running})
    
@app.route("/delete_ckpt", methods=['POST'])
def delete_ckpt():
    user = request.form['user']
    project = request.form['project']
    file.delete_ckpt(user, project, p.get_root())
    return ''

@app.route("/training_test", methods=['POST'])
def training_test():
    user = request.form['user']
    project = request.form['project']
    token_ids = eval(request.form['token_ids'])
    res = wt.training_test(user, project, token_ids)
    
    return jsonify({'reply' : str(res)})

if (__name__ == "__main__"): 
    app.run(threaded=True, host=ip_addr, port = 5003)
    