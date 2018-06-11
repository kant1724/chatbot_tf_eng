root = './user'
working_directory = 'working_dir'
buckets = [(3, 10), (6, 10), (10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (70, 10)]

def get_root():
    return root

def get_working_directory():
    return working_directory

def get_buckets():
    return buckets

def get_training_config(user, project):
    layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor = None, None, None, None, None, None
    with open(root + '/' + user + '/' + project + '/config/training_config') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split("=")[0]
            value = line.split("=")[1].replace("\n", "")
            if key == 'layer_size':
                layer_size = int(value)
            elif key == 'num_layers':
                num_layers = int(value)
            elif key == 'max_gradient_norm':
                max_gradient_norm = float(value)
            elif key == 'batch_size':
                batch_size = int(value)
            elif key == 'learning_rate':
                learning_rate = float(value)
            elif key == 'learning_rate_decay_factor':
                learning_rate_decay_factor = float(value)
    
    return layer_size, num_layers, max_gradient_norm, batch_size, learning_rate, learning_rate_decay_factor
