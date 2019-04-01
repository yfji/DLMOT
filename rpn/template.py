import numpy as np

'''
def get_template():
    return np.array([
        [64,64],
        [128,128],
        [256,256],
        [384,384]], dtype=np.float32)
'''
'''
def get_template():
    return np.array([
        [64,64],
        [112,112],
        [160,160],
        [208,208],
        [256,256]], dtype=np.float32)
'''
def get_template(min_size=64, max_size=448, num_templates=5):
    size=np.linspace(min_size, max_size, num_templates)
    size=np.round(size).astype(np.int32).reshape(-1,1)
    templates=np.hstack((size, size))
    return templates