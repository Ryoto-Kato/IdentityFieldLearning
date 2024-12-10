import pickle
import os
import sys

def dump_pckl(data, save_root, pickel_fname):
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    
    pickel_file_path = os.path.join(save_root, pickel_fname)

    if not os.path.exists(pickel_file_path):
        f = open(pickel_file_path, 'x')
        f.close()

    with open(pickel_file_path, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def load_from_memory(path2pkl):
    """
    Returns @dataclass
    """

    # path_to_pickel = os.path.join(save_root, pickle_fname)

    with open(path2pkl, 'rb') as f:
        data = pickle.load(f)
    f.close()
    
    return data

def load_binary_pickle(path2pkl):
    with open(path2pkl, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data
