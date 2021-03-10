import os.path
import pickle
import requests
import torch


FILE_NAME = 'decays.pkl'


def download_data(path: str = 'https://bwsyncandshare.kit.edu/s/xDm7iBdKBZKYoJG/download') -> torch.Tensor:
    """
    Download the data containing particle decays simulated with phasespace
    """
    if not os.path.exists(FILE_NAME):
        req = requests.get(path, allow_redirects=True)
        with open(FILE_NAME, 'wb') as handle:
            handle.write(req.content)
        
    with open(FILE_NAME, 'rb') as handle:
        data = pickle.load(handle)
        
    return torch.tile(data, (10, 2, 1,)).cpu()

