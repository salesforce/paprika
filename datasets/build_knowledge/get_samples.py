import os
import numpy as np
from tqdm import tqdm
import random
import pickle

from datasets.build_knowledge.helper import *


def get_samples(args, logger):
    logger.info('getting video segment samples...')
    
    # get all video IDs
    videos = get_all_video_ids(args, logger)
    
    samples = list()
    for v in tqdm(videos):
        video_s3d = np.load(os.path.join(args.howto100m_dir, 'feats', v, 'video.npy'))
        
        for c_idx in range(video_s3d.shape[0]):
            samples.append((v, c_idx))
    random.shuffle(samples)
    
    samples_id2name = dict()
    samples_name2id = dict()
    for i in range(len(samples)):
        samples_id2name[i] = samples[i]
        samples_name2id[samples[i]] = i
    
    os.makedirs(os.path.join(args.howto100m_dir, 'samples'), exist_ok=True)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'wb') as f:
        pickle.dump(samples_id2name, f)
    with open(os.path.join(args.howto100m_dir, 'samples/samples_reverse.pickle'), 'wb') as f:
        pickle.dump(samples_name2id, f)
    logger.info('collected video segment samples!')
    return 