import os
import time
import numpy as np
from tqdm import tqdm
try:
    from sentence_transformers import util
except ModuleNotFoundError:
    print('No module named `sentence_transformers`!')
    print('Please perform `pip install -U sentence-transformers`!')
    print('More info: https://www.sbert.net/docs/installation.html')
    os._exit(0)

from datasets.build_knowledge.helper import *


def gatther_all_narration_MPNet_embeddings(args, logger):
    start_time = time.time()
    
    # get all video IDs
    videos = get_all_video_ids(args, logger)
    
    narration_embeddings = []
    narration_lookup_table = []
    
    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            text_mpnet = np.load(os.path.join(args.howto100m_dir, 'feats', v, 'text_mpnet.npy'))
            # text_mpnet shape: (num_clips, num_subclips, 768)

            for c_idx in range(text_mpnet.shape[0]):
                narration_embeddings.append(np.mean(text_mpnet[c_idx], axis=0))
                narration_lookup_table.append((v, c_idx))

        except:
            videos_missing_features.add(v)

    logger.info("number of videos missing narration MPNet features: {}".format(
        len(videos_missing_features)))
    # assert len(videos_missing_features) == 0
    
    narration_embeddings = np.array(narration_embeddings)
    
    logger.info("segment narration embeddings shape: {}".format(narration_embeddings.shape))
    # segment narration embeddings shape: (3741608, 768) for the subset
    # segment narration embeddings shape: (51053844, 768) for the fullset
    logger.info("getting all segment narration embeddings took {} s".format(round(time.time()-start_time, 2)))
    return narration_embeddings, narration_lookup_table


def find_step_similarities_for_segments_using_narration(
    args, logger,
    step_des_feats, segment_narration_embeddings, segment_narration_lookup_table):
    
    start = time.time()
    
    for segment_id in tqdm(range(len(segment_narration_embeddings))):
        v, cidx = segment_narration_lookup_table[segment_id]
        save_path = os.path.join(args.howto100m_dir, 'DS_sim_scores', v, 'segment_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            
            cos_scores = util.cos_sim(
                step_des_feats, 
                segment_narration_embeddings[segment_id])
            
            os.makedirs(os.path.join(args.howto100m_dir, 'DS_sim_scores', v), exist_ok=True)
            try:
                np.save(save_path, cos_scores[:, 0].numpy())
            except OSError:
                continue
        
    logger.info('finding step similarity scores for segments using frames ' + 
                'took {} seconds'.format(time.time() - start))
    # os._exit(0)
    return 


def DS_get_sim_scores(args, logger):
    step_des_feats = get_step_des_feats(args, logger, language_model="MPNet")

    segment_narration_embeddings, segment_narration_lookup_table = \
        gatther_all_narration_MPNet_embeddings(args, logger)

    find_step_similarities_for_segments_using_narration(
        args, logger, 
        step_des_feats, segment_narration_embeddings, segment_narration_lookup_table)
