import os
import time
import numpy as np
from tqdm import tqdm
import pickle


def get_step_des_feats(args, logger, language_model='MPNet'):
    """
    language_model: 'MPNet' or 'S3D'
    """
    
    if language_model == 'MPNet':
        step_des_feats = np.load(
            os.path.join(args.wikihow_dir, 'mpnet_feat.npy'))
    
    elif language_model == 'S3D':
        
        with open(
            os.path.join(
                args.wikihow_dir, 
                's3d_text_feat/step_embeddings.pickle'
            ), 'rb') as f:
            
            step_des_feats = pickle.load(f)
        
    return step_des_feats


def get_all_video_ids(args, logger):
    start_time = time.time()
    if os.path.exists(os.path.join(args.howto100m_dir, 'video_IDs.npy')):
        videos = np.load(os.path.join(args.howto100m_dir, 'video_IDs.npy'))
    else:
        videos = []
        for f in tqdm(os.listdir(os.path.join(args.howto100m_dir, 'feats'))):
            if os.path.isdir(os.path.join(args.howto100m_dir, 'feats', f)):
                if os.path.exists(os.path.join(args.howto100m_dir, 'feats', f, 'text_mpnet.npy')): 
                    # raw_captions.pickle  segment_time.npy  status.txt  text_mpnet.npy  text.npy  video.npy
                    videos.append(f)
        logger.info("number of videos: {}".format(len(videos)))
        np.save(os.path.join(args.howto100m_dir, 'video_IDs.npy'), videos)
        
    logger.info("getting all video IDs took {} s".format(round(time.time()-start_time, 2)))
    return videos


def find_matching_of_a_segment(
    sim_scores, criteria="threshold", threshold=0.7, topK=3):
    
    sorted_values = np.sort(sim_scores)[::-1]  # sort in descending order
    sorted_indices = np.argsort(-sim_scores)  # indices of sorting in descending order

    matched_steps, matched_steps_score = find_matching_of_a_segment_given_sorted_val_corres_idx(
        sorted_values, sorted_indices, criteria=criteria, threshold=threshold, topK=topK
    )
    
    return matched_steps, matched_steps_score


def find_matching_of_a_segment_given_sorted_val_corres_idx(
    sorted_values, sorted_indices, criteria="threshold", threshold=0.7, topK=3):
    
    matched_steps = list()
    matched_steps_score = list()
    if criteria == "threshold":
        # Pick all steps with sim-score > threshold.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])
        
    elif criteria == "threshold+topK":
        # From the ones with sim-score > threshold, 
        # pick the top K if existing.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                if len(matched_steps) < topK:
                    matched_steps.append(sorted_indices[i])
                    matched_steps_score.append(sorted_values[i])
                else:
                    break
    
    elif criteria == "topK":
        # Pick the top K
        for i in range(len(sorted_indices)):
            if len(matched_steps) < topK:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])
            else:
                break
                
    else:
        print('The criteria is not implemented!\nFunc: {}\nFile:{}'.format(
            __name__, __file__))
        os._exit(0)
    
    return matched_steps, matched_steps_score


