import os
import numpy as np
from tqdm import tqdm
import pickle

from datasets.build_knowledge.helper import *


def get_pseudo_label_DS_for_one_segment(args, sample_gt_path):
    
    step_scores = np.load(sample_gt_path)
    video_sid = sample_gt_path.split('/')[-2]
    segment_sid = sample_gt_path.split('/')[-1].split('.')[0]
    
    matched_steps, matched_steps_scores = find_matching_of_a_segment(
        step_scores, 
        criteria=args.label_find_matched_steps_criteria, 
        threshold=args.label_find_matched_steps_for_segments_thresh,
        topK=args.label_find_matched_steps_for_segments_topK)
    
    pseudo_label_DS = {'indices': matched_steps, 
                       'values': matched_steps_scores
                      }
    return pseudo_label_DS


def get_pseudo_label_DS(args, logger):
    logger.info('getting DS pseudo labels...')
    if not os.path.exists(os.path.join(args.howto100m_dir, 'samples/samples.pickle')):
        from datasets.build_knowledge.get_samples import get_samples
        get_samples(args, logger)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'rb') as f:
        samples = pickle.load(f)
        
    sim_score_path = os.path.join(args.howto100m_dir, 'DS_sim_scores')
    sample_pseudo_label_savedir = os.path.join(args.howto100m_dir, 'DS_pseudo_labels')
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)
    
    sample_pseudo_label_savepath = os.path.join(
        sample_pseudo_label_savedir, 
        'DS-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_find_matched_steps_criteria, 
            args.label_find_matched_steps_for_segments_thresh,
            args.label_find_matched_steps_for_segments_topK,
            args.num_steps))
    
    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_DS = dict()
        for sample_index in tqdm(range(len(samples))):
            (video_sid, segment_iid) = samples[sample_index]
            segment_sid = 'segment_{}'.format(segment_iid)
            sample_gt_path = os.path.join(sim_score_path, video_sid, segment_sid + '.npy')

            
            pseudo_label_DS[sample_index] = get_pseudo_label_DS_for_one_segment(
                args, sample_gt_path)
        
        # save
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_label_DS, f)
        logger.info('{} saved!'.format(sample_pseudo_label_savepath))
      
    logger.info('finished getting DS pseudo labels!')
    # os._exit(0)
    return