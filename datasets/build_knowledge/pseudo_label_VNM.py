import os
import numpy as np
from tqdm import tqdm
import pickle

from datasets.build_knowledge.helper import *


def get_pseudo_label_VNM_for_one_segment(
    args, node2step, step2node, sample_gt_path):
    
    step_scores = np.load(sample_gt_path)
    video_sid = sample_gt_path.split('/')[-2]
    segment_sid = sample_gt_path.split('/')[-1].split('.')[0]

    # obtain node scores
    node_scores = dict()
    for node_id in range(len(node2step)):
        node_scores[node_id] = 0
    for step_id in range(len(step_scores)):
        node_id = step2node[step_id]
        node_scores[node_id] = max(node_scores[node_id], step_scores[step_id])
    
    node_scores_arr = np.array([node_scores[node_id] for node_id in node_scores])
    
    matched_nodes, matched_nodes_scores = find_matching_of_a_segment(
        node_scores_arr, 
        criteria=args.label_find_matched_nodes_criteria, 
        threshold=args.label_find_matched_nodes_for_segments_thresh,
        topK=args.label_find_matched_nodes_for_segments_topK)
    
    pseudo_label_VNM = {'indices': matched_nodes, 
                        'values': matched_nodes_scores
                       }
    return pseudo_label_VNM


def get_pseudo_label_VNM(args, logger):
    logger.info('getting VNM pseudo labels...')
    if not os.path.exists(os.path.join(args.howto100m_dir, 'samples/samples.pickle')):
        from datasets.build_knowledge.get_samples import get_samples
        get_samples(args, logger)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'rb') as f:
        samples = pickle.load(f)
        
    from datasets.build_knowledge.get_nodes import get_nodes
    node2step, step2node = get_nodes(args, logger)
    
    sim_score_path = os.path.join(args.howto100m_dir, 'sim_scores')
    sample_pseudo_label_savedir = os.path.join(args.howto100m_dir, 'pseudo_labels')
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)

    sample_pseudo_label_savepath = os.path.join(
        sample_pseudo_label_savedir, 
        'VNM-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_find_matched_nodes_criteria, 
            args.label_find_matched_nodes_for_segments_thresh,
            args.label_find_matched_nodes_for_segments_topK,
            args.num_nodes))
    
    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_VNM = dict()
        for sample_index in tqdm(range(len(samples))):
            (video_sid, segment_iid) = samples[sample_index]
            segment_sid = 'segment_{}'.format(segment_iid)
            sample_gt_path = os.path.join(sim_score_path, video_sid, segment_sid + '.npy')

            pseudo_label_VNM[sample_index] = get_pseudo_label_VNM_for_one_segment(
                    args, node2step, step2node, sample_gt_path)

        # save
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_label_VNM, f)
        logger.info('{} saved!'.format(sample_pseudo_label_savepath))
    
    logger.info('finished getting VNM pseudo labels!')
    # os._exit(0)
    return 