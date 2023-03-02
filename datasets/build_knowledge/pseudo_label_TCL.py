import os
import numpy as np
from tqdm import tqdm
import time
import pickle

from datasets.build_knowledge.helper import *
    

def get_pseudo_label_TCL_for_one_segment(
    args, node2step, step2node, 
    VTM_matched_wikihow_tasks, VTM_matched_howto100m_tasks, 
    wikihow_step_task_occurrence, howto100m_step_task_occurrence):
    
    wikihow_tasknodes_this_segment = dict()
    for task_id in VTM_matched_wikihow_tasks:
        wikihow_stepids = np.where(wikihow_step_task_occurrence[:, task_id] > 0)[0]
        for step_id in wikihow_stepids:
            node_id = step2node[step_id]
            if node_id not in wikihow_tasknodes_this_segment:
                wikihow_tasknodes_this_segment[node_id] = wikihow_step_task_occurrence[step_id, task_id]
            else:
                wikihow_tasknodes_this_segment[node_id] = max(
                    wikihow_tasknodes_this_segment[node_id], 
                    wikihow_step_task_occurrence[step_id, task_id])
    
    howto100m_tasknodes_this_segment = dict()
    for task_id in VTM_matched_howto100m_tasks:
        howto100m_stepids = np.where(howto100m_step_task_occurrence[:, task_id] > 0)[0]
        for step_id in howto100m_stepids:
            node_id = step2node[step_id]
            if node_id not in howto100m_tasknodes_this_segment:
                howto100m_tasknodes_this_segment[node_id] = howto100m_step_task_occurrence[step_id, task_id]
            else:
                howto100m_tasknodes_this_segment[node_id] = max(
                    howto100m_tasknodes_this_segment[node_id], 
                    howto100m_step_task_occurrence[step_id, task_id])
                
                
    howto100m_tasknodes_scores_sorted = sorted(
        howto100m_tasknodes_this_segment.items(), key=lambda item: item[1], reverse=True)
    # [(node_id, node_score), ... , (node_id, node_score)]
    
    results = dict()
    results['wikihow_tasknodes'] = list(wikihow_tasknodes_this_segment.keys())
    matched_tasknodes, matched_tasknodes_scores = find_matching_of_a_segment_given_sorted_val_corres_idx(
                [node_score for (node_id, node_score) in howto100m_tasknodes_scores_sorted],
                [node_id for (node_id, node_score) in howto100m_tasknodes_scores_sorted],
                criteria=args.label_find_tasknodes_criteria, 
                threshold=args.label_find_tasknodes_thresh,
                topK=args.label_find_tasknodes_topK

            )
    results['howto100m_tasknodes'] = {
        'indices': matched_tasknodes, 
        'values': matched_tasknodes_scores
    }
    return results


def get_pseudo_label_TCL(args, logger):
    logger.info('getting TCL pseudo labels...')
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
        'TCL-criteria_{}-threshold_{}-topK_{}.pickle'.format(
            args.label_find_tasknodes_criteria, 
            args.label_find_tasknodes_thresh,
            args.label_find_tasknodes_topK))
    
    with open(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_step_task_occurrence.pickle'), 'rb') as f:
        wikihow_step_task_occurrence = pickle.load(f)
    with open(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_step_task_occurrence.pickle'), 'rb') as f:
        howto100m_step_task_occurrence = pickle.load(f)
        
    # load VTM pseudo label file
    load_time_start = time.time()
    with open(os.path.join(
        sample_pseudo_label_savedir,
        'VTM-criteria_{}-threshold_{}-topK_{}.pickle'.format(
            args.label_find_tasks_criteria, 
            args.label_find_tasks_thresh,
            args.label_find_tasks_topK)), 'rb') as f:
        pseudo_label_VTM = pickle.load(f)
    logger.info('loading VTM pseudo labels for ALL {} samples took {} seconds'.format(
        len(pseudo_label_VTM), round(time.time()-load_time_start, 2)))

    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_TCL = dict()
        for sample_index in tqdm(range(len(samples))):
            VTM_matched_wikihow_tasks = pseudo_label_VTM[
                sample_index]['wikihow_tasks']
            VTM_matched_howto100m_tasks = pseudo_label_VTM[
                sample_index]['howto100m_tasks']['indices'][:args.label_num_howto100m_tasks_to_consider]

            pseudo_label_TCL[sample_index] = get_pseudo_label_TCL_for_one_segment(
                args, node2step, step2node,
                VTM_matched_wikihow_tasks, VTM_matched_howto100m_tasks, 
                wikihow_step_task_occurrence, howto100m_step_task_occurrence)

        # save
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_label_TCL, f)
        logger.info('{} saved!'.format(sample_pseudo_label_savepath))
      
    logger.info('finished getting TCL pseudo labels!')
    # os._exit(0)
    return
