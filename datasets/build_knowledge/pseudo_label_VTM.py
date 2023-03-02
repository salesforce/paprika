import os
import numpy as np
from tqdm import tqdm
import time
import pickle
import json
from collections import defaultdict
import pandas as pd
pd.options.display.max_colwidth = 500

from datasets.build_knowledge.helper import *


def obtain_wikihow_step_task_occurrence(args, logger):
    with open(os.path.join(args.wikihow_dir, 'step_label_text.json'), 'r') as f:
        wikihow = json.load(f)
     
    step_id = 0
    step_id_to_article_po = defaultdict(tuple)
    for article_id in range(len(wikihow)):
        for article_step_idx in range(len(wikihow[article_id])):
            step_id_to_article_po[step_id] = (article_id, article_step_idx)
            step_id += 1
    
    with open(os.path.join(args.wikihow_dir, 'article_id_to_title.txt'), 'r') as f:
        article_id_to_wikhow_taskname = {
            int(line.rstrip().split('\t')[0]): line.rstrip().split('\t')[1] for line in f.readlines()}
    
    wikihow_tasknames = set(article_id_to_wikhow_taskname.values())
    wikihow_taskname_to_taskid = dict()
    wikihow_taskid_to_taskname = dict()
    for task_name in wikihow_tasknames:
        wikihow_taskname_to_taskid[task_name] = len(wikihow_taskname_to_taskid)
        wikihow_taskid_to_taskname[wikihow_taskname_to_taskid[task_name]] = task_name
    
    wikihow_step_task_occurrence = np.zeros((len(step_id_to_article_po), len(wikihow_tasknames)))
    for step_id in range(len(step_id_to_article_po)):
        article_id, _ = step_id_to_article_po[step_id]
        wikihow_step_task_occurrence[step_id, wikihow_taskname_to_taskid[article_id_to_wikhow_taskname[article_id]]] += 1
    return wikihow_step_task_occurrence, wikihow_taskid_to_taskname, wikihow_taskname_to_taskid


def obtain_howto100m_step_task_occurrence(
    args, logger, node2step, step2node, pseudo_label_VNM, samples_reverse):
    
    video_meta_csv = pd.read_csv(args.video_meta_csv_path)
    task_ids_csv = pd.read_csv(args.task_id_to_task_name_csv_path, sep='\t', header=None)
    
    task_id_to_task_name_original_map = dict()
    for index, row in task_ids_csv.iterrows():
        task_id = row[0]
        task_name = row[1]
        task_id_to_task_name_original_map[task_id] = task_name
    
    video_id_to_task_name = dict()
    for index, row in tqdm(video_meta_csv.iterrows()):
        video_id = row['video_id']
        task_id = row['task_id']
        video_id_to_task_name[video_id] = task_id_to_task_name_original_map[task_id]
        
    task_names = set()
    for (video_sid, segment_iid) in tqdm(samples_reverse):
        VNM_matched_nodes = pseudo_label_VNM[
            samples_reverse[(video_sid, segment_iid)]]['indices']
        for node_id in VNM_matched_nodes:
            step_ids_this_node = node2step[node_id]
            for step_id in step_ids_this_node:
                task_name = video_id_to_task_name[video_sid]
                task_names.add(task_name)
    
    assert len(task_names) == len(
        task_names.intersection(set(video_id_to_task_name.values())))
    
    howto100m_taskid_to_taskname = dict()
    howto100m_taskname_to_taskid = dict()
    for task_name in task_names:
        howto100m_taskname_to_taskid[task_name] = len(howto100m_taskname_to_taskid)
        howto100m_taskid_to_taskname[howto100m_taskname_to_taskid[task_name]] = task_name
    
    howto100m_step_task_occurrence = np.zeros((len(step2node), len(task_names)))
    for (video_sid, segment_iid) in tqdm(samples_reverse):
        VNM_matched_nodes = pseudo_label_VNM[
            samples_reverse[(video_sid, segment_iid)]]['indices']
        for node_id in VNM_matched_nodes:
            step_ids_this_node = node2step[node_id]
            for step_id in step_ids_this_node:
                howto100m_step_task_occurrence[
                    step_id, howto100m_taskname_to_taskid[video_id_to_task_name[video_sid]]] += 1       
    return howto100m_step_task_occurrence, howto100m_taskid_to_taskname, howto100m_taskname_to_taskid


def get_pseudo_label_VTM_for_one_segment(
    args, node2step, step2node, 
    VNM_matched_nodes, wikihow_step_task_occurrence, howto100m_step_task_occurrence):
    
    wikihow_tasks_this_segment = dict()
    howto100m_tasks_this_segment = dict()
    for node_id in VNM_matched_nodes:
        step_ids_this_node = node2step[node_id]
        for step_id in step_ids_this_node:
            wikihow_taskids = np.where(wikihow_step_task_occurrence[step_id] > 0)[0]
            for task_id in wikihow_taskids:
                if task_id not in wikihow_tasks_this_segment:
                    wikihow_tasks_this_segment[task_id] = wikihow_step_task_occurrence[step_id, task_id]
                else:
                    wikihow_tasks_this_segment[task_id] = max(
                        wikihow_tasks_this_segment[task_id],
                        wikihow_step_task_occurrence[step_id, task_id])

            howto100m_taskids = np.where(howto100m_step_task_occurrence[step_id] > 0)[0]
            for task_id in howto100m_taskids:
                if task_id not in howto100m_tasks_this_segment:
                    howto100m_tasks_this_segment[task_id] = howto100m_step_task_occurrence[step_id, task_id]
                else:
                    howto100m_tasks_this_segment[task_id] = max(
                        howto100m_tasks_this_segment[task_id], 
                        howto100m_step_task_occurrence[step_id, task_id])
             
    howto100m_task_scores_sorted = sorted(
        howto100m_tasks_this_segment.items(), key=lambda item: item[1], reverse=True)
    # [(task_id, task_score), ... , (task_id, task_score)]
    
    results = dict()
    results['wikihow_tasks'] = list(wikihow_tasks_this_segment.keys())
    matched_tasks, matched_tasks_scores = find_matching_of_a_segment_given_sorted_val_corres_idx(
                [task_score for (task_id, task_score) in howto100m_task_scores_sorted],
                [task_id for (task_id, task_score) in howto100m_task_scores_sorted],
                criteria=args.label_find_tasks_criteria, 
                threshold=args.label_find_tasks_thresh,
                topK=args.label_find_tasks_topK

            )
    results['howto100m_tasks'] = {
        'indices': matched_tasks, 
        'values': matched_tasks_scores
    }
    return results
    
    

def get_pseudo_label_VTM(args, logger):
    logger.info('getting VTM pseudo labels...')
    if not os.path.exists(os.path.join(args.howto100m_dir, 'samples/samples.pickle')):
        from datasets.build_knowledge.get_samples import get_samples
        get_samples(args, logger)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(args.howto100m_dir, 'samples/samples_reverse.pickle'), 'rb') as f:
        samples_reverse = pickle.load(f)
        
    from datasets.build_knowledge.get_nodes import get_nodes
    node2step, step2node = get_nodes(args, logger)
    
    sim_score_path = os.path.join(args.howto100m_dir, 'sim_scores')
    sample_pseudo_label_savedir = os.path.join(args.howto100m_dir, 'pseudo_labels')
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)
    
    sample_pseudo_label_savepath = os.path.join(
        sample_pseudo_label_savedir, 
        'VTM-criteria_{}-threshold_{}-topK_{}.pickle'.format(
            args.label_find_tasks_criteria, 
            args.label_find_tasks_thresh,
            args.label_find_tasks_topK))
    
    # load VNM pseudo label file
    load_time_start = time.time()
    with open(os.path.join(
        sample_pseudo_label_savedir,
        'VNM-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_find_matched_nodes_criteria, 
            args.label_find_matched_nodes_for_segments_thresh,
            args.label_find_matched_nodes_for_segments_topK,
            args.num_nodes)), 'rb') as f:
        pseudo_label_VNM = pickle.load(f)
    logger.info('loading VNM pseudo labels for ALL {} samples took {} seconds'.format(
        len(pseudo_label_VNM), round(time.time()-load_time_start, 2)))
    
    # obtain wikihow_step_task_occurrence 
    if not os.path.exists(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_step_task_occurrence.pickle')):
        (wikihow_step_task_occurrence, 
         wikihow_taskid_to_taskname,
         wikihow_taskname_to_taskid) = obtain_wikihow_step_task_occurrence(args, logger)
        
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_step_task_occurrence.pickle'), 'wb') as f:
            pickle.dump(wikihow_step_task_occurrence, f)
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_taskid_to_taskname.pickle'), 'wb') as f:
            pickle.dump(wikihow_taskid_to_taskname, f)
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_taskname_to_taskid.pickle'), 'wb') as f:
            pickle.dump(wikihow_taskname_to_taskid, f)
    else:
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-wikihow_step_task_occurrence.pickle'), 'rb') as f:
            wikihow_step_task_occurrence = pickle.load(f)
            
    # obtain howto100m_step_task_occurrence
    if not os.path.exists(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_step_task_occurrence.pickle')):
        (howto100m_step_task_occurrence,
         howto100m_taskid_to_taskname, 
         howto100m_taskname_to_taskid) = obtain_howto100m_step_task_occurrence(
            args, logger, node2step, step2node, pseudo_label_VNM, samples_reverse)
    
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_step_task_occurrence.pickle'), 'wb') as f:
            pickle.dump(howto100m_step_task_occurrence, f)
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_taskid_to_taskname.pickle'), 'wb') as f:
            pickle.dump(howto100m_taskid_to_taskname, f)
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_taskname_to_taskid.pickle'), 'wb') as f:
            pickle.dump(howto100m_taskname_to_taskid, f)
    else:
        with open(os.path.join(sample_pseudo_label_savedir, 'VTM-howto100m_step_task_occurrence.pickle'), 'rb') as f:
            howto100m_step_task_occurrence = pickle.load(f)    
        
    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_VTM = dict()
        for sample_index in tqdm(range(len(samples))):
            VNM_matched_nodes = pseudo_label_VNM[sample_index]['indices']
            
            pseudo_label_VTM[sample_index] = get_pseudo_label_VTM_for_one_segment(
                args, node2step, step2node, 
                VNM_matched_nodes, wikihow_step_task_occurrence, howto100m_step_task_occurrence)
            
        # save
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_label_VTM, f)
        logger.info('{} saved!'.format(sample_pseudo_label_savepath))
      
    logger.info('finished getting VTM pseudo labels!')
    # os._exit(0)
    return
