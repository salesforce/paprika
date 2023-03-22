import os
import pickle
import random
from tqdm import tqdm


def partition_dataset(args, logger):
    """ Randomly partition the dataset's features and pseudo labels. """
    
    with open(os.path.join(args.howto100m_dir, 'feats', 
                           'feats_all-mean_agg.pickle'), 'rb') as f:
        feats_all = pickle.load(f)
        
    all_indices = list(feats_all.keys())
    num_each_partition = len(all_indices)//args.num_partitions 
    num_allsamples = num_each_partition * args.num_partitions
        
    random.shuffle(all_indices)
    all_indices = all_indices[:num_allsamples]
    
    # VNM
    sample_pseudo_label_savepath = os.path.join(
        args.howto100m_dir, 'pseudo_labels', 
        'VNM-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_find_matched_nodes_criteria, 
            args.label_find_matched_nodes_for_segments_thresh,
            args.label_find_matched_nodes_for_segments_topK,
            args.num_nodes))
    with open(sample_pseudo_label_savepath, 'rb') as f:
        pseudo_labels_all_VNM = pickle.load(f)
        
    # VTM
    sample_pseudo_label_savepath = os.path.join(
        args.howto100m_dir, 'pseudo_labels', 
        'VTM-criteria_{}-threshold_{}-topK_{}.pickle'.format(
            args.label_find_tasks_criteria, 
            args.label_find_tasks_thresh,
            args.label_find_tasks_topK))
    with open(sample_pseudo_label_savepath, 'rb') as f:
        pseudo_labels_all_VTM = pickle.load(f)
    
    # TCL
    sample_pseudo_label_savepath = os.path.join(
        args.howto100m_dir, 'pseudo_labels', 
        'TCL-criteria_{}-threshold_{}-topK_{}.pickle'.format(
            args.label_find_tasks_criteria, 
            args.label_find_tasks_thresh,
            args.label_find_tasks_topK))
    with open(sample_pseudo_label_savepath, 'rb') as f:
        pseudo_labels_all_TCL = pickle.load(f)
        
    # NRL
    sample_pseudo_label_savepath = os.path.join(
        args.howto100m_dir, 'pseudo_labels', 
        'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_khop,
            args.label_find_neighbors_criteria, 
            args.label_find_neighbors_thresh,
            args.label_find_neighbors_topK,
            args.num_nodes))
    with open(sample_pseudo_label_savepath, 'rb') as f
        pseudo_labels_all_NRL = pickle.load(f)
    
    # Start partitioning
    for i in range(args.num_partitions):
        index_keys_this_partition = all_indices[i*num_each_partition:(i+1)*num_each_partition]
        
        feats_this = dict()
        pseudo_labels_VNM_this = dict()
        pseudo_labels_VTM_this = dict()
        pseudo_labels_TCL_this = dict()
        pseudo_labels_NRL_this = dict()
        
        for key in tqdm(index_keys_this_partition):
            feats_this[key] = feats_all[key]
            pseudo_labels_VNM_this[key] = pseudo_labels_all_VNM[key]
            pseudo_labels_VTM_this[key] = pseudo_labels_all_VTM[key]
            pseudo_labels_TCL_this[key] = pseudo_labels_all_TCL[key]
            pseudo_labels_NRL_this[key] = pseudo_labels_all_NRL[key]
            
            
        with open(os.path.join(
            args.howto100m_dir,
            'index_keys_this_partition-rank_{}-of-{}.pickle'.format(
                i, args.num_partitions)), 'wb') as f:
            pickle.dump(index_keys_this_partition, f)
            
        with open(os.path.join(
                args.howto100m_dir, 'feats', 
                'feats_all-mean_agg-rank_{}-of-{}.pickle'.format(
                    i, args.num_partitions)), 'wb') as f:
            pickle.dump(feats_this, f)
            
            
        sample_pseudo_label_savepath = os.path.join(
            args.howto100m_dir, 'pseudo_labels', 
            'VNM-criteria_{}-threshold_{}-topK_{}-size_{}-rank_{}-of-{}.pickle'.format(
                args.label_find_matched_nodes_criteria, 
                args.label_find_matched_nodes_for_segments_thresh,
                args.label_find_matched_nodes_for_segments_topK,
                args.num_nodes,
                i, 
                args.num_partitions))
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_labels_VNM_this, f)
            
        
        sample_pseudo_label_savepath = os.path.join(
            args.howto100m_dir, 'pseudo_labels', 
            'VTM-criteria_{}-threshold_{}-topK_{}-rank_{}-of-{}.pickle'.format(
                args.label_find_tasks_criteria, 
                args.label_find_tasks_thresh,
                args.label_find_tasks_topK,
                i, 
                args.num_partitions))
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_labels_VTM_this, f)
            
            
        sample_pseudo_label_savepath = os.path.join(
            args.howto100m_dir, 'pseudo_labels',
                'TCL-criteria_{}-threshold_{}-topK_{}-rank_{}-of-{}.pickle'.format(
                    args.label_find_tasks_criteria, 
                    args.label_find_tasks_thresh,
                    args.label_find_tasks_topK,
                    i, 
                    args.num_partitions))
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_labels_TCL_this, f)
            
            
        sample_pseudo_label_savepath = os.path.join(
            args.howto100m_dir, 'pseudo_labels', 
            'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}-rank_{}-of-{}.pickle'.format(
                args.label_khop,
                args.label_find_neighbors_criteria, 
                args.label_find_neighbors_thresh,
                args.label_find_neighbors_topK,
                args.num_nodes,
                i, 
                args.num_partitions))
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_labels_NRL_this, f)
        
        
            
    
    
        