import os
import numpy as np
from tqdm import tqdm
import time
import pickle
from scipy.sparse import csr_matrix

from datasets.build_knowledge.helper import *


def get_khop_neighbors_inStepIDs(matched_nodes, node2step, max_hop, G_wikihow, G_howto100m):
    neighbors = dict() # store khop neigbors of matched nodes
    for node in matched_nodes:
        neighbors[node] = dict()  # store khop neighbors
        for khop in range(1, max_hop + 1):
            neighbors[node][khop] = dict()
            neighbors[node][khop]['neis_wikihow'] = list()
            neighbors[node][khop]['neis_wikihow_scores'] = list()
            neighbors[node][khop]['neis_howto100m'] = list()
            neighbors[node][khop]['neis_howto100m_scores'] = list()

        steps_this_node = node2step[node]
        for step in steps_this_node:
        
            for khop in range(1, max_hop + 1):
                if khop == 1:
                    # get khop out neighbors of wikihow graph
                    neis_wikihow, neis_wikihow_scores = [], []
                    # for nei in np.where(G_wikihow[step] > 0)[0]:
                    for nei in G_wikihow[step].indices:
                        neis_wikihow.append(nei)
                        neis_wikihow_scores.append(G_wikihow[step, nei])
                    neighbors[node][khop]['neis_wikihow'] += neis_wikihow
                    neighbors[node][khop]['neis_wikihow_scores'] += neis_wikihow_scores
                    
                    # get khop out neighbors of howto100m graph
                    neis_howto100m, neis_howto100m_scores = [], []
                    # for nei in np.where(G_howto100m[step] > 0)[0]:
                    for nei in G_howto100m[step].indices:
                        neis_howto100m.append(nei)
                        neis_howto100m_scores.append(G_howto100m[step, nei])
                    neighbors[node][khop]['neis_howto100m'] += neis_howto100m
                    neighbors[node][khop]['neis_howto100m_scores'] += neis_howto100m_scores
                    
                else:
                    # get khop out neighbors of wikihow graph
                    neis_wikihow, neis_wikihow_scores = [], []
                    for prehop_nei_idx in range(len(neighbors[node][khop - 1]['neis_wikihow'])):
                        prehop_nei = neighbors[node][khop - 1]['neis_wikihow'][prehop_nei_idx]
                        prehop_nei_score = neighbors[node][khop - 1]['neis_wikihow_scores'][prehop_nei_idx]
                        # for nei in np.where(G_wikihow[prehop_nei] > 0)[0]:
                        for nei in G_wikihow[prehop_nei].indices:
                            neis_wikihow.append(nei)
                            neis_wikihow_scores.append(prehop_nei_score * G_wikihow[prehop_nei, nei])
                    neighbors[node][khop]['neis_wikihow'] = neis_wikihow
                    neighbors[node][khop]['neis_wikihow_scores'] = neis_wikihow_scores

                    # get khop out neighbors of howto100m graph
                    neis_howto100m, neis_howto100m_scores = [], []
                    for prehop_nei_idx in range(len(neighbors[node][khop - 1]['neis_howto100m'])):
                        prehop_nei = neighbors[node][khop - 1]['neis_howto100m'][prehop_nei_idx]
                        prehop_nei_score = neighbors[node][khop - 1]['neis_howto100m_scores'][prehop_nei_idx]
                        # for nei in np.where(G_howto100m[prehop_nei] > 0)[0]:
                        for nei in G_howto100m[prehop_nei].indices:
                            neis_howto100m.append(nei)
                            neis_howto100m_scores.append(prehop_nei_score * G_howto100m[prehop_nei, nei])
                    neighbors[node][khop]['neis_howto100m'] = neis_howto100m
                    neighbors[node][khop]['neis_howto100m_scores'] = neis_howto100m_scores
    # keys are node ids, values are step headline ids
    return neighbors



def get_pseudo_label_NRL_for_one_segment_from_scratch(
    args, node2step, step2node, matched_nodes,
    G_wikihow, G_howto100m, G_wikihow_tr, G_howto100m_tr, max_hop):
    
    # get khop in/out neighbors
    khop_out_neighbors = get_khop_neighbors_inStepIDs(
        matched_nodes, node2step, max_hop, G_wikihow, G_howto100m)
    
    khop_in_neighbors = get_khop_neighbors_inStepIDs(
        matched_nodes, node2step, max_hop, G_wikihow_tr, G_howto100m_tr)
    
    # collect NRL pseudo label
    results = dict()
    direction = {
        'out': khop_out_neighbors,
        'in': khop_in_neighbors
    }
    for direction_key in ['out', 'in']: # loop over in & out
        khop_neighbors = direction[direction_key]
        for khop in range(1, max_hop + 1):
            all_neighbors_thishop = []
            all_neighbors_thishop_scores = []
            for node in matched_nodes:
                all_neighbors_thishop += (
                    khop_neighbors[node][khop]['neis_wikihow'] + 
                    khop_neighbors[node][khop]['neis_howto100m']
                )

                all_neighbors_thishop_scores += (
                    khop_neighbors[node][khop]['neis_wikihow_scores'] + 
                    khop_neighbors[node][khop]['neis_howto100m_scores']
                )
            
            # turn step to node and obtain node scores
            node_scores = dict()
            for node_id in [step2node[step_id] for step_id in all_neighbors_thishop]:
                node_scores[node_id] = 0
            for i in range(len(all_neighbors_thishop)):
                step_id = all_neighbors_thishop[i]
                node_id = step2node[step_id]
                node_scores[node_id] = max(node_scores[node_id], all_neighbors_thishop_scores[i])
            
            node_scores_sorted = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
            # [(node_id, node_score), ... , (node_id, node_score)]
            
            matched_neihgbor_nodes, matched_neihgbor_nodes_scores = find_matching_of_a_segment_given_sorted_val_corres_idx(
                [node_score for (node_id, node_score) in node_scores_sorted],
                [node_id for (node_id, node_score) in node_scores_sorted],
                criteria=args.label_find_neighbors_criteria, 
                threshold=args.label_find_neighbors_thresh,
                topK=args.label_find_neighbors_topK
                
            )
            pseudo_label_NRL = {'indices': matched_neihgbor_nodes, 
                               'values': matched_neihgbor_nodes_scores
                              }
            results['{}-hop-{}'.format(khop, direction_key)] = pseudo_label_NRL
    
    return results


def get_pseudo_label_NRL_from_scratch(args, logger):
    logger.info('getting NRL pseudo labels...')
    
    from datasets.build_knowledge.get_nodes import get_nodes
    node2step, step2node = get_nodes(args, logger)
    
    from datasets.build_knowledge.get_edges import get_edges
    _, G_wikihow, G_howto100m = get_edges(args, logger)
    
    
    G_wikihow_tr = np.transpose(G_wikihow)
    G_howto100m_tr = np.transpose(G_howto100m)
    
    G_wikihow, G_howto100m = csr_matrix(G_wikihow), csr_matrix(G_howto100m)
    G_wikihow_tr, G_howto100m_tr = csr_matrix(G_wikihow_tr), csr_matrix(G_howto100m_tr)
    
    sim_score_path = os.path.join(args.howto100m_dir, 'sim_scores')
    sample_pseudo_label_savedir = os.path.join(args.howto100m_dir, 'pseudo_labels')
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)
    
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

    sample_pseudo_label_savepath = os.path.join(
        sample_pseudo_label_savedir, 
        'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
            args.label_khop,
            args.label_find_neighbors_criteria, 
            args.label_find_neighbors_thresh,
            args.label_find_neighbors_topK,
            args.num_nodes))
    if not os.path.exists(sample_pseudo_label_savepath):
        # start processing
        pseudo_label_NRL = dict()
        for sample_index in tqdm(range(len(pseudo_label_VNM))):
            pseudo_label_NRL[sample_index] = get_pseudo_label_NRL_for_one_segment_from_scratch(
                args, node2step, step2node, pseudo_label_VNM[sample_index]['indices'],
                G_wikihow, G_howto100m, G_wikihow_tr, G_howto100m_tr, args.label_khop)
        
        # save
        with open(sample_pseudo_label_savepath, 'wb') as f:
            pickle.dump(pseudo_label_NRL, f)
        logger.info('{} saved!'.format(sample_pseudo_label_savepath))

    logger.info('finished getting NRL pseudo labels!')
    # os._exit(0)
    return


def get_pseudo_label_NRL_for_one_segment(
    args, logger, khop, in_neighbors_previous_hop, out_neighbors_previous_hop, pkg, pkg_tr):
      
    results = dict()
    for direction_key in ['out', 'in']: # loop over in & out
        if direction_key == 'out': # out neighbors
            G, neighbors_previous_hop = pkg, out_neighbors_previous_hop
        else:  # in neighors
            G, neighbors_previous_hop = pkg_tr, in_neighbors_previous_hop
                
        node_scores = dict()
        for i, nei_prehop in enumerate(neighbors_previous_hop['indices']):
            for direct_nei in G[nei_prehop].indices:
                if khop > 1:
                    node_scores[direct_nei] = neighbors_previous_hop['values'][i] * G[nei_prehop, direct_nei]
                else:
                    node_scores[direct_nei] = G[nei_prehop, direct_nei]
        node_scores_sorted = sorted(node_scores.items(), key=lambda item: item[1], reverse=True)
        # [(node_id, node_score), ... , (node_id, node_score)]
        
        (matched_neihgbor_nodes, 
         matched_neihgbor_nodes_scores) = find_matching_of_a_segment_given_sorted_val_corres_idx(
            [node_score for (node_id, node_score) in node_scores_sorted],
            [node_id for (node_id, node_score) in node_scores_sorted],
            criteria=args.label_find_neighbors_criteria, 
            threshold=args.label_find_neighbors_thresh,
            topK=args.label_find_neighbors_topK

        )
        results['{}-hop-{}'.format(khop, direction_key)] = {
            'indices': matched_neihgbor_nodes, 
            'values': matched_neihgbor_nodes_scores
        }
    return results


def get_pseudo_label_NRL(args, logger):
    logger.info('getting NRL pseudo labels...')
    
    from datasets.build_knowledge.get_nodes import get_nodes
    node2step, step2node = get_nodes(args, logger)
    
    from datasets.build_knowledge.get_edges import get_edges
    pkg, _, _ = get_edges(args, logger)
    
    pkg_tr = np.transpose(pkg)
    pkg, pkg_tr = csr_matrix(pkg), csr_matrix(pkg_tr)
    
    sim_score_path = os.path.join(args.howto100m_dir, 'sim_scores')
    sample_pseudo_label_savedir = os.path.join(args.howto100m_dir, 'pseudo_labels')
    os.makedirs(sample_pseudo_label_savedir, exist_ok=True)
    
    for khop in range(1, args.label_khop+1):
        logger.info('Hop {}...'.format(khop))
        sample_pseudo_label_savepath = os.path.join(
            sample_pseudo_label_savedir, 
            'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
                khop,
                args.label_find_neighbors_criteria, 
                args.label_find_neighbors_thresh,
                args.label_find_neighbors_topK,
                args.num_nodes))
        if not os.path.exists(sample_pseudo_label_savepath):
            # start processing
            pseudo_label_NRL = dict()
            if khop == 1:
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
                
                for sample_index in tqdm(range(len(pseudo_label_VNM))):
                    pseudo_label_NRL[sample_index] = get_pseudo_label_NRL_for_one_segment(
                        args, logger, khop, pseudo_label_VNM[sample_index], pseudo_label_VNM[sample_index], pkg, pkg_tr)
                    
            else:
                load_time_start = time.time()
                with open(os.path.join(
                    sample_pseudo_label_savedir, 
                    'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
                        khop-1,
                        args.label_find_neighbors_criteria, 
                        args.label_find_neighbors_thresh,
                        args.label_find_neighbors_topK,
                        args.num_nodes)), 'rb') as f:
                    pseudo_label_NRL_previous_hop = pickle.load(f)
                logger.info('loading NRL pseudo labels PREVIOUS HOP for ALL {} samples took {} seconds'.format(
                    len(pseudo_label_NRL_previous_hop), round(time.time()-load_time_start, 2)))
                
                for sample_index in tqdm(range(len(pseudo_label_NRL_previous_hop))):
                
                    in_neighbors_previous_hop = pseudo_label_NRL_previous_hop[sample_index]['{}-hop-in'.format(khop-1)]
                    out_neighbors_previous_hop = pseudo_label_NRL_previous_hop[sample_index]['{}-hop-out'.format(khop-1)]

                    pseudo_label_NRL[sample_index] = get_pseudo_label_NRL_for_one_segment(
                        args, logger, khop, in_neighbors_previous_hop, out_neighbors_previous_hop, pkg, pkg_tr)
                    pseudo_label_NRL[sample_index].update(pseudo_label_NRL_previous_hop[sample_index])

            # save
            with open(sample_pseudo_label_savepath, 'wb') as f:
                pickle.dump(pseudo_label_NRL, f)
            logger.info('{} saved!'.format(sample_pseudo_label_savepath))

    logger.info('finished getting NRL pseudo labels!')
    # os._exit(0)
    return


