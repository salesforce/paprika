import os
import numpy as np
import glob
import json
from tqdm import tqdm
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from scipy.sparse import csr_matrix

from datasets.build_knowledge.helper import *


def get_num_neighbors_of_nodes(G):
    """
    - G: an nxn array to represent adj matrx
    """
    num_neighbors = []
    for i in range(len(G)):
        num_neighbors.append(len(np.where(G[i] > 0)[0]))
    return num_neighbors


def threshold_and_normalize(args, logger, G, edge_min_aggconf=1000):
    logger.info('thresholding edges...')
    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > edge_min_aggconf:
                G_new[i, j] = G[i, j]
    G = G_new
    
    G_flat = G.reshape(G.shape[0]*G.shape[0],)
    x = [np.log(val) for val in G_flat if val != 0]
    assert len(x) > 0, 'No edges remain after thresholding! Please use a smaller edge_min_aggconf!'
    max_val, min_val = np.max(x), 0
    
    logger.info('normalizing edges...')
    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > 0:
                G_new[i, j] = (np.log(G[i, j])-0)/(max_val-0)  # log min max norm
    G = G_new    
    return G


def get_edges_between_wikihow_steps_in_wikihow(args, logger):
    with open(os.path.join(args.wikihow_dir, 'step_label_text.json'), 'r') as f:
        wikihow = json.load(f)

    step_id = 0
    article_po_to_step_id = defaultdict()
    for article_id in range(len(wikihow)):
        for article_step_idx in range(len(wikihow[article_id])):
            article_po_to_step_id[(article_id, article_step_idx)] = step_id
            step_id += 1
    total_num_steps = len(article_po_to_step_id)

    wikihow_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
    for article_id in range(len(wikihow)):
        for article_step_idx in range(1, len(wikihow[article_id])):
            predecessor = article_po_to_step_id[(article_id, article_step_idx-1)]
            successor = article_po_to_step_id[(article_id, article_step_idx)]

            wikihow_steps_1hop_edges[predecessor, successor] += 1
    
    return wikihow_steps_1hop_edges


def get_edges_between_wikihow_steps_of_one_howto100m_video(args, video, sim_score_path):
    sim_score_paths_of_segments_this_video = sorted(
        glob.glob(os.path.join(sim_score_path, video, 'segment_*.npy')))
    
    edges_meta = list()
    # loop over segments
    for video_segment_idx in range(1, len(sim_score_paths_of_segments_this_video)): 
        segment_pre_sim_scores = np.load(sim_score_paths_of_segments_this_video[video_segment_idx-1])
        segment_suc_sim_scores = np.load(sim_score_paths_of_segments_this_video[video_segment_idx])
        
        
        predecessors, _ = find_matching_of_a_segment(
            segment_pre_sim_scores, 
            criteria=args.graph_find_matched_steps_criteria, 
            threshold=args.graph_find_matched_steps_for_segments_thresh,
            topK=args.graph_find_matched_steps_for_segments_topK)
        
        successors, _ = find_matching_of_a_segment(
            segment_suc_sim_scores, 
            criteria=args.graph_find_matched_steps_criteria, 
            threshold=args.graph_find_matched_steps_for_segments_thresh,
            topK=args.graph_find_matched_steps_for_segments_topK)
        
        for predecessor in predecessors:
            for successor in successors:
                if predecessor != successor:  # a step transition
                    edges_meta.append(
                        [predecessor, 
                         successor, 
                         segment_pre_sim_scores[predecessor] * segment_suc_sim_scores[successor]]
                    )
    return edges_meta


def get_edges_between_wikihow_steps_in_howto100m(args, logger, total_num_steps):
    sim_score_path = os.path.join(args.howto100m_dir, 'sim_scores')
    
    videos = get_all_video_ids(args, logger)

    logger.info('use multiprocessing to get edges between wikihow step headlines from howto100m...')

    with Pool(processes=args.num_workers) as pool:
        edges_metas = pool.starmap(get_edges_between_wikihow_steps_of_one_howto100m_video, 
                     zip(repeat(args), videos, repeat(sim_score_path)))

    howto100m_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
    for edges_meta in edges_metas:
        for [predecessor, successor, confidence] in edges_meta:
            howto100m_steps_1hop_edges[predecessor, successor] += confidence

    logger.info('multiprocessing finished! going to threshold and normalize edges...')
    howto100m_steps_1hop_edges = threshold_and_normalize(args, logger, howto100m_steps_1hop_edges, args.edge_min_aggconf)

    return howto100m_steps_1hop_edges


def get_node_transition_candidates(args, logger, step2node, G_wikihow, G_howto100m):
    candidates = defaultdict(list)
    
    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_wikihow[step_id].indices:
            conf = G_wikihow[step_id, direct_outstep_id]
            
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            
            candidates[(node_id, direct_outnode_id)].append(conf)

    logger.info('collected node transition candidates (len: {}) from wikiHow...'.format(
        len(candidates)))       
    
    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_howto100m[step_id].indices:
            conf = G_howto100m[step_id, direct_outstep_id]
            
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            
            candidates[(node_id, direct_outnode_id)].append(conf)
            
    logger.info('collected node transition candidates (len: {}) from howTo100M...'.format(
        len(candidates)))       
    
    return candidates


def keep_highest_conf_for_each_candidate(args, logger, candidates):
    edges = defaultdict()
    for (node_id, direct_outnode_id) in tqdm(candidates):
        max_conf = np.max(candidates[(node_id, direct_outnode_id)])
        
        edges[(node_id, direct_outnode_id)] = max_conf
    logger.info('kept only the highest conf score for each node transition candidate... len(edges): {}'.format(
        len(edges)))
    return edges


def build_pkg_adj_matrix(edges, num_nodes):
    pkg = np.zeros((num_nodes, num_nodes))
    for (node_id, direct_outnode_id) in tqdm(edges):
        pkg[node_id, direct_outnode_id] = edges[(node_id, direct_outnode_id)]
    return pkg


def get_edges(args, logger):

    # --  get the edges between step headlines
    logger.info('get edges between wikihow step headlines in wikihow...')
    G_wikihow = get_edges_between_wikihow_steps_in_wikihow(
        args, logger)
    # num_neighbors = get_num_neighbors_of_nodes(G_wikihow)

    logger.info('get edges between wikihow step headlines in howto100m...')
    G_howto100m = get_edges_between_wikihow_steps_in_howto100m(
        args, logger, G_wikihow.shape[0])
    # num_neighbors = get_num_neighbors_of_nodes(G_howto100m)

    G_wikihow_csr, G_howto100m_csr = csr_matrix(G_wikihow), csr_matrix(G_howto100m)

    # -- turn edges between step headlines into edges between nodes
    logger.info('turn edges between step headlines into edges between nodes...')
    from datasets.build_knowledge.get_nodes import get_nodes
    node2step, step2node = get_nodes(args, logger)

    node_transition_candidates = get_node_transition_candidates(
        args, logger, step2node, G_wikihow_csr, G_howto100m_csr)

    pkg_edges = keep_highest_conf_for_each_candidate(
        args, logger, node_transition_candidates)

    pkg = build_pkg_adj_matrix(pkg_edges, len(node2step))
    logger.info('pkg built!')
    
    return pkg, G_wikihow, G_howto100m
