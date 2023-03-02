import os
import numpy as np
import time
import pickle
from collections import defaultdict

from datasets.build_knowledge.helper import *


def get_nodes_by_removing_step_duplicates(args, logger, step_des_feats=None):
    if args.remove_step_duplicates:
        start_time = time.time()
        if os.path.exists(
            os.path.join(args.wikihow_dir, 'node2step.pickle')
        ) and os.path.exists(
            os.path.join(args.wikihow_dir, 'step2node.pickle')
        ):
            with open(os.path.join(args.wikihow_dir, 'node2step.pickle'), 'rb') as f:
                node2step= pickle.load(f)
            with open(os.path.join(args.wikihow_dir, 'step2node.pickle'), 'rb') as f:
                step2node = pickle.load(f)  
        else:
            assert step_des_feats is not None
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                linkage=args.step_clustering_linkage, 
                distance_threshold=args.step_clustering_distance_thresh, 
                affinity=args.step_clustering_affinity).fit(step_des_feats)
                # distance_threshold: 
                #   The linkage distance threshold above which, clusters will not be merged. 
            num_nodes = clustering.n_clusters_

            node2step, step2node = defaultdict(), defaultdict()
            for cluster_id in range(num_nodes):
                cluster_members = np.where(clustering.labels_ == cluster_id)[0]
                node2step[cluster_id] = cluster_members
                for step_id in cluster_members:
                    step2node[step_id] = cluster_id
            with open(os.path.join(args.wikihow_dir, 'node2step.pickle'), 'wb') as f:
                pickle.dump(node2step, f)
            with open(os.path.join(args.wikihow_dir, 'step2node.pickle'), 'wb') as f:
                pickle.dump(step2node, f)  

            logger.info("from steps to nodes took {} s".format(round(time.time()-start_time, 2)))

    else:
        node2step = {i: [i] for i in range(args.num_nodes)}
        step2node = {i: i for i in range(args.num_nodes)}
        
    return node2step, step2node


def get_nodes(args, logger):
    
    step_des_feats = get_step_des_feats(args, logger, language_model="MPNet")

    node2step, step2node = get_nodes_by_removing_step_duplicates(args, logger, step_des_feats)
    return node2step, step2node