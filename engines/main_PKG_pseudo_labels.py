import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import warnings
warnings.filterwarnings("ignore")

import platform

from args.args_pseudo_labeling import get_args_parser
from datasets.build_knowledge.build_knowledge import obtain_external_knowledge
from utils.common_utils import set_seed, getLogger


if __name__ == '__main__':
    
    args = get_args_parser()
    set_seed(args.seed)
    logfile_path = os.path.abspath(
        os.path.join(args.log_dir, 'PKG_pseudo_labeling.log'))
    logger = getLogger(name=__name__, path=logfile_path)

    logger.info("Working config: {}\n".format(args))
    logger.info("Host: {}".format(platform.node()))
    logger.info("Logfile path: {}".format(logfile_path))
    
    obtain_external_knowledge(args, logger)
    