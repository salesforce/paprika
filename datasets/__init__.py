import os
import pickle


def return_dataset(args, logger, dataset_name, dataset_split='train'):
        
    if 'HowTo100M' in dataset_name:
        from datasets.ht100m import HT100M
        return HT100M(args, logger)
    
    elif dataset_name == 'CrossTask':
        if 'task' in args.downstream_task_name: 
            if (not os.path.exists(
                os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)))) or (
                not os.path.exists(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)))):
            
                from datasets.cross_task import get_task_cls_train_test_splits
                train_split, test_split = get_task_cls_train_test_splits(
                    args.cross_task_video_dir, train_ratio=0.8)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(train_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(test_split, f)
        
        elif 'step' in args.downstream_task_name:
            if (not os.path.exists(
                os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)))) or (
                not os.path.exists(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)))):
            
                from datasets.cross_task import get_step_cls_train_test_splits
                task_meta, steps, train_split, test_split = get_step_cls_train_test_splits(
                    args.cross_task_annot_dir, args.cross_task_s3d_feat_dir, train_ratio=0.8)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(train_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(test_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_task2step_meta.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(task_meta, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_set_of_steps.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(steps, f)
        
        if args.downstream_task_name == 'task_cls':
            from datasets.cross_task import CrossTask_Task_CLS as CrossTask
            
        elif args.downstream_task_name == 'step_cls':
            from datasets.cross_task import CrossTask_Step_CLS as CrossTask
            
        elif args.downstream_task_name == 'step_forecasting':
            from datasets.cross_task import CrossTask_Step_Forecast as CrossTask
        
        return CrossTask(args, logger, split=dataset_split)
    
    elif dataset_name == 'COIN':
        if args.downstream_task_name == 'task_cls':
            from datasets.coin import COIN_Task_CLS as COIN
        elif args.downstream_task_name == 'step_cls':
            from datasets.coin import COIN_Step_CLS as COIN
        elif args.downstream_task_name == 'step_forecasting':
            from datasets.coin import COIN_Step_Forecast as COIN
            
        return COIN(args, logger, split=dataset_split)
    
