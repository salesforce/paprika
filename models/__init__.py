from utils.common_utils import need_logging


def create_model(args, logger, model_name):
    
    if model_name == 'adapter_mlp':
        if hasattr(args, "adapter_inference") and args.adapter_inference:
            from models.adapter_inference import Adapter
        else:
            from models.adapter import Adapter
        model = Adapter(args, logger)
        
    elif 'downstream' in model_name:
        if args.downstream_task_name == 'task_cls':
            from models.task_head_task_cls import Task_Head
        elif args.downstream_task_name == 'step_cls':
            from models.task_head_step_cls import Task_Head
        elif args.downstream_task_name == 'step_forecasting':
            from models.task_head_step_forecasting import Task_Head
        model = Task_Head(args, logger)
        
    else:
        raise ValueError("Model {} not recognized.".format(args.adapter_name))


    if need_logging(args):
        logger.info(model)
        logger.info("--> model {} was created".format(model_name))

    return model

