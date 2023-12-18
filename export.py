"""MindYolo Export Script. Transform MindSpore weight format"""

import argparse
import ast
import os
import sys
import numpy as np
import mindspore as ms

from mindspore import Tensor, context, export
import mindspore.context as context

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.utils import set_seed

def get_parser_export(parents=None):
    parser = argparse.ArgumentParser(description="Export", parents=[parents] if parents else [])
    parser.add_argument("--device_target", type=str, default="CPU", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--file_name", type=str, default="yolov8-l.onnx", help="the name of output file")
    parser.add_argument("--weight", type=str, default="./yolov8-l_500e_mAP528-6e96d6bb.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--file_format", type=str, default="ONNX", help="treat as single-class dataset")
    parser.add_argument("--save_dir", type=str, default="./export", help="save dir")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    return parser



def set_default_export(args):
    # Set Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target, max_call_depth=2000)
    if args.device_target == "Ascend":
        context.set_context(device_id=int(os.getenv("DEVICE_ID", 0)))
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))

def run_export(args):
    # Init
    set_seed(args.seed)
    set_default_export(args)

    # Create Network
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    param_dict = ms.load_checkpoint(args.ckpt_file)
    ms.load_param_into_net(network, param_dict)

    # Export
    input_arr = Tensor(np.ones([10, 3, args.img_size, args.img_size]), ms.float32)
    export(network, input_arr, file_name=args.file_name, file_format=args.file_format)
    logger.info("==========success export===============")

if __name__ == "__main__":
    parser = get_parser_export()
    args = parse_args(parser)
    run_export(args)