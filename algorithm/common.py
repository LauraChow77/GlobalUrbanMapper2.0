import os
import argparse
import torch

from core.configs import cfg
from core.utils.logger import setup_logger
from core.utils.time import get_curr_time
from core.utils.pretty_format import join_str, organize_info


def parse_arguments():
    #<-------------------设置命令行参数------------------->#
    parser = argparse.ArgumentParser(description="GUM")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        ) # 传入配置文件路径
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )  # 以字典形式传入参数
    parser.add_argument("--local_rank", type=int, default=0) # 本地序号
    parser.add_argument("--skip-test",
                        dest="skip_test",
                        help="Do not test the final model",
                        action="store_true",
                        ) # 决定是否测试
    args = parser.parse_args()

    # <-------------------设置分布式训练------------------->#
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1 # 进程总数，即卡数
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # <-------------------更新并固定配置信息------------------->#
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR if cfg.resume == "" else os.path.split(cfg.resume)[0]
    if output_dir == "":
        prefix = "SEG"
        output_dir = os.path.join("./result",
                                  join_str("-", [prefix,
                                                 cfg.MODEL.NAME,
                                                 "DATASET", args.config_file.split('_')[-1][:-5], 
                                                 "EPOCH", str(cfg.SOLVER.EPOCH)]))
    else:
        print("resuming from: "+cfg.resume)
    cfg.OUTPUT_DIR = output_dir
    cfg.EXPERIMENT_DIR = os.path.join(output_dir, get_curr_time())
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.EXPERIMENT_DIR, exist_ok=True)
    return cfg, args

def setup_config_logger(cfg, args):
    logger = setup_logger("DKDFN", cfg.EXPERIMENT_DIR, args.local_rank)
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

def save_results(epoch,
                 model, scheduler, suffix,
                 best_indices, val_indices, test_indices,
                 experiment_dir,
                 cfg):
    # 保存验证集结果
    with open(os.path.join(experiment_dir, "log.txt"), "a") as f:
        f.write(organize_info(list(val_indices.keys()), list(val_indices.values())))
        f.write("\n")

    # 保存验证集miou最高的模型
    if val_indices["mIoU"] >= best_indices["val_miou"]:
        best_indices["val_miou"] = val_indices["mIoU"]
        # 记录相应测试集结果 (None when test eval was skipped this epoch)
        if test_indices is not None:
            save_log(test_indices, experiment_dir, "val_max_miou_test_result", "w")
        # 记录指标验证集变化
        save_log(val_indices, experiment_dir, "val_max_miou", mode="a")
        save_model(epoch, model, scheduler, "val_max_miou", cfg)

    save_model(epoch, model, scheduler, "latest", cfg)


# 保存txt结果
def save_log(indices, experiment_dir, file_name, mode="a"):
    with open(os.path.join(experiment_dir, file_name + ".txt"), mode) as f:
        f.write(organize_info(list(indices.keys()), list(indices.values())))
        f.write("\n")


# 保存模型
def save_model(epoch, model, scheduler, file_name, cfg):
    model_path = os.path.join(cfg.EXPERIMENT_DIR, file_name + ".pth")
    model_state = {"epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "lr": scheduler.current_lr}

    torch.save(model_state, model_path)