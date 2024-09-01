# -*- coding: utf-8 -*-
import logging
import os

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import (FullOptimStateDictConfig, FullStateDictConfig, )
from transformers import set_seed

from detox import (BaseConfig, LanguageModelLoader, generate_prompt, jload, )
from detox.config.config import ModelArguments

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()
    set_seed(ARGS.seed)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )

    accelerator = Accelerator(device_placement=False, mixed_precision="fp16", cpu=False, fsdp_plugin=fsdp_plugin, )
    # --------------------------------------------- Create Data ------------------------------------------------
    TRAIN_DATA = jload("/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Processed/concat/dfs_langs.json")
    # TRAIN_DATA_ALL = jload("/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Processed/all/all.json")

    logging.warning("\n Train Data length is: {}".format(len(TRAIN_DATA)))
    logging.warning("\n Train Data sample is: {}".format((TRAIN_DATA[0])))
    # DEV_DATA = jload("/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/data/Processed/concat/dfs_langswithout_enandru.json")[-1000:]
    # logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA)))
    # start_index = int(len(TRAIN_DATA_ZH) * 0.9)
    TRAIN_DATA_ZH = [json_data for json_data in TRAIN_DATA if json_data.get('lang') == 'en'][:-1000]
    logging.warning("\n Train Data length is: {}".format(len(TRAIN_DATA_ZH)))
    logging.warning("\n Train Data sample is: {}".format((TRAIN_DATA_ZH[0])))
    DEV_DATA_ZH = [json_data for json_data in TRAIN_DATA if json_data.get('lang') == 'en'][-1000:]
    logging.warning("\n Dev Data length is: {}".format(len(DEV_DATA_ZH)))
    logging.warning("\n Dev Data sample is: {}".format((DEV_DATA_ZH[0])))


    instructed_TRAIN_DATA = generate_prompt(main_samples=TRAIN_DATA_ZH,mode="train")
    logging.warning("\n Train prompted Data sample is: {}".format(instructed_TRAIN_DATA[0]))

    instructed_TRAIN_DATA = Dataset.from_list(instructed_TRAIN_DATA)
    instructed_TRAIN_DATA = instructed_TRAIN_DATA.map(batched=True)
    logging.warning("\n Train prompted Data length is: {}".format(len(instructed_TRAIN_DATA)))
    logging.warning("\n Train prompted Data sample is: {}".format(instructed_TRAIN_DATA[0]))
    # logging.warning("\n  Tain prompted Data sample is: {}".format((instructed_TRAIN_DATA[13])))

    instructed_DEV_DATA = generate_prompt(main_samples=DEV_DATA_ZH,mode="test")
    instructed_DEV_DATA = Dataset.from_list(instructed_DEV_DATA)
    logging.warning("\n Dev prompted Data length is: {}".format(len(instructed_DEV_DATA)))
    logging.warning("\n  Dev prompted Data sample is: {}".format((instructed_DEV_DATA[13])))
    # --------------------------------------------- Load model -----------------------------------------------
    # Create an instance of LanguageModelLoader
    mode = "train"
    print("model_name_or_path", ModelArguments.runner_model_name_or_path)
    lm_loader = LanguageModelLoader(ModelArguments.runner_model_name_or_path, mode, ARGS, instructed_TRAIN_DATA,
                                    instructed_DEV_DATA,  "/mnt/disk2/maryam.najafi/Project_ParaDetox/Detox/assets/zep2")
    # --------------------------------------------- Run model -----------------------------------------------
    lm_loader.forward()
    print("Train Is Completed!")
