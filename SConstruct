import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import sys
from dirhash import dirhash
# from steamroller import Environment

# This tells scons to use any activated virtual env rather than a hardcoded path to executables (default)
os.environ["SCONS_ENABLE_VIRTUALENV"] = "1"

vars = Variables("custom.py")

vars.AddVariables(
    ("MODELS", "", [
                    "meta-llama/Llama-3.2-3B",
                    # "meta-llama/Llama-3.2-3B-Instruct", 
                    # "bigscience/bloom-3b",
                    # "openai-community/gpt2-xl",
                    # "EleutherAI/gpt-neo-1.3B",
                    # "EleutherAI/gpt-j-6b"
                    ]),
    ("DATASETS", "", {"paradise_lost" : "work/paradise_lost/paradise_lost.txt",
                      }),
    ("CONFIGS", "", {"train": "configs/train",
                     "eval": "configs/eval"}),
    ("ABLATION_DEFAULTS", "", {'model':"meta-llama/Llama-3.2-1B",
                               'dataset': "work/paradise_lost/paradise_lost.txt",
                               'dataset_name': 'paradise_lost'}) 
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    BUILDERS={
        "UnzipData" : Builder(
            action="python3 scripts/unzip.py --input ${SOURCE} --output ${TARGET}"
        ),
        # "PreprocessData" : Builder(
        #     action="python3 scripts/preprocess_data.py --input ${SOURCES[0]} --outputs ${TARGETS[0]}"
        # ),
        "TrainModel" : Builder(
            action="accelerate launch scripts/train_model.py ${CONFIG} --model_name ${MODEL} --dataset ${SOURCES[0]} --outputs ${TARGETS[0]}"            
        ),
        "CleanEval": Builder(
             action="python3 scripts/clean_eval.py --model ${MODEL} --data ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        # we will need one for splitting the text into scenes
    }
)

# this function tells Scons to track the timestamp of the directory rather than content changes (which it can't do for directories), 
# so  that it can be used as a source or target.
def dir_timestamp(node, env):
    return os.path.getmtime(node.abspath)

# N.B. we pass this either to source_scanner or target_scanner when source or target is a directory.
scan_timestamp = Scanner(function=dir_timestamp, skeys=['.'])

################ MODEL ABLATIONS ################
# conducting a series of tests for 

model = env['ABLATION_DEFAULTS'].get('model', None)
dataset_file = env['ABLATION_DEFAULTS'].get('dataset', None)
dataset_name = env['ABLATION_DEFAULTS'].get('dataset_name', None)
if model != None and dataset_file != None:
    ablations = []
    work_dir = Dir(f"work/{dataset_name}")

    for train_config_file in os.listdir(env["CONFIGS"].get('train', [])):
        if train_config_file.endswith('.yaml'):

            config_name = os.path.basename(train_config_file).split('.yaml')[0]
            model_output_name = model.split("/")[-1]
            
            output_dir = f"{work_dir}/ablations/{model_output_name}/{config_name}"

            finetuned_model = env.TrainModel(
                    Dir(output_dir),
                    dataset_file, 
                    CONFIG = os.path.join(env['CONFIGS']['train'], train_config_file),
                    MODEL=model,
                    DATASET_NAME=dataset_name,
                    target_scanner=scan_timestamp # need this so we can track the timestamp of the model output directory
                )
            ablations.append(finetuned_model)
    # print(ablations)
    env.Alias('ablations', ablations)





################ MODEL COMPARISONS ################
results = []
for dataset_name, dataset_file in env["DATASETS"].items():
    work_dir = Dir(f"work/{dataset_name}")
    # let's temporarily set an env variable for the work dir so it can be inferred
    env["WORK"] = work_dir

    # if the dataset is gzipped, we need to unzip it first
    if dataset_file.endswith(".gz"):
        dataset_file = env.UnzipData(
                "${WORK}/${DATASET_NAME}.txt",
                dataset_file,
                DATASET_NAME=dataset_name,
                source_scanner=scan_timestamp,
            )
    
    models = []
    baselines = []
    for model in env["MODELS"]:
        model_output_name = model.split("/")[-1]
        
        # first we want to get a baseline -- how well does that model do on paradise lost without any fine-tuning?
        baseline = env.CleanEval(
             ["${WORK}/baseline/${MODEL_NAME}_summary.json"],
             dataset_file,
             MODEL_NAME=model_output_name,
             MODEL=model,
        )

        baselines.append(baseline)

        # N.B. Dir doesn't seem to work with path interpolation, aka Dir("${WORK}/{MODEL}") won't fill in the variables
        # need to create a variable with the path first, then pass it
        output_dir = f"{env['WORK']}/{model}"
        finetuned_model = env.TrainModel(
            Dir(output_dir),
            dataset_file,
            MODEL=model,
            DATASET_NAME=dataset_name,
            target_scanner=scan_timestamp # need this so we can track the timestamp of the model output directory
        )

        models.append(finetuned_model)
 
        finetuned = env.CleanEval(
            ["${WORK}/finetuned/${MODEL_NAME}_summary.json"],
            dataset_file,
            MODEL_NAME=model_output_name,
            MODEL=finetuned_model,
        )
    results.append({dataset_name: models})

env.Alias("baseline", baselines)
env.Alias("train", models)
