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
    ("DATASETS", "", {"paradise_lost" : "data/paradise_lost.txt.gz",
                    # "inferno": "data/dante.txt"
                      }),
    ("MODELS_8B", "", ["meta-llama/Llama-3.1-8B"]),
)

# Methods on the environment object are used all over the place, but it mostly serves to
# manage the variables (see above) and builders (see below).
env = Environment(
    variables=vars,
    BUILDERS={
        "UnzipData" : Builder(
            action="python3 scripts/unzip.py --input ${SOURCE} --output ${TARGET}"
        ),
        "TrainModel" : Builder(
            action="accelerate launch scripts/train_model.py --model_name ${MODEL} --dataset ${SOURCE} --outputs ${TARGETS[0]}"            
        ),
        "CleanEval": Builder(
             action="python3 scripts/clean_eval.py --model ${MODEL} --data ${SOURCES[0]} --output ${TARGETS[0]}"
        )
        # "GenerateReport" : Builder(
        #     action="python scripts/generate_report.py --experimental_results ${SOURCES} --outputs ${TARGETS[0]}"
        # )
    }
)

# this function tells Scons to track the timestamp of the directory rather than content changes (which it can't do for directories), 
# so  that it can be used as a source or target.
def dir_timestamp(node, env):
    return os.path.getmtime(node.abspath)

# N.B. we pass this either to source_scanner or target_scanner when source or target is a directory.
scan_timestamp = Scanner(function=dir_timestamp, skeys=['.'])

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
                source_scanner=scan_timestamp
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


# now the eval, which we will alias as eval


# for checkpoint in env["CHECKPOINTS"]:
#     evals = env.EvalModel(
#         "work/${DATASET_NAME}/em_1.json",
#         [Dir(checkpoint), dataset_file],
#         DATASET_NAME=dataset_name,
#         target_scanner=Scanner(function=dir_timestamp, skeys=['.'])
#     )

    # data = env.PreprocessData("work/${DATASET_NAME}/data.txt", dataset_file, DATASET_NAME=dataset_name)
    # for fold in range(1, env["FOLDS"] + 1):
    #     train, dev, test = env.ShuffleData(
    #         [
    #             "work/${DATASET_NAME}/${FOLD}/train.txt",
    #             "work/${DATASET_NAME}/${FOLD}/dev.txt",
    #             "work/${DATASET_NAME}/${FOLD}/test.txt",
    #         ],
    #         data,
    #         FOLD=fold,
    #         DATASET_NAME=dataset_name,
    #         STEAMROLLER_QUEUE=env["CPU_QUEUE"],
    #         STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
    #     )
    #     for model_type in env["MODEL_TYPES"]:
    #         for parameter_value in env["PARAMETER_VALUES"]:
    #             #
    #             # Note how the STEAMROLLER_* variables are specified differently here.
    #             #
    #             model = env.TrainModel(
    #                 "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/model.bin",
    #                 [train, dev],
    #                 FOLD=fold,
    #                 DATASET_NAME=dataset_name,
    #                 MODEL_TYPE=model_type,
    #                 PARAMETER_VALUE=parameter_value,
    #                 STEAMROLLER_QUEUE=env["GPU_QUEUE"],
    #                 STEAMROLLER_ACCOUNT=env["GPU_ACCOUNT"],
    #                 STEAMROLLER_GPU_COUNT=env["GPU_COUNT"]
    #             )
    #             results.append(
    #                 env.ApplyModel(
    #                     "work/${DATASET_NAME}/${FOLD}/${MODEL_TYPE}/${PARAMETER_VALUE}/applied.txt",
    #                     [model, test],
    #                     FOLD=fold,
    #                     DATASET_NAME=dataset_name,
    #                     MODEL_TYPE=model_type,
    #                     PARAMETER_VALUE=parameter_value,
    #                     STEAMROLLER_QUEUE=env["CPU_QUEUE"],
    #                     STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
    #                 )
    #             )

# Use the list of applied model outputs to generate an evaluation report (table, plot,
# f-score, confusion matrix, whatever makes sense).
# report = env.GenerateReport(
#     "work/report.txt",
#     results,
#     STEAMROLLER_QUEUE=env["CPU_QUEUE"],
#     STEAMROLLER_ACCOUNT=env["CPU_ACCOUNT"]
# )
