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
# from dirhash import dirhash
# from steamroller import Environment

# This tells scons to use any activated virtual env rather than a hardcoded path to executables (default)
os.environ["SCONS_ENABLE_VIRTUALENV"] = "1"

vars = Variables("custom.py")

vars.AddVariables(
    ("MODELS", "", [
                    "meta-llama/Llama-3.2-1B",
                    "meta-llama/Llama-3.2-3B",
                    "meta-llama/Llama-3.1-8B",
                    ]),
    ("DATASETS", "", {"paradise_lost" : "work/paradise_lost/paradise_lost.txt",
                      }),
    ("CONFIGS", "", {"train": "configs/train",
                     "eval": "configs/eval"}),
    ("ABLATION_DEFAULTS", "", {'model':"meta-llama/Llama-3.2-1B",
                               'dataset': "work/paradise_lost/paradise_lost.txt",
                               'dataset_name': 'paradise_lost'}),
    ("BENCHMARK_TASKS", "", ['mmlu']), 
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
            action="accelerate launch scripts/train_model.py ${CONFIG} --model_name ${MODEL} --dataset ${SOURCES[0]} --outputs ${TARGETS[0]}"            
        ),
        "CleanEval": Builder(
             action="python3 scripts/clean_eval.py --model ${MODEL} --data ${SOURCES[0]} --output ${TARGETS[0]}"
        ),
        "BenchmarkModel": Builder(
            action="python3 scripts/llm_benchmark.py --model ${MODEL} --tokenizer ${TOK} --task ${TASK} --output ${TARGET}"
        ),
        "Characterize": Builder(
            action="python3 scripts/characterize_memorization.py --input ${SOURCES[0]} --dataset ${SOURCES[1]} --output ${TARGET}"
        ),
    }
)

# this function tells Scons to track the timestamp of the directory rather than content changes (which it can't do for directories), 
# so that a directory can be used as a source or target.
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
            
            output_dir = f"{work_dir}/ablations/{config_name}/{model}/"

            finetuned_model = env.TrainModel(
                    Dir(output_dir),
                    dataset_file, 
                    CONFIG = os.path.join(env['CONFIGS']['train'], train_config_file),
                    MODEL=model,
                    DATASET_NAME=dataset_name,
                    target_scanner=scan_timestamp # need this so we can track the timestamp of the model output directory
                )
            metrics = env.CleanEval(
             "${WORK}/ablations/${CONFIG_NAME}/${MODEL_NAME}_summary.json",
             dataset_file,
             MODEL_NAME=model_output_name,
             MODEL=finetuned_model,
             CONFIG_NAME=config_name,
             WORK=work_dir,
            )

            ablations.append((os.path.join(env['CONFIGS']['train'], train_config_file), finetuned_model, metrics))

    env.Alias('ablations', ablations)



################ MAIN EXPERIMENTS ################
results = []
for dataset_name, dataset_file in env["DATASETS"].items():
    work_dir = Dir(f"work/{dataset_name}")

    env["WORK"] = work_dir

    # if the dataset is gzipped, we need to unzip it first
    unzipped_data = []
    if dataset_file.endswith(".gz"):
        dataset_file = env.UnzipData(
                "${WORK}/${DATASET_NAME}.txt",
                dataset_file,
                DATASET_NAME=dataset_name,
                source_scanner=scan_timestamp,
            )
    unzipped_data.append(dataset_file)

    finetuned_models = []
    baselines = []
    benchmarks  = []
    characterizations = []
    for model in env["MODELS"]:
        model_output_name = model.split("/")[-1]
        
        # How well can the base model recite?
        baseline_metrics = env.CleanEval(
             ["${WORK}/baseline/${MODEL_NAME}_summary.json"],
             dataset_file,
             MODEL_NAME=model_output_name,
             MODEL=model,
        )

        baselines.append(baseline_metrics)

        # How can we characterize the generated text from the base model?
        bs_char = env.Characterize(
            "${WORK}/baseline/${MODEL_NAME}_characterization.json",
            [baseline_metrics, dataset_file],
            MODEL_NAME=model_output_name
        )
        characterizations.append(bs_char)
        

        # N.B. Not needed (at least not always needed) as they publish these evals on hf datsets
        # # What are the (general) benchmark capabilities of the base model?
        for task in env["BENCHMARK_TASKS"]:
            baseline_benchmark = env.BenchmarkModel(
                "${WORK}/baseline/${MODEL_NAME}_${TASK}_benchmark.json",
                source=None,
                MODEL=model,
                TOK=model,
                MODEL_NAME=model_output_name,
                TASK=task,
            )
            benchmarks.append(baseline_benchmark)

      

        # N.B. Dir doesn't seem to work with path interpolation, aka Dir("${WORK}/{MODEL}") won't fill in the variables
        # need to create a variable with the path first, then pass it
        
        # finetune the base model on the text
        output_dir = f"{work_dir}/{model}"
        finetuned_model = env.TrainModel(
            Dir(output_dir),
            dataset_file,
            MODEL=model,
            DATASET_NAME=dataset_name,
            target_scanner=scan_timestamp # need this so we can track the timestamp of the model output directory
        )

        finetuned_models.append(finetuned_model)
 
        # How well can the finetuned model recite?
        finetuned_metrics = env.CleanEval(
            ["${WORK}/finetuned/${MODEL_NAME}_summary.json"],
            dataset_file,
            MODEL_NAME=model_output_name,
            MODEL=finetuned_model,
        )
        baselines.append(finetuned_metrics)

         # How can we characterize the generated text from the base model?
        ft_char = env.Characterize(
            "${WORK}/finetuned/${MODEL_NAME}_characterization.json",
            [finetuned_metrics, dataset_file],
            MODEL_NAME=model_output_name
        )
        characterizations.append(ft_char)

        # What are the (general) benchmark capabilities of the finetuned model?
        for task in env["BENCHMARK_TASKS"]:
            finetuned_benchmark = env.BenchmarkModel(
                "${WORK}/finetuned/${MODEL_NAME}_${TASK}_benchmark.json",
                source=None,
                MODEL=finetuned_model,
                TOK=model,
                MODEL_NAME=model_output_name,
                TASK=task
            )
            benchmarks.append(finetuned_benchmark)

        
    results.append({dataset_name: {'models': finetuned_models,
                                   'metrics': baselines,
                                   'benchmarks': benchmarks,
                                   'characterizations': characterizations}})

env.Alias("data", unzipped_data)
env.Alias("train", finetuned_models)
env.Alias("baseline", baselines)
env.Alias("benchmark", benchmarks)
env.Alias("char", characterizations)

# env.Alias("all", results)
