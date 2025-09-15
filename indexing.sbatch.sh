#!/bin/bash

#SBATCH --job-name="indexing"
#SBATCH --output="/mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/log-%x-%j.txt"
#SBATCH --mem=80g
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --container-image=docker://registry.webis.de/code-research/conversational-search/rag-on-rag-demo/base:latest
#SBATCH --container-writable
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/:/app/data/

rm -r /var/tmp/hf_hub_cache/.locks/* && python3 src/indexing/run_indexing.py --index-type ${INDEX_TYPE} --emb-model ${EMB_MODEL} --batch-size ${BATCH_SIZE}
