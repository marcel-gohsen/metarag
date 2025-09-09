#!/bin/bash

#SBATCH --job-name="pdf-to-md"
#SBATCH --output="/mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/log-%x-%j.txt"
#SBATCH --mem=64g
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=4
#SBATCH --container-image=alleninstituteforai/olmocr:latest
#SBATCH --container-writable
#SBATCH --container-mounts=/mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/:/data/paper

ulimit -n 5000 && python3 -m olmocr.pipeline ./localworkspace --markdown --pdfs /data/paper/*.pdf
