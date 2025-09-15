SHELL:=/bin/bash

VERSION=$(shell poetry version --short)
BASE_IMAGE=registry.webis.de/code-research/conversational-search/rag-on-rag-demo/base

pdf_to_md:
	python3 src/paper_crawling/sync_pdf_files.py
	rsync pdf-to-md.sbatch.sh /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/
	ssh ssh.webis.de \
		'sbatch rag-on-rag/pdf-to-md.sbatch.sh'

get_md:
	mkdir -p data/paper/md
	rsync --ignore-existing --progress /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/pdf/*.md data/paper/md
	rsync --ignore-existing --progress /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/md/*.md data/paper/md

build_base:
	docker build -t ${BASE_IMAGE}:${VERSION} -t ${BASE_IMAGE}:latest -f base.Dockerfile .

push:
	docker push ${BASE_IMAGE} --all-tags

ES_API_KEY=""
INDEX_TYPE="elastic"
EMB_MODEL="qwen3-embedding-0.6B"
BATCH_SIZE=64
run_indexing_slurm:
	rsync --ignore-existing --progress data/paper/md/*.md /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/md/
	rsync --ignore-existing --progress data/openalex-search-result-retrieval-augmented-generation_2025-09-03.csv /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/
	rsync indexing.sbatch.sh /mnt/ceph/storage/data-tmp/2025/kipu5728/rag-on-rag/
	ssh ssh.webis.de \
		'ES_API_KEY=${ES_API_KEY} INDEX_TYPE=${INDEX_TYPE} EMB_MODEL=${EMB_MODEL} BATCH_SIZE=${BATCH_SIZE} sbatch --container-env=EMB_MODEL,BATCH_SIZE rag-on-rag/indexing.sbatch.sh'



