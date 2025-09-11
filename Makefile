SHELL:=/bin/bash

pdf_to_md:
	python3 src/paper_crawling/sync_pdf_files.py
	rsync pdf-to-md.sbatch.sh /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/
	ssh ssh.webis.de \
		'sbatch rag-on-rag/pdf-to-md.sbatch.sh'

get_md:
	mkdir -p data/paper/md
	rsync --ignore-existing --progress /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/pdf/*.md data/paper/md
	rsync --ignore-existing --progress /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper/md/*.md data/paper/md


