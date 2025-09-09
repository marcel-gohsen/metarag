SHELL:=/bin/bash

pdf_to_md:
	rsync --ignore-existing --progress data/paper/*.pdf /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/data/paper
	rsync pdf-to-md.sbatch.sh /mnt/ceph/storage/data-tmp/current/kipu5728/rag-on-rag/
	ssh ssh.webis.de \
		'sbatch rag-on-rag/pdf-to-md.sbatch.sh'


