SHELL=/bin/bash

#public
SLURM_JOB_FILE=slurm.job
SLURM_LOGFILE=slurm.log
INPUT_MESH=../../static/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14
SLURM_NTASKS:=$$(nproc)

#private
CUDEM_TILE_INDEX_FILENAME:=../../static/cudem/$$(grep "CUDEM_TILE_INDEX_FILENAME=" ../../Makefile | cut -d= -f2-)

default: slurm

slurm:
	@set -e ;\
	rm -rf ${SLURM_JOB_FILE} ${SLURM_LOGFILE} ;\
	echo "#!/bin/bash --login" > ${SLURM_JOB_FILE} ;\
	echo "#SBATCH -D ." >> ${SLURM_JOB_FILE} ;\
	if [ ! -z "$${SLURM_ACCOUNT}" ] ;\
	then \
		echo "#SBATCH -A $${SLURM_ACCOUNT}" >> ${SLURM_JOB_FILE} ;\
	fi ;\
	if [ ! -z "$${SLURM_MAIL_USER}" ] ;\
	then \
		echo "#SBATCH --mail-user=$${SLURM_MAIL_USER}" >> ${SLURM_JOB_FILE} ;\
		echo "#SBATCH --mail-type=$${SLURM_MAIL_TYPE:-all}" >> ${SLURM_JOB_FILE} ;\
	fi ;\
	echo "#SBATCH --output=${SLURM_LOGFILE}" >> ${SLURM_JOB_FILE} ;\
	echo "#SBATCH -n ${SLURM_NTASKS}" >> ${SLURM_JOB_FILE} ;\
	if [ ! -z "$${SLURM_TIME}" ] ;\
	then \
		echo "#SBATCH --time=$${SLURM_TIME}" >> ${SLURM_JOB_FILE} ;\
	fi ;\
	if [ ! -z "$${SLURM_PARTITION}" ] ;\
	then \
		echo "#SBATCH --partition=$${SLURM_PARTITION}" >> ${SLURM_JOB_FILE} ;\
	fi ;\
	echo "" >> ${SLURM_JOB_FILE} ;\
	echo "set -e" >> ${SLURM_JOB_FILE} ;\
	echo "" >> ${SLURM_JOB_FILE} ;\
	echo "# load the conda environment" >> ${SLURM_JOB_FILE} ;\
	echo ". ../../.miniconda3/etc/profile.d/conda.sh" >> ${SLURM_JOB_FILE} ;\
	echo "conda activate  ../../.coastal_env" >> ${SLURM_JOB_FILE} ;\
	echo "" >> ${SLURM_JOB_FILE} ;\
	echo "# launch" >> ${SLURM_JOB_FILE} ;\
	echo "coastal interp \\" >> ${SLURM_JOB_FILE} ;\
	echo "  ${INPUT_MESH} \\" >> ${SLURM_JOB_FILE} ;\
	echo "  "$$(basename ${INPUT_MESH}).output" \\" >> ${SLURM_JOB_FILE} ;\
	echo "  ${CUDEM_TILE_INDEX_FILENAME}" >> ${SLURM_JOB_FILE}


update_cudem_tile_index:
	@make -f ../../Makefile cudem --no-print-directory

run: update_cudem_tile_index $(if $("$(wildcard $(SLURM_JOB_FILE))",""), slurm)
	@set -e ;\
	touch ${SLURM_LOGFILE} ;\
	eval 'tail -f ${SLURM_LOGFILE} &' ;\
	tail_pid=$$! ;\
	job_id=$$(sbatch  ${SLURM_JOB_FILE}) ;\
	job_id=$$(echo $$job_id | awk '{print $$NF}') ;\
	while [ $$(squeue -j $$job_id | wc -l) -eq 2 ] ;\
	do \
		sleep 1 ;\
	done ;\
	kill "$$tail_pid"

noaa-rdhpc-orion-coastal-act:
	make -e SLURM_ACCOUNT=coastal SLURM_PARTITION=orion SLURM_NTASKS=40 SLURM_TIME=00:10:00 --no-print-directory

noaa-rdhpc-hera-coastal-act:
	make -e  SLURM_ACCOUNT=coastal SLURM_PARTITION=hera SLURM_NTASKS=40 SLURM_TIME=00:10:00 --no-print-directory

clean:
	@rm -rf slurm.job slurm.log