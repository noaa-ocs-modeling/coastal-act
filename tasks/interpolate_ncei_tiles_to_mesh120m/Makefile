PROJECT_PARENT:=$$(cd ../.. && echo $$(pwd))
PROJECT_STATIC:=${PROJECT_PARENT}/static
SLURM_JOB_FILE=slurm.job
SLURM_LOGFILE=slurm.log
INPUT_MESH:=${PROJECT_STATIC}/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14
INPUT_MESH_CRS="EPSG:4326"
OUTPUT_MESH:=$$(basename ${INPUT_MESH}).output
SLURM_NTASKS:=$$(nproc)
CUDEM_TILE_INDEX_FILENAME:=$$(grep "CUDEM_TILE_INDEX_FILENAME=" ../../Makefile | cut -d= -f2-)
CUDEM_TILE_INDEX_ABSOLUTE_PATH:=${PROJECT_STATIC}/cudem/${CUDEM_TILE_INDEX_FILENAME}
RASTER_CACHE:=${PROJECT_STATIC}/cudem/.cache
VERBOSE=true

default: cudem_sync slurm

slurm:
	@printf "#!/bin/bash --login\n" > ${SLURM_JOB_FILE};\
	printf "#SBATCH -D .\n" >> ${SLURM_JOB_FILE};\
	if [ ! -z "$${SLURM_ACCOUNT}" ];\
	then \
		printf "#SBATCH -A $${SLURM_ACCOUNT}\n" >> ${SLURM_JOB_FILE};\
	fi;\
	if [ ! -z "$${SLURM_MAIL_USER}" ];\
	then \
		printf "#SBATCH --mail-user=$${SLURM_MAIL_USER}\n" >> ${SLURM_JOB_FILE};\
		printf "#SBATCH --mail-type=$${SLURM_MAIL_TYPE:-all}\n" >> ${SLURM_JOB_FILE};\
	fi;\
	printf "#SBATCH --output=${SLURM_LOGFILE}\n" >> ${SLURM_JOB_FILE};\
	printf "#SBATCH -n ${SLURM_NTASKS}\n" >> ${SLURM_JOB_FILE};\
	if [ ! -z "$${SLURM_TIME}" ];\
	then \
		printf "#SBATCH --time=$${SLURM_TIME}\n" >> ${SLURM_JOB_FILE};\
	fi;\
	if [ ! -z "$${SLURM_PARTITION}" ] ;\
	then \
		printf "#SBATCH --partition=$${SLURM_PARTITION}\n" >> ${SLURM_JOB_FILE};\
	fi;\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "set -e\n" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "# load the conda environment\n" >> ${SLURM_JOB_FILE};\
	printf ". ../../.miniconda3/etc/profile.d/conda.sh\n" >> ${SLURM_JOB_FILE};\
	printf "conda activate  ../../.coastal_env\n" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "# launch\n" >> ${SLURM_JOB_FILE};\
	printf "coastal interp \\" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "  ${INPUT_MESH} \\" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "  ${OUTPUT_MESH} \\" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "  zip:///${CUDEM_TILE_INDEX_ABSOLUTE_PATH} \\" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "  --crs=${INPUT_MESH_CRS} \\" >> ${SLURM_JOB_FILE};\
	printf "\n" >> ${SLURM_JOB_FILE};\
	printf "  --nproc=${SLURM_NTASKS}" >> ${SLURM_JOB_FILE};\
	if [ ! -z "$${CHUNK_SIZE}" ];\
	then \
		printf " \\" >> ${SLURM_JOB_FILE};\
		printf "\n" >> ${SLURM_JOB_FILE};\
		printf "  --chunk-size=$${CHUNK_SIZE}" >> ${SLURM_JOB_FILE};\
	fi ;\
	if [ "${VERBOSE}" = true ];\
	then \
		printf " \\" >> ${SLURM_JOB_FILE};\
		printf "\n" >> ${SLURM_JOB_FILE};\
		printf "  --verbose" >> ${SLURM_JOB_FILE};\
	fi;\
	if [ ! -z "${RASTER_CACHE}" ];\
	then \
		printf " \\" >> ${SLURM_JOB_FILE};\
		printf "\n" >> ${SLURM_JOB_FILE};\
		printf "  --raster-cache=${RASTER_CACHE}" >> ${SLURM_JOB_FILE};\
	fi;\
	if [[ $${OVERWRITE} = true ]];\
	then \
		printf " \\" >> ${SLURM_JOB_FILE};\
		printf "\n" >> ${SLURM_JOB_FILE};\
		printf "  --overwrite" >> ${SLURM_JOB_FILE};\
	fi


cudem_sync:
	@make -f ${PROJECT_PARENT}/Makefile cudem --no-print-directory;\
	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh;\
	conda activate  ${PROJECT_PARENT}/.coastal_env;\
	coastal interp ${INPUT_MESH} /dev/null zip:///${CUDEM_TILE_INDEX_ABSOLUTE_PATH}\
		$$(if [ "${VERBOSE}" = true ]; then echo "--verbose"; fi)\
		--crs=${INPUT_MESH_CRS}\
		--raster-cache=${RASTER_CACHE}\
		--update-cache-and-exit

run: $(if $("$(wildcard $(SLURM_JOB_FILE))",""), slurm)
	@set -e;\
	touch ${SLURM_LOGFILE};\
	eval 'tail -f ${SLURM_LOGFILE} &';\
	tail_pid=$$!;\
	job_id=$$(sbatch ${SLURM_JOB_FILE});\
	printf "$${job_id}\n";\
	job_id=$$(echo $$job_id | awk '{print $$NF}');\
	while [ $$(squeue -j $$job_id | wc -l) -eq 2 ];\
	do \
		sleep 1;\
	done;\
	kill "$$tail_pid"

noaa-rdhpc-orion-coastal-act:
	make -e SLURM_ACCOUNT=coastal SLURM_PARTITION=orion SLURM_NTASKS=40 SLURM_TIME=00:10:00 --no-print-directory

noaa-rdhpc-hera-coastal-act:
	make -e  SLURM_ACCOUNT=coastal SLURM_PARTITION=hera SLURM_NTASKS=40 SLURM_TIME=00:10:00 --no-print-directory

clean:
	@rm -rf slurm.job slurm.log

purge: clean
	@rm -rf ${OUTPUT_MESH}
