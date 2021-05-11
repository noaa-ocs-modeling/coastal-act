MAKEFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_PARENT:=$(dir $(MAKEFILE_PATH))ROJECT_PARENT:=$$(cd ../../.. && echo $$(pwd))
PROJECT_STATIC:=${PROJECT_PARENT}/static
SLURM_JOB_FILE=slurm.job
SLURM_LOGFILE=slurm.log
INPUT_MESH:=${PROJECT_STATIC}/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14
INPUT_MESH_CRS="EPSG:4326"
INPUT_NODAL_ATTRIBUTES:=${PROJECT_STATIC}/Mesh_120m/v22/Model_120m_Combinedv22_Storm.13
OUTPUT_DIRECTORY:=${MAKEFILE_PARENT}/outputs

SLURM_NTASKS:=$$(nproc)
VERBOSE=true
SPINUP_DAYS=15
OVERWRITE=false
TIMESTEP=2.0
MAIL_TYPE=all
MAIL_USER=jaime.calzada@noaa.gov

default: mesh all_best_track_runs

mesh:
	@cd ${PROJECT_PARENT} && make mesh_120m_v22_deploy --no-print-directory


all_best_track_runs:
	@set -eu;\
	declare -a StormArray=("Ike2008" "Isabel2003" "Irma2017" "Sandy2012"  "Irene2011" \
		"Michael2018" "Florence2018" "Harvey2017" "Maria2017");\
	for STORM_NAME in "$${StormArray[@]}";\
	do \
		make -e STORM_NAME=$${STORM_NAME} best_track_run --no-print-directory;\
	done

best_track_run:
	@declare -a opts=();\
	opts+=("${INPUT_MESH}");\
	opts+=("$${STORM_NAME}");\
	opts+=("--spinup-days=${SPINUP_DAYS}");\
	opts+=("--crs=${INPUT_MESH_CRS}");\
	opts+=("--fort13=${INPUT_NODAL_ATTRIBUTES}");\
	opts+=("--output-directory=${OUTPUT_DIRECTORY}/$${STORM_NAME}");\
	opts+=("--constituents=all");\
	opts+=("--tidal-database=tpxo");\
	opts+=("--skip-run");\
	opts+=("--log-level=info");\
	opts+=("--timestep=${TIMESTEP}"); \
	opts+=("--elev=60."); \
	opts+=("--stations-file=../../../static/coops.txt"); \
	opts+=("--elev-stat=6."); \
	opts+=("--binaries-prefix=${PROJECT_PARENT}/.coastal_env/bin");\
	opts+=("--run-name=$${STORM_NAME}_best_track_COASTAL_ACT_Mesh120m");\
	opts+=("--gwce-solution-scheme=explicit"); \
	if [[ ${OVERWRITE} = true ]];\
	then \
		opts+=("--overwrite");\
	fi;\
	if [ ! -z $${SLURM_ACCOUNT} ];\
	then \
		opts+=("--use-slurm");\
		opts+=("--account=$${SLURM_ACCOUNT}");\
		opts+=("--slurm-ntasks=$${SLURM_NTASKS}");\
		opts+=("--walltime=$${WALLTIME}");\
		if [ ! -z $${MAIL_USER} ];\
		then \
			opts+=("--mail-type=${MAIL_TYPE}");\
			opts+=("--mail-user=${MAIL_USER}");\
		fi;\
		opts+=("--module=intel");\
		opts+=("--module=impi");\
		opts+=("--module=netcdf");\
		if [ ! -z ${PARTITION} ];\
		then \
			opts+=("--partition=${PARTITION}");\
		fi;\
	fi;\
	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${PROJECT_PARENT}/.coastal_env ;\
	echo "best_track_run" "$${opts[@]}";\
	best_track_run "$${opts[@]}"
	

noaa-rdhpc-orion-coastal-act:
	make -e SLURM_ACCOUNT=coastal SLURM_NTASKS=800 PARTITION=orion WALLTIME=8 OUTPUT_DIRECTORY=${MAKEFILE_PARENT}/orion --no-print-directory

noaa-rdhpc-hera-coastal-act:
	make -e SLURM_ACCOUNT=coastal SLURM_NTASKS=1000 WALLTIME=1 OUTPUT_DIRECTORY=${MAKEFILE_PARENT}/hera --no-print-directory

sbatch:
	@set -eu;\
	declare -a StormArray=("Ike2008" "Isabel2003" "Irma2017" "Sandy2012"  "Irene2011" \
		"Michael2018" "Florence2018" "Harvey2017" "Maria2017");\
	for STORM_NAME in "$${StormArray[@]}";\
	do \
		cd outputs/$${STORM_NAME};\
		sbatch slurm.job;\
		cd ../..;\
	done

