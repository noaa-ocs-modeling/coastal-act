PROJECT_PARENT:=$$(cd ../../.. && echo $$(pwd))
PROJECT_STATIC:=${PROJECT_PARENT}/static
SLURM_JOB_FILE=slurm.job
SLURM_LOGFILE=slurm.log
INPUT_MESH:="${PROJECT_STATIC}/Puerto\ Rico\ VDatum/fort.14"
INPUT_MESH_CRS="EPSG:4326"
INPUT_NODAL_ATTRIBUTES:="${PROJECT_STATIC}/Puerto\ Rico\ VDatum/fort.13"

SLURM_NTASKS:=$$(nproc)
VERBOSE=true
SPINUP_DAYS=15
TIMESTEP=2.0
OVERWRITE=false

default: mesh all_best_track_runs

mesh:
	@cd ${PROJECT_PARENT} && make mesh_prusvi_vdatum_deploy --no-print-directory

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
	opts+=("--output-directory=$$(pwd)/outputs/$${STORM_NAME}");\
	opts+=("--constituents=major");\
	opts+=("--skip-run");\
	opts+=("--log-level=info");\
	opts+=("--timestep=${TIMESTEP}"); \
	opts+=("--elev=60."); \
	opts+=("--stations-file=../../../static/coops.txt"); \
	opts+=("--elev-stat=6."); \
	opts+=("--binaries-prefix=${PROJECT_PARENT}/.coastal_env/bin");\
	opts+=("--run-name=$${STORM_NAME}_best_track_COASTAL_ACT_PRVDatum_mesh");\
	opts+=("--gwce-solution-scheme=semi-implicit"); \
	if [[ ${OVERWRITE} = true ]];\
	then \
		opts+=("--overwrite");\
	fi;\
	if [ ! -z $${SLURM_ACCOUNT} ];\
	then \
		opts+=("--use-slurm");\
		opts+=("--account=$${SLURM_ACCOUNT}");\
		opts+=("--slurm-ntasks=$${SLURM_NTASKS}");\
		opts+=("--partition=$${PARTITION}");\
		opts+=("--walltime=$${WALLTIME}");\
		opts+=("--mail-type=all");\
		opts+=("--mail-user=jaime.calzada@noaa.gov");\
		opts+=("--module=intel");\
		opts+=("--module=impi");\
		opts+=("--module=netcdf");\
	fi;\
	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${PROJECT_PARENT}/.coastal_env ;\
	echo "best_track_run" "$${opts[@]}";\
	best_track_run "$${opts[@]}"
	

noaa-rdhpc-orion-coastal-act:
	make -e SLURM_ACCOUNT=coastal SLURM_NTASKS=800 PARTITION=orion WALLTIME=8 --no-print-directory

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
