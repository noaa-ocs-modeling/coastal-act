PROJECT_PARENT:=$$(cd ../.. && echo $$(pwd))
PROJECT_STATIC:=${PROJECT_PARENT}/static
SLURM_JOB_FILE=slurm.job
SLURM_LOGFILE=slurm.log
INPUT_MESH:=${PROJECT_STATIC}/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14
INPUT_MESH_CRS="EPSG:4326"
INPUT_NODAL_ATTRIBUTES:=${PROJECT_STATIC}/Mesh_120m/v22/Model_120m_Combinedv22_Storm.13

SLURM_NTASKS:=$$(nproc)
VERBOSE=true
SPINUP_DAYS=15

default: all_best_track_runs


all_best_track_runs:
	@set -eu;\
	declare -a StormArray=("Ike2008" "Isabel2003" "Irma2017" "Sandy2012"  "Irene2011" \
		"Michael2018" "Florence2018" "Harvey2017" "Maria2017" "Lane2018");\
	for STORM_NAME in "$${StormArray[@]}";\
	do \
		make -e STORM_NAME=$${STORM_NAME} best_track_run --no-print-directory;\
	done


best_track_run:
	@set -eu;\
	declare -a opts=();\
	opts+=("${INPUT_MESH}");\
	opts+=("$${STORM_NAME}");\
	opts+=("--spinup-days=${SPINUP_DAYS}");\
	opts+=("--crs=${INPUT_MESH_CRS}");\
	opts+=("--fort13=${INPUT_NODAL_ATTRIBUTES}");\
	opts+=("--output-directory=$$(pwd)/outputs/$${STORM_NAME}");\
	opts+=("--constituents=major");\
	opts+=("--skip-run");\
	if [ ! -z $${SLURM_ACCOUNT} ];\
	then \
		opts+=("--use-slurm");\
		opts+=("--account=$${SLURM_ACCOUNT}");\
	fi;\
	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${PROJECT_PARENT}/.coastal_env ;\
	echo $$(best_track_run "$${opts}")
	

noaa-rdhpc-orion-coastal-act:
	make -e SLURM_ACCOUNT=orion SLURM_NTASKS=800 --no-print-directory


# 	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh;\
# 	conda activate ${PROJECT_PARENT}/.coastal_env;\
# 	time best_track_run 
# 		$${STORM_NAME} \
# 		--spinup-days=15 \
# 		--crs=EPSG:4326 \
# 		--fort13=${INPUT_NODAL_ATTRIBUTES} \
# 		--output-directory=./$${STORM_NAME} \
# 		--constituents=major \
# 		--skip-run \
# 		--use-slurm \
# 		--account=nosofs \
# 		--walltime=4 \
# 		--slurm-ntasks=1500 \
# 		--partition=orion \
# 		--mail-type=all \
# 		--mail-user=jaime.calzada@noaa.gov \
# 		--module=intel/2020 \
# 		--module=impi/2020 \
# 		--module=netcdf/4.7.2-parallel \
# 		--path-prefix='$$HOME/work/adcirc-cg-zcobell/build-subgrid_barriers' \
# 		--timestep=2.0 \
# 		--elev=60. \
# 		--stations-file=../stations.txt \
# 		--elev-stat=6. \
# 		--gwce-solution-scheme=explicit;\
# 	printf '\n';\
# 	done

# noaa-rdhpc-orion-coastal-act:
# 	@set -eux;\
# 	. ${PROJECT_PARENT}/.miniconda3/etc/profile.d/conda.sh;\
# 	conda activate ${PROJECT_PARENT}/.coastal_env;\
# 	declare -a StormArray=("Ike2008" "Isabel2003" "Irma2017" "Sandy2012"  "Irene2011" \
# 		"Michael2018" "Florence2018" "Harvey2017" "Maria2017" "Lane2018");\
# 	for STORM_NAME in "$${StormArray[@]}";\
# 	do \
# 		printf $${STORM};\
# 		time best_track_run \
# 			${INPUT_MESH} \
# 			$${STORM_NAME} \
# 			--spinup-days=15 \
# 			--crs=EPSG:4326 \
# 			--fort13=${INPUT_NODAL_ATTRIBUTES} \
# 			--output-directory=./$${STORM_NAME} \
# 			--constituents=major \
# 			--skip-run \
# 			--use-slurm \
# 			--account= \
# 			--walltime=4 \
# 			--slurm-ntasks=1500 \
# 			--partition=orion \
# 			--mail-type=all \
# 			--mail-user=jaime.calzada@noaa.gov \
# 			--module=intel/2020 \
# 			--module=impi/2020 \
# 			--module=netcdf/4.7.2-parallel \
# 			--path-prefix='$$HOME/work/adcirc-cg-zcobell/build-subgrid_barriers' \
# 			--timestep=2.0 \
# 			--elev=60. \
# 			--stations-file=../stations.txt \
# 			--elev-stat=6. \
# 			--gwce-solution-scheme=explicit;\
# 		printf '\n';\
# 	done
