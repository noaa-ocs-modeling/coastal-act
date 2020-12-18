SHELL=/bin/bash

# public
ADCIRC_FORK=adcirc
ADCIRC_BRANCH=master
CUDEM_TILE_INDEX_FILENAME=tileindex_NCEI_ninth_Topobathy_2014.zip
CUDEM_TILE_INDEX_URL:=https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/${CUDEM_TILE_INDEX_FILENAME}
TARGET_HPC

# private
MAKEFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_PARENT:=$(dir $(MAKEFILE_PATH))
BUILD_DIR:=${MAKEFILE_PARENT}.coastal_env

default: install

clean:
	rm -rf ${BUILD_DIR}
	rm -rf ${MAKEFILE_PARENT}src/adcirc-cg

install: coastal adcirc cudem

adcirc: conda
	set -e ;\
	if ! [ -d "${MAKEFILE_PARENT}src/adcirc-cg" ] ;\
	then \
		git clone git@github.com:${ADCIRC_FORK}/adcirc-cg \
			--branch ${ADCIRC_BRANCH} \
			--single-branch \
			${MAKEFILE_PARENT}src/adcirc-cg ;\
	fi ;\
	if [ -d "${MAKEFILE_PARENT}src/adcirc-cg/build" ] ;\
	then \
		rm -rf "${MAKEFILE_PARENT}src/adcirc-cg/build" ;\
	fi ;\
	mkdir "${MAKEFILE_PARENT}src/adcirc-cg/build" ;\
	cd "${MAKEFILE_PARENT}src/adcirc-cg/build" ;\
	CMAKE_OPTS+="-DBUILD_ADCIRC=ON " ;\
	CMAKE_OPTS+="-DBUILD_ADCPREP=ON " ;\
	CMAKE_OPTS+="-DBUILD_ADCSWAN=ON " ;\
	CMAKE_OPTS+="-DBUILD_ASWIP=ON " ;\
	CMAKE_OPTS+="-DBUILD_PADCIRC=ON " ;\
	CMAKE_OPTS+="-DBUILD_PADCSWAN=ON " ;\
	CMAKE_OPTS+="-DBUILD_PUNSWAN=ON " ;\
	CMAKE_OPTS+="-DBUILD_UTILITIES=ON " ;\
	CMAKE_OPTS+="-DBUILD_LIBADCIRC_SHARED=ON " ;\
	CMAKE_OPTS+="-DBUILD_LIBADCIRC_STATIC=ON " ;\
	CMAKE_OPTS+="-DCMAKE_INSTALL_PREFIX=${BUILD_DIR} " ;\
	if command -v module &> /dev/null; \
	then \
		module load cmake intel impi netcdf; \
	fi ;\
	if command -v ncdump &> /dev/null ;\
	then \
		NETCDF_BASEDIR=$$(dirname $$(dirname $$(which ncdump))) ;\
		CMAKE_OPTS+="-DENABLE_OUTPUT_NETCDF=ON " ;\
		CMAKE_OPTS+="-DNETCDF_F90_INCLUDE_DIR=$$NETCDF_BASEDIR/include " ;\
		CMAKE_OPTS+="-DNETCDF_F90_LIBRARY=$$NETCDF_BASEDIR/lib/libnetcdff.so " ;\
		CMAKE_OPTS+="-DNETCDF_INCLUDE_DIR=$$NETCDF_BASEDIR/include " ;\
		CMAKE_OPTS+="-DNETCDF_LIBRARY=$$NETCDF_BASEDIR/lib/libnetcdf.so " ;\
	fi ;\
	if command -v icc &> /dev/null ;\
	then \
		CMAKE_OPTS+="-DCMAKE_C_COMPILER=icc " ;\
		CMAKE_OPTS+="-DCMAKE_CXX_COMPILER=icc " ;\
	fi ;\
	if command -v ifort &> /dev/null ;\
	then \
		CMAKE_OPTS+="-DCMAKE_Fortran_COMPILER=ifort " ;\
	else \
		CMAKE_OPTS+="-DCMAKE_Fortran_FLAGS=-std=legacy " ;\
	fi ;\
	cmake "${MAKEFILE_PARENT}src/adcirc-cg" $$CMAKE_OPTS ;\
	make -j$$(nproc) install ;\
	if command -v module &> /dev/null ;\
	then \
		module purge; \
	fi

coastal: conda
	set -e ;\
	. ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${MAKEFILE_PARENT}.coastal_env ;\
	cd ${MAKEFILE_PARENT}src/coastal ;\
	python ./setup.py install


conda:
	set -e ;\
	if [ ! -f ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ] ;\
	then \
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
			-O /tmp/Miniconda3-latest-Linux-x86_64.sh; \
		bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ${MAKEFILE_PARENT}.miniconda3 ;\
		. ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ;\
	fi ;\
	if [ ! -d ${BUILD_DIR} ] ;\
	then \
		conda create --prefix=${MAKEFILE_PARENT}.coastal_env python=3.7 -y ;\
	fi

cudem:
	@mkdir -p ${MAKEFILE_PARENT}static/cudem/.cache ;\
	if [[ $$(find ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} -ctime +1 -print) ]] ;\
	then \
		wget -O ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} ${CUDEM_TILE_INDEX_URL} ;\
	fi