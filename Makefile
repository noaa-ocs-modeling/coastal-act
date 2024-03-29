SHELL=/bin/bash

# public
ADCIRC_FORK=adcirc
ADCIRC_BRANCH=master
CUDEM_TILE_INDEX_FILENAME=tileindex_NCEI_ninth_Topobathy_2014.zip
CUDEM_TILE_INDEX_URL:=https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/${CUDEM_TILE_INDEX_FILENAME}


# private
MAKEFILE_PATH:=$(abspath $(lastword $(MAKEFILE_LIST)))
MAKEFILE_PARENT:=$(dir $(MAKEFILE_PATH))
BUILD_DIR:=${MAKEFILE_PARENT}.coastal_env


default: install


clean:
	rm -rf ${BUILD_DIR}
	rm -rf ${MAKEFILE_PARENT}src/adcirc-cg
	rm -rf ${MAKEFILE_PARENT}src/adcircpy
	rm -rf ${MAKEFILE_PARENT}.miniconda3
	rm -rf ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME}
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/fort.15
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/fort.26
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/Model_120m_Combinedv22_Storm.13
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/swaninit
	rm -rf ${MAKEFILE_PARENT}static/Mesh_120m/v22/fort.15


install: coastal adcircpy adcirc mesh_120m_v22_deploy mesh_prusvi_vdatum_deploy cudem adcirc


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
	@set -e ;\
	if [ ! -f ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ] ;\
	then \
		if [[ -n $$CONDA_INSTALL_PREFIX ]] ;\
		then \
			ln -sf $$CONDA_INSTALL_PREFIX ${MAKEFILE_PARENT}.miniconda3 ;\
		else \
			wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
				-O /tmp/Miniconda3-latest-Linux-x86_64.sh;\
			bash /tmp/Miniconda3-latest-Linux-x86_64.sh -u -b -p ${MAKEFILE_PARENT}.miniconda3;\
			rm -rf /tmp/Miniconda3-latest-Linux-x86_64.sh;\
		fi; \
		. ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ;\
	fi ;\
	if [ ! -d ${BUILD_DIR} ] ;\
	then \
		conda create --prefix=${MAKEFILE_PARENT}.coastal_env python=3.7 -y ;\
	fi

cudem:
	@mkdir -p ${MAKEFILE_PARENT}static/cudem/.cache ;\
	if [ ! -f ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} ] ;\
	then \
		wget -O ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} ${CUDEM_TILE_INDEX_URL} ;\
	else \
		if [[ $$(find ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} -ctime +1 -print) ]] ;\
		then \
			wget -O ${MAKEFILE_PARENT}static/cudem/${CUDEM_TILE_INDEX_FILENAME} ${CUDEM_TILE_INDEX_URL} ;\
		fi ;\
	fi

mesh_120m_v22_deploy:
	@if [ ! -f ${MAKEFILE_PARENT}static/Mesh_120m/v22/Model_120m_Combinedv22_Storm.14 ];\
	then \
		cd ${MAKEFILE_PARENT}static;\
		wget https://www.dropbox.com/s/fvtubp13aqt9gjq/Mesh_120m_v22.tar.gz;\
		tar xvf Mesh_120m_v22.tar.gz;\
		rm -rf Mesh_120m_v22.tar.gz;\
	fi

mesh_prusvi_vdatum_deploy:
	@if [ ! -f ${MAKEFILE_PARENT}static/Puerto\ Rico\ VDatum/fort.14 ];\
	then \
		cd ${MAKEFILE_PARENT}static;\
		wget https://www.dropbox.com/s/rhsfetthytm23fu/PuertoRicoVDatum.tar.gz;\
		tar xvf PuertoRicoVDatum.tar.gz;\
		rm -rf PuertoRicoVDatum.tar.gz;\
	fi


adcircpy:
	@. ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${MAKEFILE_PARENT}.coastal_env ;\
	cd ${MAKEFILE_PARENT}src;\
	git clone https://github.com/JaimeCalzadaNOAA/adcircpy.git;\
	cd adcircpy;\
	python ./setup.py install

coastal-develop: coastal
	@. ${MAKEFILE_PARENT}.miniconda3/etc/profile.d/conda.sh ;\
	conda activate ${MAKEFILE_PARENT}.coastal_env ;\
	cd ${MAKEFILE_PARENT}src/coastal ;\
	python ./setup.py develop


noaa-rdhpc-hera:
	@make -e CONDA_INSTALL_PREFIX=/contrib/miniconda3/4.5.12  --no-print-directory


noaa-rdhpc-orion:
	@make --no-print-directory