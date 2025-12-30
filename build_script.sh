set -x
set -e

module load rocm/6.3.1
module load PrgEnv-gnu
module load craype-accel-amd-gfx90a
ml cmake

# Setup MGARD installation dir

mgard_install_dir=/lustre/orion/cfd164/proj-shared/gongq/Software/MGARD/install-hip-frontier/
adios_install_dir=/lustre/orion/proj-shared/cfd164/gongq/Software/ADIOS2/install-adios/

export CC=amdclang
export CXX=amdclang++

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export GPU_TARGET=gfx90a
export OMPI_CC=hipcc

rm -f build/CMakeCache.txt

cmake -S .  -B ./build \
            -DCMAKE_PREFIX_PATH="${mgard_install_dir};${adios_install_dir}"

cmake --build ./build
