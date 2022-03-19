#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "utils.hpp"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

constexpr int THREADS_PER_BLOCK = 128;

void update_positions_cu(float4* position_mass_pack, float4* velocity, int n_particles);

#endif
#endif // GALAX_MODEL_GPU
