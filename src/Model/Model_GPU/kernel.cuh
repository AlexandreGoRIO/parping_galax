#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

void update_position_cu(float4* positions_masses, float4* velocities, int n_particles);

#endif

#endif // GALAX_MODEL_GPU
