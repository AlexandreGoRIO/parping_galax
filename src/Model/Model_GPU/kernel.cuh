#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>
#include "particle.hpp"

void update_position_cu(ParticleDev* particles, int n_particles);

#endif

#endif // GALAX_MODEL_GPU
