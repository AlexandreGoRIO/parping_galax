#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "operators.cuh"

__global__ void compute_acc(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }

	accelerationsGPU[i] = make_float4(0.0, 0.0, 0.0, 0.0);
    
	for (int j = 0; j < n_particles; j++) {
		if(i == j) {
            continue;
		}

        float4 diff = positionsGPU[j] - positionsGPU[i];

        float dij = rnorm3df(diff.x, diff.y, diff.z);
        dij = fminf(dij, 1.0);
        dij = 10.0 * dij * dij * dij;

        accelerationsGPU[i] += diff * dij * massesGPU[j];
	}
}

__global__ void integrate(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, int n_particles) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }

	velocitiesGPU[i] += accelerationsGPU[i] * 2.0f;
	positionsGPU[i] += velocitiesGPU[i] * 0.1f;
}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, float4* accelerationsGPU, float* massesGPU, int n_particles) {
	int nthreads = 128;
	int nblocks = (n_particles + (nthreads - 1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	integrate<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU