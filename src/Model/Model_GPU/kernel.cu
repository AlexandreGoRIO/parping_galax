#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "operators.cuh"

__global__ void update_velocities(const float4* positions_masses, float4* velocities, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }
    
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int x = 0; x < n_particles - 1; x++) {
		int j = x < i ? x : x + 1;

        float4 diff = positions_masses[j] - positions_masses[i];

        float dij = rnorm3df(diff.x, diff.y, diff.z);
        dij = fminf(dij, 1.0f);
        dij = 10.0f * dij * dij * dij;

        float massj = positions_masses[j].w;

        acc += diff * dij * massj;
	}

	velocities[i] += acc * 2.0f;
}

__global__ void update_positions(float4* positions, const float4* velocities, int n_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }

	positions[i] += velocities[i] * 0.1f;
}

int divup(int a, int b) {
    return (a + b - 1) / b;
}

void update_position_cu(float4* positions_masses, float4* velocities, int n_particles) {
	int nthreads = 128;
	int nblocks = divup(n_particles, nthreads);

    update_velocities<<<nblocks, nthreads>>>(positions_masses, velocities, n_particles);
	update_positions<<<nblocks, nthreads>>>(positions_masses, velocities, n_particles);
}


#endif // GALAX_MODEL_GPU