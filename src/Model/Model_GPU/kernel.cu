#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "operators.cuh"

__global__ void update_velocities(ParticleDev* particles, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }
    
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int x = 0; x < n_particles - 1; x++) {
		int j = x < i ? x : x + 1;

        float4 diff = particles[j].position_mass_pack - particles[i].position_mass_pack;

        float dij = rnorm3df(diff.x, diff.y, diff.z);
        dij = fminf(dij, 1.0f);
        dij = 10.0f * dij * dij * dij;

        float massj = particles[j].position_mass_pack.w;

        acc += diff * dij * massj;
	}

	particles[i].velocity_id_pack += acc * 2.0f;
}

__global__ void update_positions(ParticleDev* particles, int n_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_particles) {
        return;
    }

	particles[i].position_mass_pack += particles[i].velocity_id_pack * 0.1f;
}

int divup(int a, int b) {
    return (a + b - 1) / b;
}

void update_position_cu(ParticleDev* particles, int n_particles) {
	int nthreads = 128;
	int nblocks = divup(n_particles, nthreads);

    update_velocities<<<nblocks, nthreads>>>(particles, n_particles);
	update_positions<<<nblocks, nthreads>>>(particles, n_particles);
}


#endif // GALAX_MODEL_GPU