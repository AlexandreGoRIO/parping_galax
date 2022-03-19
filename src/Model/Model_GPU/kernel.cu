#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#include "operators.cuh"

// Compute one interaction between particle i and particle j and return the acceleration of i
__device__ float4 compute_interaction(float4 position_mass_pack_i, float4 position_mass_pack_j) {
    // Relative position
    float4 diff = position_mass_pack_j - position_mass_pack_i;

    // 1 / sqrt(distance)^3
    float dij = rsqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    dij = fminf(dij, 1.0f);
    dij = 10.0f * dij * dij * dij;

    float massj = position_mass_pack_j.w;
    return diff * (dij * massj);
}

// Updates the velocities of all the particles
__global__ void update_velocities(float4* position_mass_pack, float4* velocity, int n_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 position_mass_pack_i = position_mass_pack[i];

    // Compute the total acceleration
	for (int j = 0; j < n_particles; j++) {
        float4 position_mass_pack_j = position_mass_pack[j];
        acc += compute_interaction(position_mass_pack_i, position_mass_pack_j);
	}

    velocity[i] += acc * 2.0f;
}

// Updates the velocities of all the particles, but slightly faster
// https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
__global__ void update_velocities_tiled(float4* position_mass_pack, float4* velocity, int n_particles) {
    __shared__ float4 shared_particles_j[THREADS_PER_BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 position_mass_pack_i = position_mass_pack[i];

    // Compute the total acceleration
    int j_begin = 0;
	for (int tile = 0; tile < n_particles / THREADS_PER_BLOCK; tile++) {
        // Copy a group of particles into shared memory
        shared_particles_j[threadIdx.x] = position_mass_pack[j_begin + threadIdx.x];
        __syncthreads();

        // Compute the interactions with this group of particles
        for (int x = 0; x < THREADS_PER_BLOCK; x++) {
            float4 position_mass_pack_j = shared_particles_j[x];
            acc += compute_interaction(position_mass_pack_i, position_mass_pack_j);
        }
        __syncthreads();
        
        j_begin += THREADS_PER_BLOCK;
    }

    velocity[i] += acc * 2.0f;
}

// Add the velocities of the particles to their position
__global__ void update_positions(float4* position_mass_pack, float4* velocity, int n_particles) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	position_mass_pack[i] += velocity[i] * 0.1f;
}

// Do one integration step
void update_positions_cu(float4* position_mass_pack, float4* velocity, int n_particles) {
    if (n_particles % THREADS_PER_BLOCK != 0) {
        std::cout << "error: n_particles must be a multiple of THREADS_PER_BLOCK" << std::endl;
    }

	int nblocks = n_particles / THREADS_PER_BLOCK;
    update_velocities_tiled<<<nblocks, THREADS_PER_BLOCK>>>(position_mass_pack, velocity, n_particles);
	update_positions<<<nblocks, THREADS_PER_BLOCK>>>(position_mass_pack, velocity, n_particles);
    cudaDeviceSynchronize();
}

#endif // GALAX_MODEL_GPU