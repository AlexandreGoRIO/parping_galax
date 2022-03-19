#ifdef GALAX_MODEL_GPU


#include "Model_GPU.hpp"
#include "kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

Model_GPU::Model_GPU(const Initstate& initstate, Particles& particles): Model(initstate, particles) {
	// Init CUDA
	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "error: unable to setup cuda device" << std::endl;
        return;
    }

	// Interlace the positions and velocities into float4 arrays
    int n_particles_padded = div_round_up(n_particles, THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    host_position_mass.resize(n_particles_padded);
    std::vector<float4> temp_host_velocity(n_particles_padded);

    for (int i = 0; i < n_particles_padded; i++) {
        host_position_mass[i] = make_float4(
            initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i], initstate.masses[i]
        );
        temp_host_velocity[i] = make_float4(
            initstate.velocitiesx[i], initstate.velocitiesy[i], initstate.velocitiesz[i], 0.0f
        );
    }
    for (int i = n_particles; i < n_particles_padded; i++) {
        host_position_mass[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        temp_host_velocity[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Create the device buffers
    dev_position_mass = CudaBuffer(host_position_mass);
    dev_velocity = CudaBuffer(temp_host_velocity);
}

void Model_GPU::step() {    
	// Do calculations
    update_positions_cu(dev_position_mass.dev_ptr(), dev_velocity.dev_ptr(), host_position_mass.size());

	// Copy positions to host
    dev_position_mass.retrieve(host_position_mass);

    // De-interlace the positions
	for (int i = 0; i < n_particles; ++i) {
        particles.x[i] = host_position_mass[i].x;
        particles.y[i] = host_position_mass[i].y;
        particles.z[i] = host_position_mass[i].z;
	}
}

void Model_GPU::debug_vectors()
{
    int n = 10;

    std::cout << "posx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_position_mass[i].x << '\t';} std::cout << std::endl;
    std::cout << "posy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_position_mass[i].y << '\t';} std::cout << std::endl;
    std::cout << "posz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_position_mass[i].z << '\t';} std::cout << std::endl;

    // std::cout << "speedx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.x << '\t';} std::cout << std::endl;
    // std::cout << "speedy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.y << '\t';} std::cout << std::endl;
    // std::cout << "speedz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.z << '\t';} std::cout << std::endl;

    // std::cout << "accx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.x << '\t';} std::cout << std::endl;
    // std::cout << "accy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.y << '\t';} std::cout << std::endl;
    // std::cout << "accz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.z << '\t';} std::cout << std::endl;
    std::cout << "\n" << std::flush;
    return;
}

#endif // GALAX_MODEL_GPU
