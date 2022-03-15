#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"
#include "utils.hpp"

Model_GPU::Model_GPU(const Initstate& initstate, Particles& particles): Model(initstate, particles) {
	// Init CUDA
	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "error: unable to setup cuda device" << std::endl;
        return;
    }

	// Interlace the positions and velocities into float4 arrays
	host_pos_mass = std::vector<float4>(n_particles);
    auto temp_host_vel = std::vector<float4>(n_particles);
    for (int i = 0; i < n_particles; i++) {
		host_pos_mass[i] = make_float4(initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i], initstate.masses[i]);
		temp_host_vel[i] = make_float4(initstate.velocitiesx[i], initstate.velocitiesy[i], initstate.velocitiesz[i], 0.0);
	}

    // Create the device buffers
    dev_pos_mass = CudaBuffer(host_pos_mass);
    dev_vel = CudaBuffer(temp_host_vel);
}

void Model_GPU::step() {
	// Do calculations
	update_position_cu(dev_pos_mass.dev_ptr(), dev_vel.dev_ptr(), n_particles);
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;

	// Copy positions to host
	dev_pos_mass.retrieve(host_pos_mass);

    // De-interlace the positions
	for (int i = 0; i < n_particles; i++) {
		particles.x[i] = host_pos_mass[i].x;
		particles.y[i] = host_pos_mass[i].y;
		particles.z[i] = host_pos_mass[i].z;
	}
}

#endif // GALAX_MODEL_GPU
