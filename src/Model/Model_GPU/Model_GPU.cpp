#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"
#include "octree.hpp"


Model_GPU::Model_GPU(const Initstate& initstate, Particles& particles): Model(initstate, particles) {
	// Init CUDA
	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "error: unable to setup cuda device" << std::endl;
        return;
    }

	// Interlace the positions and velocities into float4 arrays
    host_particles.resize(n_particles);
    for (int i = 0; i < n_particles; i++) {
        host_particles[i].position = make_float3(initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i]);
        host_particles[i].mass = initstate.masses[i];
        host_particles[i].velocity = make_float3(initstate.velocitiesx[i], initstate.velocitiesy[i], initstate.velocitiesz[i]);
        host_particles[i].id = i;
    }

    // Create the device buffers
    dev_particles = CudaBuffer(host_particles);
}

void Model_GPU::step() {
    // Make octree and reorder particles
    auto octree = make_octree(host_particles, 2048);
    dev_particles.send(host_particles);

    // Sanity checks
    int num_nodes = 0;
    int num_leaves = 0;
    int num_particles = 0;
    float total_mass = 0.0f;
    octree->walk([&](const Node& n) {
        num_nodes++;
        if (n.is_leaf()) {
            num_leaves++;
            num_particles += n.particles_end - n.particles_begin;
            for (int i = n.particles_begin; i < n.particles_end; ++i) {
                total_mass += host_particles[i].mass;
            }
        }
    });
    // std::cout << total_mass << " == " << octree->total_mass << std::endl;
    // if (num_particles != n_particles)
    //     std::cout << "AAAAAAAAAAAAAA" << std::endl;

	// Do calculations
	update_position_cu((ParticleDev*)dev_particles.dev_ptr(), n_particles);
	if (cudaDeviceSynchronize() != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;

	// Copy positions to host
    dev_particles.retrieve(host_particles);

    // De-interlace the positions
	for (const Particle& p: host_particles) {
		particles.x[p.id] = p.position.x;
		particles.y[p.id] = p.position.y;
		particles.z[p.id] = p.position.z;
	}
}

void Model_GPU::debug_vectors()
{
    int n = 10;

    std::cout << "posx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].position.x << '\t';} std::cout << std::endl;
    std::cout << "posy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].position.y << '\t';} std::cout << std::endl;
    std::cout << "posz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].position.z << '\t';} std::cout << std::endl;

    std::cout << "speedx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.x << '\t';} std::cout << std::endl;
    std::cout << "speedy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.y << '\t';} std::cout << std::endl;
    std::cout << "speedz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].velocity.z << '\t';} std::cout << std::endl;

    // std::cout << "accx = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.x << '\t';} std::cout << std::endl;
    // std::cout << "accy = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.y << '\t';} std::cout << std::endl;
    // std::cout << "accz = " << ' '; for (int i=0; i < n; i++) {std::cout << host_particles[i].acceleration.z << '\t';} std::cout << std::endl;
    std::cout << "\n" << std::flush;
    return;
}

#endif // GALAX_MODEL_GPU
