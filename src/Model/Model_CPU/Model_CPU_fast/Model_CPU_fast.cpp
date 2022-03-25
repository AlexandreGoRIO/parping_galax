#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include "particle.hpp"
#include "Model_CPU_fast.hpp"
#include "octree.hpp"
#include <iostream>
#include "vector_types.h"
#include "vector_functions.h"
//#include <xsimd/xsimd.hpp>
//#include <omp.h>

//namespace xs = xsimd;
//using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
    host_particles.resize(n_particles);
    for (int i = 0; i < n_particles; i++) {
        host_particles[i].position = make_float3(initstate.positionsx[i], initstate.positionsy[i], initstate.positionsz[i]);
        host_particles[i].mass = initstate.masses[i];
        host_particles[i].velocity = make_float3(initstate.velocitiesx[i], initstate.velocitiesy[i], initstate.velocitiesz[i]);
        host_particles[i].id = i;
    }
}

void Model_CPU_fast
::step()
{
    
        // Make octree and reorder particles
    if (host_particles.size()==0){
        std::cerr << "MON DIEU oh"  << std::endl;
    }
    auto octree = make_octree(host_particles, n_particles/16);

    // Sanity checks
    int num_nodes = 0;
    int num_leaves = 0;
    int num_particles = 0;
    float total_mass = 0.0f;
    //std::cout << "coucous"  << std::endl;
    if (octree == nullptr){
        std::cerr << "MON DIEU"  << std::endl;
    }
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
    //std::cout << total_mass << " == " << octree->total_mass << std::endl;
    if (num_particles != n_particles)
        std::cout << "AAAAAAAAAAAAAA" << std::endl;

    //do the calculation

    std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

    
    octree->walk([&](const Node& n) {
        if (n.is_leaf()) {
            //num_leaves++;
            //num_particles += n.particles_end - n.particles_begin;
            for (int i = n.particles_begin; i < n.particles_end; ++i) {
                for (int j = i; j < n.particles_end; j++)
	            {
                    // if(i!=j){
                    //     continue;
                    // }
                    const float diffx = host_particles[j].position.x - host_particles[i].position.x;
                    const float diffy = host_particles[j].position.y - host_particles[i].position.y;
                    const float diffz = host_particles[j].position.z - host_particles[i].position.z;

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                    if (dij < 1.0)
                    {
                        dij = 10.0;
                    }
                    else
                    {
                        dij = std::sqrt(dij);
                        dij = 10.0 / (dij * dij * dij);
                    }

                    accelerationsx[i] += diffx * dij * initstate.masses[j];
                    accelerationsy[i] += diffy * dij * initstate.masses[j];
                    accelerationsz[i] += diffz * dij * initstate.masses[j];
                }
                octree->walk([&](const Node& n2) {
                    if (!n.equals(n2)){
                        if (n2.is_leaf()){

                            const float diffx = n2.center_of_mass.x - host_particles[i].position.x;
                            const float diffy = n2.center_of_mass.y - host_particles[i].position.y;
                            const float diffz = n2.center_of_mass.z - host_particles[i].position.z;

                            float dij = diffx * diffx + diffy * diffy + diffz * diffz;
                            if (dij==0){
                                dij = 0;
                            }

                            else if (dij < 1.0)
                            {
                                dij = 10.0;
                            }
                            else
                            {
                                dij = std::sqrt(dij);
                                dij = 10.0 / (dij * dij * dij);
                            }

                            accelerationsx[i] += diffx * dij * n2.total_mass;
                            accelerationsy[i] += diffy * dij * n2.total_mass;
                            accelerationsz[i] += diffz * dij * n2.total_mass;
                        }
                    }
                });

            }
        }
        //std::cout << "x="<<n.center_of_mass.x << " y= "<<n.center_of_mass.y << std::endl;
    });


	// for (int i = 0; i < n_particles; i++)
	// {
	// 	for (int j = 0; j < n_particles; j++)
	// 	{

    //         const float diffx = host_particles[j].position.x - host_particles[i].position.x;
    //         const float diffy = host_particles[j].position.y - host_particles[i].position.y;
    //         const float diffz = host_particles[j].position.z - host_particles[i].position.z;

    //         float dij = diffx * diffx + diffy * diffy +500
    //         {
    //             dij = 10.0;
    //         }
    //         else
    //         {
    //             dij = std::sqrt(dij);
    //             dij = 10.0 / (dij * dij * dij);
    //         }

    //         accelerationsx[i] += diffx * dij * initstate.masses[j];
    //         accelerationsy[i] += diffy * dij * initstate.masses[j];
    //         accelerationsz[i] += diffz * dij * initstate.masses[j];

        
	// 	}
	// }
    // std::cout << accelerationsy[1] << std::endl;

	for (int i = 0; i < n_particles; i++)
	{
		host_particles[i].velocity.x += accelerationsx[i] * 2.0f;
		host_particles[i].velocity.y += accelerationsy[i] * 2.0f;
		host_particles[i].velocity.z += accelerationsz[i] * 2.0f;
		host_particles[i].position.x += host_particles[i].velocity.x * 0.1f;
		host_particles[i].position.y += host_particles[i].velocity.y * 0.1f;
		host_particles[i].position.z += host_particles[i].velocity.z * 0.1f;
	}
    // De-interlace the positions

	for (const Particle& p: host_particles) {
		particles.x[p.id] = p.position.x;
		particles.y[p.id] = p.position.y;
		particles.z[p.id] = p.position.z;
	}
}


// OMP  version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i ++)
//     {
//     }


// OMP + xsimd version
// #pragma omp parallel for
//     for (int i = 0; i < n_particles; i += b_type::size)
//     {
//         // load registers body i
//         const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
//         const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
//         const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
//               b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
//               b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
//               b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

//         ...
//     }



#endif // GALAX_MODEL_CPU_FAST
