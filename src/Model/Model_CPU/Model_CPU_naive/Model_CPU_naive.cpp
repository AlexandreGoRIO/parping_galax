#include <cmath>

#include "Model_CPU_naive.hpp"
#include <iostream>

Model_CPU_naive
::Model_CPU_naive(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_naive
::step()
{
	std::fill(accelerationsx.begin(), accelerationsx.end(), 0);
	std::fill(accelerationsy.begin(), accelerationsy.end(), 0);
	std::fill(accelerationsz.begin(), accelerationsz.end(), 0);

	for (int i = 0; i < n_particles; i++)
	{
		for (int j = 0; j < n_particles; j++)
		{
			if(i != j)
			{
				const float diffx = particles.x[j] - particles.x[i];
				const float diffy = particles.y[j] - particles.y[i];
				const float diffz = particles.z[j] - particles.z[i];

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
		}
	}

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}
}

void Model_CPU_naive
::debug_vectors()
{
    int n = 10;

    std::cout << "posx = " << ' '; for (int i=0; i < n; i++) {std::cout << particles.x[i] << '\t';} std::cout << std::endl;
    std::cout << "posy = " << ' '; for (int i=0; i < n; i++) {std::cout << particles.y[i] << '\t';} std::cout << std::endl;
    std::cout << "posz = " << ' '; for (int i=0; i < n; i++) {std::cout << particles.z[i] << '\t';} std::cout << std::endl;

    std::cout << "speedx = " << ' '; for (int i=0; i < n; i++) {std::cout << velocitiesx[i] << '\t';} std::cout << std::endl;
    std::cout << "speedy = " << ' '; for (int i=0; i < n; i++) {std::cout << velocitiesy[i] << '\t';} std::cout << std::endl;
    std::cout << "speedz = " << ' '; for (int i=0; i < n; i++) {std::cout << velocitiesz[i] << '\t';} std::cout << std::endl;

    std::cout << "accx = " << ' '; for (int i=0; i < n; i++) {std::cout << accelerationsx[i] << '\t';} std::cout << std::endl;
    std::cout << "accy = " << ' '; for (int i=0; i < n; i++) {std::cout << accelerationsy[i] << '\t';} std::cout << std::endl;
    std::cout << "accz = " << ' '; for (int i=0; i < n; i++) {std::cout << accelerationsz[i] << '\t';} std::cout << std::endl;
    std::cout << "\n" << std::flush;
    return;
}