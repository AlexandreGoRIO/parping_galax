#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"
#include "utils.hpp"
#include "particle.hpp"

class Model_GPU : public Model
{
private:
    std::vector<Particle> host_particles;
    CudaBuffer<Particle> dev_particles;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual void step();
    virtual void debug_vectors();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
