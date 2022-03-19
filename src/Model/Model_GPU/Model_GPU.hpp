#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include "kernel.cuh"
#include "utils.hpp"
#include <array>

class Model_GPU : public Model
{
private:
    // CPU-side memory
    std::vector<float4> host_position_mass;

    // GPU-side memory
    CudaBuffer<float4> dev_position_mass;
    CudaBuffer<float4> dev_velocity;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual void step();
    virtual void debug_vectors();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
