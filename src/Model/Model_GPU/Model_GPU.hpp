#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"
#include "utils.hpp"

class Model_GPU : public Model
{
private:
	std::vector<float4> host_pos;

	CudaBuffer<float4> dev_pos;
    CudaBuffer<float4> dev_vel;
    CudaBuffer<float4> dev_acc;
    CudaBuffer<float> dev_mass;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
