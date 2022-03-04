#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__device__ __host__ float3 operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ __host__ float3 operator-=(float3& a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__device__ __host__ float3 operator*=(float3& a, const float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}

__device__ __host__ float3 operator+(float3 a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ __host__ float3 operator-(float3 a, const float3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

__device__ __host__ float3 operator*(float3 a, const float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}


__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	accelerationsGPU[i]=make_float3(0.0, 0.0, 0.0);
	for (int j=0;j < n_particles; j++)
	{
		if(i != j)
		{
			const float3 diff = positionsGPU[j] - positionsGPU[i];

			float dij = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

			if (dij < 1.0)
			{
				dij = 10.0;
			}
			else
			{
				dij = sqrt(dij);
				dij = 10.0 / (dij * dij * dij);
			}

			accelerationsGPU[i] += diff * dij * massesGPU[j];
		}
	}
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	velocitiesGPU[i]+=accelerationsGPU[i]*2.0f;
	positionsGPU[i]+=velocitiesGPU[i]*0.1f;

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU);
}


#endif // GALAX_MODEL_GPU