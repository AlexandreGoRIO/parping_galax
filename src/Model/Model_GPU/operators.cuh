#include "cuda.h"
#include "vector_types.h"

// --------------- Float 4 operations, but with the 4th component ignored ---------------

inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline __device__ __host__ float4 operator*(float4 a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

inline __device__ __host__ float4& operator+=(float4& a, float4 b) {
    a = a + b;
    return a;
}

inline __device__ __host__ float4& operator-=(float4& a, float4 b) {
    a = a - b;
    return a;
}

inline __device__ __host__ float4& operator*=(float4& a, float b) {
    a = a * b;
    return a;
}

// --------------- Float 3 operations ---------------

inline float3 operator+(float3 a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline float3 operator-(float3 a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline float3 operator*(float3 a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

inline float3& operator+=(float3& a, float3 b) {
    a = a + b;
    return a;
}

inline float3& operator-=(float3& a, float3 b) {
    a = a - b;
    return a;
}

inline float3& operator*=(float3& a, float b) {
    a = a * b;
    return a;
}