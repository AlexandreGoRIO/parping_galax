#include "cuda.h"


inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    float4 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

inline __device__ __host__ float4 operator*(float4 a, float b) {
    float4 c;
    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;
    return c;
}

inline __device__ __host__ float4 operator+=(float4& a, float4 b) {
    a = a + b;
    return a;
}

inline __device__ __host__ float4 operator-=(float4& a, float4 b) {
    a = a - b;
    return a;
}

inline __device__ __host__ float4 operator*=(float4& a, float b) {
    a = a * b;
    return a;
}

inline __device__ __host__ float4 max(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), 0.0);
}

inline __device__ __host__ float4 min(float4 a, const float4 b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), 0.0);
}