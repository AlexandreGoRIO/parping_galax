#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector_types.h>

struct Particle {
    float3 position;
    float mass;
    float3 velocity;
    int id;
};

struct ParticleDev {
    float4 position_mass_pack;
    float4 velocity_id_pack;
};

static_assert(sizeof(Particle) == sizeof(ParticleDev));

#endif /* PARTICLE_H */