#ifdef GALAX_MODEL_CPU_FAST

#ifndef MODEL_CPU_FAST_HPP_
#define MODEL_CPU_FAST_HPP_

#include "../Model_CPU.hpp"
#include "particle.hpp"

class Model_CPU_fast : public Model_CPU
{
private:
    std::vector<Particle> host_particles;
public:
    Model_CPU_fast(const Initstate& initstate, Particles& particles);

    virtual ~Model_CPU_fast() = default;

    virtual void step();
};
#endif // MODEL_CPU_FAST_HPP_

#endif // GALAX_MODEL_CPU_FAST
