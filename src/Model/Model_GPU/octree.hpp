#ifndef OCTREE_H
#define OCTREE_H

#include <array>
#include <vector>
#include <memory>
#include "vector_types.h"
#include "particle.hpp"

struct Node;

struct Node {
    float3 min; // Min corner of the cube
    float3 max; // Max corner of the cube
    float3 center_of_mass;
    float total_mass;
    int particles_begin; // Index of the first particle that belongs to this node
    int particles_end; // Index of the last particle (not included) that belongs to this node
    std::array<std::unique_ptr<Node>, 8> children; // Pointer to the 8 children

    bool is_leaf() const {
        for (const auto& c: children) {
            if (c != nullptr) {
                return false;
            }
        }
        return true;
    }

    // Apply a function to a node and its descendants recursively
    template<typename Fn>
    void walk(Fn fn) const {
        fn(*this);
        for (const auto& c: children) {
            if (c != nullptr) {
                c->walk(fn);
            }
        }
    }
};

// Sort the list of particles and return the root of the octree
std::unique_ptr<Node> make_octree(
    std::vector<Particle>& particles, int max_pop, int max_depth=128
);

std::unique_ptr<Node> make_octree(
    float3 min, float3 max, int particles_begin, int particles_end, int max_pop, int max_depth, Particle* particles
);

std::array<std::unique_ptr<Node>, 8> make_children(
    float3 min, float3 max, int particles_begin, int particles_end, int max_pop, int max_depth, Particle* particles
);

#endif /* OCTREE_H */