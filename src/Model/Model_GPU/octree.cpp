#include "octree.hpp"
#include <algorithm>
#include <vector_functions.h>
#include <iostream>
#include "operators.cuh"

// Get the min and max corners of a cube that contains all the particles
std::pair<float3, float3> get_min_max(const std::vector<Particle>& particles) {
    float inf = std::numeric_limits<float>::infinity();
    float3 min = {inf, inf, inf};
    float3 max = {-inf, -inf, -inf};

    for (const Particle& p: particles) {
        min = {std::min(min.x, p.position.x), std::min(min.y, p.position.y), std::min(min.z, p.position.z)};
        max = {std::max(max.x, p.position.x), std::max(max.y, p.position.y), std::max(max.z, p.position.z)};
    }

    return {min, max};
}

// Compute the total mass and center of mass of a node
std::pair<float3, float> compute_center_of_mass(const std::unique_ptr<Node>& n, const Particle* particles) {
    float total_mass = 0.0f;
    float3 center_of_mass_sum = {0.0f, 0.0f, 0.0f};

    if (n->is_leaf()) {
        for (int i = n->particles_begin; i < n->particles_end; ++i) {
            total_mass += particles[i].mass;
            center_of_mass_sum += particles[i].position * particles[i].mass;
        }
    } else {
        for (const auto& c: n->children) {
            if (c != nullptr) {
                total_mass += c->total_mass;
                center_of_mass_sum += c->center_of_mass * c->total_mass;
            }
        }
    }

    return {center_of_mass_sum * (1.0f / total_mass), total_mass};
}

std::unique_ptr<Node> make_octree(std::vector<Particle>& particles, int max_pop, int max_depth) {
    auto [min, max] = get_min_max(particles);
    return make_octree(min, max, 0, particles.size(), max_pop, max_depth, particles.data());
}

std::unique_ptr<Node> make_octree(
    float3 min, float3 max, int particles_begin, int particles_end, int max_pop, int max_depth, Particle* particles
) {
    if (particles_begin == particles_end) {
        return nullptr;
    }

    if (max_depth == 0) {
        std::cout << "error (probably): reached the maximum depth of the octree" << std::endl;
    }

    std::unique_ptr<Node> n(new Node);
    n->min = min;
    n->max = max;
    n->particles_begin = particles_begin;
    n->particles_end = particles_end;

    // Split into 8 octants and recurse if this node is overcrowded
    if (max_depth != 0 && particles_end - particles_begin > max_pop) {
        n->children = make_children(min, max, particles_begin, particles_end, max_pop, max_depth - 1, particles);
    }

    // Set the total mass and center of mass
    auto [center_of_mass, total_mass] = compute_center_of_mass(n, particles);
    n->total_mass = total_mass;
    n->center_of_mass = center_of_mass;

    return n;
}

std::array<std::unique_ptr<Node>, 8> make_children(
    float3 min, float3 max, int particles_begin, int particles_end, int max_pop, int max_depth, Particle* particles
) {
    float3 ctr = (min + max) * 0.5f;
    
    auto partition_x = [&](const Particle& p) { return p.position.x < ctr.x; };
    auto partition_y = [&](const Particle& p) { return p.position.y < ctr.y; };
    auto partition_z = [&](const Particle& p) { return p.position.z < ctr.z; };

    // Sort the particles in 8 octants
    int split_0 = particles_begin;
    int split_7 = particles_end;
    int split_34 = std::partition(&particles[split_0], &particles[split_7], partition_x) - particles;
    int split_12 = std::partition(&particles[split_0], &particles[split_34], partition_y) - particles;
    int split_01 = std::partition(&particles[split_0], &particles[split_12], partition_z) - particles;
    int split_23 = std::partition(&particles[split_12], &particles[split_34], partition_z) - particles;
    int split_56 = std::partition(&particles[split_34], &particles[split_7], partition_y) - particles;
    int split_45 = std::partition(&particles[split_34], &particles[split_56], partition_z) - particles;
    int split_67 = std::partition(&particles[split_56], &particles[split_7], partition_z) - particles;

    // Create the 8 children
    return {
        make_octree({min.x, min.y, min.z}, {ctr.x, ctr.y, ctr.z}, split_0, split_01, max_pop, max_depth, particles), // -X, -Y, -Z
        make_octree({min.x, min.y, ctr.z}, {ctr.x, ctr.y, max.z}, split_01, split_12, max_pop, max_depth, particles), // -X, -Y, +Z
        make_octree({min.x, ctr.y, min.z}, {ctr.x, max.y, ctr.z}, split_12, split_23, max_pop, max_depth, particles), // -X, +Y, -Z
        make_octree({min.x, ctr.y, ctr.z}, {ctr.x, max.y, max.z}, split_23, split_34, max_pop, max_depth, particles), // -X, +Y, +Z
        make_octree({ctr.x, min.y, min.z}, {max.x, ctr.y, ctr.z}, split_34, split_45, max_pop, max_depth, particles), // +X, -Y, -Z
        make_octree({ctr.x, min.y, ctr.z}, {max.x, ctr.y, max.z}, split_45, split_56, max_pop, max_depth, particles), // +X, -Y, +Z
        make_octree({ctr.x, ctr.y, min.z}, {max.x, max.y, ctr.z}, split_56, split_67, max_pop, max_depth, particles), // +X, +Y, -Z
        make_octree({ctr.x, ctr.y, ctr.z}, {max.x, max.y, max.z}, split_67, split_7, max_pop, max_depth, particles) // +X, +Y, +Z
    };
}
