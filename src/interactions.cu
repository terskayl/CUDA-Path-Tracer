#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}




__host__ __device__ void sampleAndResolveDiffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    // bsdf evaluates to m.basecolor / pi. But the pdf is 1 / pi,
    // so it cancels out to m.basecolor;
    pathSegment.throughput *= m.baseColor;
}

__host__ __device__ void sampleAndResolveSpecularRefl(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    glm::vec3 dir = pathSegment.ray.direction;
    pathSegment.ray.direction = dir - 2 * glm::dot(dir, normal) * normal;
    pathSegment.throughput *= m.baseColor;
}



__host__ __device__ void sampleAndResolveSpecularTrans(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    glm::vec3 w_i = pathSegment.ray.direction;
    bool isEntering = (dot(normal, w_i) < 0);

    float etaA = 1;
    float etaB = 1.4f;// m.ior;
    float eta;


    if (isEntering) {
        eta = etaA / etaB;
    }
    else {
        eta = etaB / etaA;
        normal = -normal;
    }

    glm::vec3 reflectedDir = glm::refract(w_i, normal, eta);
    // check for total internal rr
    if (isnan(reflectedDir.x) ||
        isnan(reflectedDir.y) ||
        isnan(reflectedDir.z)) {
        pathSegment.remainingBounces = 0;
        return;
    }

    pathSegment.ray.direction = reflectedDir;
    pathSegment.throughput *= m.baseColor;
    return;
}




__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Glass fails at epsilon 1e-5.
    float epsilon = 1e-4;
    
    if (m.roughness == 0) {
        bool isEntering = dot(normal, pathSegment.ray.direction) < 0;
        if (isEntering) {
            pathSegment.ray.origin = intersect + epsilon * -normal;
        }
        else {
            pathSegment.ray.origin = intersect + epsilon * normal;
        }
        sampleAndResolveSpecularTrans(pathSegment, intersect, normal, m, rng);
    }
    else{
        pathSegment.ray.origin = intersect + epsilon * normal;
        sampleAndResolveDiffuse(pathSegment, intersect, normal, m, rng);
    }


    assert(fabs(glm::length(pathSegment.ray.direction) - 1.f) < 0.5f);

}

