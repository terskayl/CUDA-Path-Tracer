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

    const float epsilon = 1e-3;
    pathSegment.ray.origin = intersect + epsilon * normal;
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

    const float epsilon = 1e-3;
    bool isEntering = dot(normal, pathSegment.ray.direction) < 0;
    if (isEntering) {
        pathSegment.ray.origin = intersect + epsilon * normal;
    }
    else {
        pathSegment.ray.origin = intersect + epsilon * -normal;
    }

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

    const float epsilon = 1e-3;
    if (isEntering) {
        pathSegment.ray.origin = intersect + epsilon * -normal;
        eta = etaA / etaB;
    }
    else {
        pathSegment.ray.origin = intersect + epsilon * normal;
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

__host__ __device__ float calculateFrenelDielectric(
    PathSegment& pathSegment,
    glm::vec3 normal) {

    glm::vec3 w_i = pathSegment.ray.direction; // CHECK: negate?
    float cosTheta_i = dot(w_i, normal);

    float etaI = 1.;
    float etaT = 1.4; // m.ior

    if (cosTheta_i < 0) {
        etaI = etaT;
        etaT = 1.;
        cosTheta_i = abs(cosTheta_i);
    }

    if (cosTheta_i > 1.f) {
        cosTheta_i = 1.f;
    }

    float sinTheta_i = sqrt(max(0.f, 1 - cosTheta_i * cosTheta_i));
    float sinTheta_t = etaI / etaT * sinTheta_i;
    if (sinTheta_t >= 1.) {
        return 1.f;
    }
    float cosThetaT = sqrt(max(0.f, 1 - sinTheta_t * sinTheta_t));
    float Rparl = ((etaT * cosTheta_i) - (etaI * cosThetaT)) /
        ((etaT * cosTheta_i) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosTheta_i) - (etaT * cosThetaT)) /
        ((etaI * cosTheta_i) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.f;

}

__host__ __device__ void sampleAndResolveGlass(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {

    float fresnel = calculateFrenelDielectric(pathSegment, normal);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);

    if (random > fresnel) {
        sampleAndResolveSpecularTrans(pathSegment, intersect, normal, m, rng);
    }
    else {
        sampleAndResolveSpecularRefl(pathSegment, intersect, normal, m, rng);
    }

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
    float epsilon = 1e-3;
    
    if (m.roughness == 0) {
        sampleAndResolveGlass(pathSegment, intersect, normal, m, rng);
        //float fresnel = calculateFrenelDielectric(pathSegment, normal);
        //pathSegment.radiance = fresnel * glm::vec3(1, 0, 0) + (1 - fresnel) * glm::vec3(0, 0, 1);
        //if (fresnel > 0.01f && fresnel < 0.1f) {
        //    pathSegment.radiance = glm::vec3(0, 1, 0);
        //}
        //pathSegment.radiance = glm::vec3(fresnel);
        //pathSegment.throughput = fresnel * glm::vec3(100, 0, 0) + (1 - fresnel) * glm::vec3(0, 100, 0);
        //pathSegment.remainingBounces = 0;
        //sampleAndResolveSpecularTrans(pathSegment, intersect, normal, m, rng);
    }
    else{
        pathSegment.ray.origin = intersect + epsilon * normal;
        sampleAndResolveDiffuse(pathSegment, intersect, normal, m, rng);
    }


    assert(fabs(glm::length(pathSegment.ray.direction) - 1.f) < 0.5f);

}

