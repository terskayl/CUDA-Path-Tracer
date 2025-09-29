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
    glm::vec3 color,
    thrust::default_random_engine& rng) {

    const float epsilon = 1e-3;
    pathSegment.ray.origin = intersect + epsilon * normal;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    // bsdf evaluates to m.basecolor / pi. But the pdf is 1 / pi,
    // so it cancels out to m.basecolor;
    pathSegment.throughput *= color;
}

__host__ __device__ void sampleAndResolveSpecularRefl(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    glm::vec3 color,
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
    glm::vec3 color,
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

    glm::vec3 w_i = pathSegment.ray.direction;
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
    float sinTheta_t = etaT / etaI * sinTheta_i;
    //if (sinTheta_t > 1.0) {
    //    return 1.f;
    //}
    float cosThetaT = sqrt(max(0.f, 1 - sinTheta_t * sinTheta_t));
    float Rparl = ((etaT * cosTheta_i) - (etaI * cosThetaT)) /+
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
    glm::vec3 color,
    thrust::default_random_engine& rng) {

    float fresnel = calculateFrenelDielectric(pathSegment, normal);
    thrust::uniform_real_distribution<float> u01(0, 1);
    float random = u01(rng);

    if (random > fresnel) {
        sampleAndResolveSpecularTrans(pathSegment, intersect, normal, m, color, rng);
    }
    else {
        sampleAndResolveSpecularRefl(pathSegment, intersect, normal, m, color, rng);
    }

}


__host__ __device__ glm::vec3 calculateFresnelConductor(
    PathSegment& pathSegment,
    glm::vec3 normal) {

    glm::vec3 w_i = pathSegment.ray.direction;
    float cosTheta_i = dot(w_i, normal);
    // We will hard-code the indices of refraction to be
    // those of GOLD https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_03_Fresnel.pdf
    //float eta = 0.37;
    //float k = 2.82;
    // Eta and k values for r,g,b, at 630nm, 532nm, and 465nm.
    // from https://refractiveindex.info/?shelf=main&book=Au&page=Johnson
    glm::vec3 eta = glm::vec3(0.188, 0.543, 1.332);
    glm::vec3 k = glm::vec3(3.403, 2.231, 1.869);
    cosTheta_i = glm::clamp(cosTheta_i, -1.f, 1.f);
    if (cosTheta_i < 0) {
        cosTheta_i = abs(cosTheta_i);
    }

    // Implementation from https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_03_Fresnel.pdf slide 25
    glm::vec3 etaK2 = eta * eta * k * k;
    glm::vec3 Rparl = ((etaK2 * cosTheta_i * cosTheta_i) - 2.f * (eta * cosTheta_i) + glm::vec3(1.)) /
        ((etaK2 * cosTheta_i * cosTheta_i) + 2.f * (eta * cosTheta_i) + glm::vec3(1.));
    glm::vec3 Rperp = ((etaK2)+glm::vec3(cosTheta_i * cosTheta_i) - 2.f * (eta * cosTheta_i)) /
        ((etaK2)+glm::vec3(cosTheta_i * cosTheta_i) + 2.f * (eta * cosTheta_i));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2.f);
}

__host__ __device__ void sampleAndResolveMetal(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    glm::vec3 color,
    thrust::default_random_engine& rng) {

    glm::vec3 fresnel = calculateFresnelConductor(pathSegment, normal);
    
    sampleAndResolveSpecularRefl(pathSegment, intersect, normal, m, color, rng);
    // Double because there is no chance of transmission
    pathSegment.throughput *= 2.f * fresnel;
}

// Scatter ray cannot be called on the host as it samples textures.
__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    DeviceTexture* texArr,
    glm::vec2 uv,
    glm::vec3 tangent,
    glm::vec3 bitangent,
    thrust::default_random_engine& rng)
{
    // Glass fails at epsilon 1e-5.
    float epsilon = 1e-3;
    // Add normal map to normals if applicable.
    if (m.normalTexture != -1) {
        cudaTextureObject_t normalTexture = texArr[m.normalTexture].texHandle;

        float4 normalMapReading = tex2D<float4>(normalTexture, uv.x, uv.y);
        glm::vec3 normalDiff = (glm::vec3(normalMapReading.x, normalMapReading.y, normalMapReading.z) * 2.f) - glm::vec3(-1.f);

        normalDiff.x *= m.normalTextureScale;
        normalDiff.y *= m.normalTextureScale;
        normalDiff = glm::normalize(normalDiff);

        normal = normal * normalDiff.b, tangent * normalDiff.r + bitangent * normalDiff.g;
    }

    glm::vec3 color = m.baseColor;
    if (m.baseColorTexture != -1) {
        cudaTextureObject_t colorTexture = texArr[m.baseColorTexture].texHandle;

        float4 colorMapReading = tex2D<float4>(colorTexture, uv.x, uv.y);
        color = glm::vec3(
            colorMapReading.x,
            colorMapReading.y,
            colorMapReading.z);
    }
    //color = normal;

    if (m.roughness == 0) {
        sampleAndResolveMetal(pathSegment, intersect, normal, m, color, rng);
        //float fresnel = calculateFrenelDielectric(pathSegment, normal);
        //pathSegment.radiance = fresnel * glm::vec3(1, 0, 0) + (1 - fresnel) * glm::vec3(0, 0, 1);
        //if (fresnel > 0.01f && fresnel < 0.1f) {
        //    pathSegment.radiance = glm::vec3(0, 1, 0);
        //}
        //pathSegment.radiance = glm::vec3(fresnel);
        //pathSegment.throughput = fresnel * glm::vec3(100, 0, 0) + (1 - fresnel) * glm::vec3(0, 100, 0);
        //pathSegment.remainingBounces = 0;
        //sampleAndResolveSpecularTrans(pathSegment, intersect, normal, m, color, rng);
    }
    else{
        pathSegment.ray.origin = intersect + epsilon * normal;
        sampleAndResolveDiffuse(pathSegment, intersect, normal, m, color, rng);
    }


    assert(fabs(glm::length(pathSegment.ray.direction) - 1.f) < 0.5f);

}

