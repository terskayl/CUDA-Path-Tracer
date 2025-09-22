#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct BvhNode
{
    glm::vec3 maxBounds;
    glm::vec3 minBounds;
    unsigned short leftChild;
    unsigned short rightChild;
    int trisOffset;
    int trisLength;
};

struct Mesh
{
    // Geometry buffers
    glm::vec3* pos = nullptr;
    size_t posCount;
    glm::vec3* nor = nullptr;
    size_t norCount;
    glm::vec2* uv = nullptr;
    size_t uvCount;
    unsigned short* ind = nullptr;
    size_t indCount;

    //BVH
    size_t numBvhNodes;
    // BVH root at bvhNodes[0]
    BvhNode* bvhNodes = nullptr;
    // indices reordered, still form same tris
    unsigned short* indBVH = nullptr;

    __host__ __device__ ~Mesh() {

#ifndef __CUDA_ARCH__
        delete[] pos;
        delete[] nor;
        delete[] uv;
        delete[] ind;

        delete[] bvhNodes;
        delete[] indBVH;
#endif

#ifdef __CUDA_ARCH__ // Need to check these are like being used then ---
        //cudaFree(pos);
        //cudaFree(nor);
        //cudaFree(uv);
        //cudaFree(ind);

        //cudaFree(bvhRoot);
        //cudaFree(indBVH);
#endif
    }

};


struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    Mesh mesh;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 radiance;
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
