#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

// Simple bounding box intersection tests. Returns -1 if no hit and a positive
// number if it does hit.
__host__ __device__ float bboxIntersectionTest(
    Ray r,
    glm::vec3 minBounds,
    glm::vec3 maxBonuds
);
// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

/**
* Tests intersection between a ray and a triangle in world space. The triangle
* is defined by points p1, p2, and p3.
*  
* @param intersectionPoint  Output parameter for point of intersection
* @param normal             Output parameter for surface normal.
* @param backface           Output if ray intersects frontface of triangle
*                               rather than the backface
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float triangleIntersectionTest(
    glm::vec3 p1,
    glm::vec3 p2,
    glm::vec3 p3,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec3& baryWeights,
    bool& notBackface);

/**
* Tests intersection between a ray and a transformed triangle mesh naively
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float meshIntersectionTestNaive(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent,
    bool& outside);

/**
* Tests intersection between a ray and a transformed triangle mesh with BVH traversal
* 
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float meshIntersectionTestBVH(
    Geom mesh,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent,
    bool& outside);