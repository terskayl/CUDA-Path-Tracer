#include "intersections.h"
#include "utilities.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

// Interpolated normals using the nor buffers - can 
// result in black geometry if normals are backwards.
#define SMOOTHSHADING 1

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

// Assume Ray r is pretransformed into local space of the mesh.
__host__ __device__ float bboxIntersectionTest(
    Ray r, glm::vec3 minBounds, glm::vec3 maxBounds)
{
    glm::vec3 center = 0.5f * (minBounds + maxBounds);
    glm::vec3 scale = maxBounds - minBounds;
    glm::mat4 transform = {
        scale.x,  0.0f,     0.0f,     0.0f,
        0.0f,     scale.y,  0.0f,     0.0f,
        0.0f,     0.0f,     scale.z,  0.0f,
        center.x, center.y, center.z, 1.0f
    };
    glm::mat4 inverseTransform = {
        1.0f / scale.x, 0.0f,        0.0f,        0.0f,
        0.0f,        1.0f / scale.y, 0.0f,        0.0f,
        0.0f,        0.0f,        1.0f / scale.z, 0.0f,
       -center.x / scale.x, -center.y / scale.y, -center.z / scale.z, 1.0f
    };
    glm::mat4 invTranspose = {
        1.0f / scale.x, 0.0f,        0.0f,        0.0f,
        0.0f,        1.0f / scale.y, 0.0f,        0.0f,
        0.0f,        0.0f,        1.0f / scale.z, 0.0f,
        0.0f,        0.0f,        0.0f,           1.0f
    };

    Ray q;
    q.origin = multiplyMV(inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        //outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            //outside = false;
        }
        glm::vec3 intersectionPoint = multiplyMV(transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        //normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(
    glm::vec3 p1,
    glm::vec3 p2,
    glm::vec3 p3,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec3& baryWeights,
    bool& notBackface)
{

    assert(abs(glm::length(r.direction)) - 1 < 0.01);
    glm::vec3 v12 = p2 - p1;
    glm::vec3 v13 = p3 - p1;
    // assume CCW winding order
    glm::vec3 currNormal = glm::normalize(glm::cross(v12, v13));

    // Find Ray intersection with plane of triangle
    float dist = glm::dot(p1 - r.origin, currNormal);
    // how much closer the ray move per time t?
    float step = glm::dot(r.direction, currNormal);
    if (abs(step) < 0.0001) {
        return -1; // Ray direction is parallel to the triangle
    }
    notBackface = true;
    if (step > 0) {
        notBackface = false;
    }

    float t = dist / step;

    if (t < 0) {
        return -1; // Plane intersection point is behind the ray origin - hmm. Or backface
    }

    intersectionPoint = r.origin + t * r.direction;

    // Barycentric check to determine if inside triangle.
    // Let planeIntersectionPoint be denoted as s.
    float areaS12 = glm::length(glm::cross(p2 - p1, intersectionPoint - p1));
    float areaS23 = glm::length(glm::cross(p3 - p2, intersectionPoint - p2));
    float areaS31 = glm::length(glm::cross(p1 - p3, intersectionPoint - p3));

    float area123 = glm::length(glm::cross(p2 - p1, p3 - p1));

    baryWeights = glm::vec3(areaS23 / area123, areaS31 / area123, areaS12 / area123);

    float diff = fabs((areaS12 + areaS23 + areaS31) - area123);
    if (diff < 1e-5) {
        normal = currNormal;
        intersectionPoint = r.origin + r.direction * t;
        return t;
    }
    return -1;

}

// TODO
__host__ __device__ float meshIntersectionTestNaive(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent,
    bool& outside)
{
    const Mesh& mesh = geom.mesh;
    float min_t = INFINITY;
    glm::vec3 min_intersect, min_normal;
    glm::vec3 min_baryCoords;
    bool min_outside;
    
    int min_v0 = -1;
    int min_v1 = -1;
    int min_v2 = -1;

    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    assert(abs(glm::length(q.direction) - 1) < 0.01);

    for (int i = 0; i < mesh.indCount / 3.f; ++i) {
        glm::vec3 tmp_intersect, tmp_normal;
        glm::vec3 tmp_baryCoords;
        bool tmp_outside;
        int v0 = mesh.ind[3 * i];
        int v1 = mesh.ind[3 * i + 1];
        int v2 = mesh.ind[3 * i + 2];

        float t = triangleIntersectionTest(
            mesh.pos[v0],
            mesh.pos[v1],
            mesh.pos[v2],
            q, tmp_intersect, tmp_normal, tmp_baryCoords, tmp_outside);

        if (t > 0 && t < min_t) {
            min_t = t;
            min_intersect = tmp_intersect;
            min_normal = tmp_normal;
            min_baryCoords = tmp_baryCoords;
            min_outside = tmp_outside;

            min_v0 = v0;
            min_v1 = v1;
            min_v2 = v2;
        }
    }

    if (min_t == INFINITY) {
        return -1;
    }
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, min_t), 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(min_normal, 0.0f)));
    
    if (geom.mesh.uvCount > 0) {
        uv = min_baryCoords.x * mesh.uv[min_v0] + min_baryCoords.y * mesh.uv[min_v1] + min_baryCoords.z * mesh.uv[min_v2];

        // Calculate tangent and bitangent for normal mapping
        glm::vec3 p10 = mesh.pos[min_v1] - mesh.pos[min_v0];
        glm::vec3 p20 = mesh.pos[min_v2] - mesh.pos[min_v0];

        glm::vec2 uv10 = mesh.uv[min_v1] - mesh.uv[min_v0];
        glm::vec2 uv20 = mesh.uv[min_v2] - mesh.uv[min_v0];

        float r = 1.f / (uv10.x * uv20.y - uv10.y * uv20.x);
        tangent = glm::normalize((p10 * uv20.y - p20 * uv10.y) * r);
        bitangent = glm::normalize((p20 * uv10.x - p10 * uv20.x) * r);
    }
    // Override normals for a mesh. Potential TODO - make this togglable for flat shading.
#if SMOOTHSHADING
    if (geom.mesh.norCount > 0) {
        normal = min_baryCoords.x * mesh.nor[min_v0] + min_baryCoords.y * mesh.nor[min_v1] + min_baryCoords.z * mesh.nor[min_v2];
    }
#endif

    return glm::length(r.origin - intersectionPoint);
}

// Requires BVH TO OPERATE
__host__ __device__ float meshIntersectionTestBVH(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec3& tangent,
    glm::vec3& bitangent,
    bool& outside)
{
    const Mesh mesh = geom.mesh;
    float min_t = INFINITY; 
    glm::vec3 min_intersect, min_normal;
    glm::vec3 min_baryCoords;
    bool min_outside;
    
    int min_v0 = -1;
    int min_v1 = -1;
    int min_v2 = -1;

    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    assert(abs(glm::length(q.direction) - 1) < 0.01);
    
    assert(mesh.numBvhNodes > 0);
    // DFS
    const int MAX_BVH_DEPTH = 15; // Also change in scene.cpp
    unsigned short stack[MAX_BVH_DEPTH];
    stack[0] = 0;
    int stackIdx = 0;

    while (stackIdx > -1) {
        // pop from our stack
        BvhNode& currNode = mesh.bvhNodes[stack[stackIdx]];
        stackIdx--;

        int bbox_t = bboxIntersectionTest(q, currNode.minBounds, currNode.maxBounds);
        if (bbox_t == -1) continue;

        // Nodes should either have two children or none.

        assert((currNode.leftChild <= 0 && currNode.rightChild <= 0) ||
            (currNode.leftChild > 0 && currNode.rightChild > 0));
        // nodes set their child values to 0 when they are a leaf,
        // as index 0 cannot be a child
        if (currNode.leftChild > 0 && currNode.rightChild > 0) {
            stackIdx++;
            stack[stackIdx] = currNode.leftChild;
            stackIdx++;
            stack[stackIdx] = currNode.rightChild;
        }
        else {
            // Leaf node, so we should go through its triangles
            for (int i = 0; i < currNode.trisLength / 3; ++i) {
                glm::vec3 tmp_intersect, tmp_normal;
                glm::vec3 tmp_baryCoords;
                bool tmp_outside;

                int v0 = mesh.indBVH[currNode.trisOffset + 3 * i];
                int v1 = mesh.indBVH[currNode.trisOffset + 3 * i + 1];
                int v2 = mesh.indBVH[currNode.trisOffset + 3 * i + 2];

                float t = triangleIntersectionTest(
                    mesh.pos[v0],
                    mesh.pos[v1],
                    mesh.pos[v2],
                    q, tmp_intersect, tmp_normal, tmp_baryCoords, tmp_outside);

                if (t > 0 && t < min_t) {
                    min_t = t;
                    min_intersect = tmp_intersect;
                    min_normal = tmp_normal;
                    min_baryCoords = tmp_baryCoords;
                    min_outside = tmp_outside;

                    min_v0 = v0;
                    min_v1 = v1;
                    min_v2 = v2;
                }
            }

        }

    }

    if (min_t == INFINITY) {
        return -1;
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, min_t), 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(min_normal, 0.0f)));
    
    if (geom.mesh.uvCount > 0) {
        uv = min_baryCoords.x * mesh.uv[min_v0] + min_baryCoords.y * mesh.uv[min_v1] + min_baryCoords.z * mesh.uv[min_v2];

        // Calculate tangent and bitangent for normal mapping
        glm::vec3 p10 = mesh.pos[min_v1] - mesh.pos[min_v0];
        glm::vec3 p20 = mesh.pos[min_v2] - mesh.pos[min_v0];

        glm::vec2 uv10 = mesh.uv[min_v1] - mesh.uv[min_v0];
        glm::vec2 uv20 = mesh.uv[min_v2] - mesh.uv[min_v0];

        float r = 1.f / (uv10.x * uv20.y - uv10.y * uv20.x);
        tangent = glm::normalize((p10 * uv20.y - p20 * uv10.y) * r);
        bitangent = glm::normalize((p20 * uv10.x - p10 * uv20.x) * r);
    }
    // Override normals for a mesh. Potential TODO - make this togglable for flat shading.
#if SMOOTHSHADING
    if (geom.mesh.norCount > 0) {
        normal = min_baryCoords.x * mesh.nor[min_v0] + min_baryCoords.y * mesh.nor[min_v1] + min_baryCoords.z * mesh.nor[min_v2];
    }
#endif

    return glm::length(r.origin - intersectionPoint);
}