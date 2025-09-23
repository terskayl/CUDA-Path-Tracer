#include "intersections.h"

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
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ float triangleIntersectionTest(
    glm::vec3 p1,
    glm::vec3 p2,
    glm::vec3 p3,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& notBackface)
{

    assert(abs(glm::length(r.direction)) - 1 < 0.01);
    glm::vec3 v12 = p2 - p1;
    glm::vec3 v13 = p3 - p1;
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
        currNormal *= -1; // I presume we want normal facing towards ray anyways
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
    bool& outside)
{
    const Mesh& mesh = geom.mesh;
    float min_t = INFINITY;
    glm::vec3 min_intersect, min_normal;
    bool min_outside;

    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    assert(abs(glm::length(q.direction) - 1) < 0.01);

    for (int i = 0; i < mesh.indCount / 3.f; ++i) {
        glm::vec3 tmp_intersect, tmp_normal;
        bool tmp_outside;
        float t = triangleIntersectionTest(
            mesh.pos[mesh.ind[3 * i]],
            mesh.pos[mesh.ind[3 * i + 1]],
            mesh.pos[mesh.ind[3 * i + 2]],
            q, tmp_intersect, tmp_normal, tmp_outside);

        if (t > 0 && t < min_t) {
            min_t = t;
            min_intersect = tmp_intersect;
            min_normal = tmp_normal;
            min_outside = tmp_outside;
        }
    }

    if (min_t == INFINITY) {
        return -1;
    }
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, min_t), 1.0f));
    assert(!isnan(min_normal.x));
    assert(!isnan(min_normal.y));
    assert(!isnan(min_normal.z));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(min_normal, 0.0f)));
    assert(!isnan(normal.x));
    assert(!isnan(normal.y));
    assert(!isnan(normal.z));
    return glm::length(r.origin - intersectionPoint);

    return min_t;
}

// TODO
__host__ __device__ float meshIntersectionTestBVH(
    Geom geom,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    const Mesh& mesh = geom.mesh;
    return -1;
}