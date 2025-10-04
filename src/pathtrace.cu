#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <stream_compaction/efficient.cu>
#include <OpenImageDenoise/oidn.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define STREAMCOMPACTION 1
#define MATERIALSORTING 1
#define BVH 1;
#define ACES 1;
#define REINHARD 0;
#define GAMMACORRECTION 1;
#define RUSSIANROULETTE 1;
#define OIDN 1;

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        glm::vec3 postReinhard;

        pix.x /= iter;
        pix.y /= iter;
        pix.z /= iter;

#if REINHARD
        // Reinhard Operator
        pix.x = pix.x / (pix.x + 1);
        pix.y = pix.y / (pix.y + 1);
        pix.z = pix.z / (pix.z + 1);
#endif

#if ACES
        float a = 2.51f;
        float b = 0.03f;
        float c = 2.43f;
        float d = 0.59f;
        float e = 0.14f;
        pix = (pix * (a * pix + b)) / (pix * (c * pix + d) + e);
#endif

#if GAMMACORRECTION
        pix.x = glm::pow(pix.x, 1.0 / 2.2);
        pix.y = glm::pow(pix.y, 1.0 / 2.2);
        pix.z = glm::pow(pix.z, 1.0 / 2.2);
#endif 
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);;

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static glm::vec3* dev_image_albedo = NULL;
static glm::vec3* dev_image_normal = NULL;
static glm::vec3* dev_image_denoised = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int* dev_isValidIntersection = NULL;
static DeviceTexture* dev_textures = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    // Checkpointed image copy
    printf("Init called with iter: %ui", hst_scene->state.currIteration);
    if (hst_scene->state.currIteration != 0) {
        cudaMemcpy(dev_image, hst_scene->state.image.data(),
            pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_image_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_image_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_normal, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_image_denoised, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image_denoised, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    printf("INITING");

    // Copy all mesh data onto the GPU
    checkCUDAError("other");
    for (Geom& geom : scene->geoms) {
        if (geom.type == MESH && !geom.mesh.onGpu) {
            Mesh& mesh = geom.mesh;

            checkCUDAError("other");
            glm::vec3 *posTmp, *norTmp;
            glm::vec2* uvTmp;
            unsigned short* indTmp, *indBvhTmp;
            BvhNode* nodesTmp;

            checkCUDAError("other");
            cudaMalloc((void**)&posTmp, mesh.posCount * sizeof(glm::vec3));
            cudaMemcpy(posTmp, mesh.pos, mesh.posCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            free(mesh.pos);
            mesh.pos = posTmp;

            checkCUDAError("other");
            cudaMalloc((void**)&norTmp, mesh.norCount * sizeof(glm::vec3));
            cudaMemcpy(norTmp, mesh.nor, mesh.norCount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            free(mesh.nor);
            mesh.nor = norTmp;


            size_t size = mesh.uvCount * sizeof(float3);
            std::cout << "Allocating " << size << " bytes (" << (size / (1024.0 * 1024.0)) << " MB)\n";
            checkCUDAError("other");
            cudaMalloc((void**)&uvTmp, mesh.uvCount * sizeof(glm::vec2));
            cudaMemcpy(uvTmp, mesh.uv, mesh.uvCount * sizeof(glm::vec2), cudaMemcpyHostToDevice);
            free(mesh.uv);
            mesh.uv = uvTmp;

            checkCUDAError("other");
            cudaMalloc((void**)&indTmp, mesh.indCount * sizeof(unsigned short));
            cudaMemcpy(indTmp, mesh.ind, mesh.indCount * sizeof(unsigned short), cudaMemcpyHostToDevice);
            free(mesh.ind);
            mesh.ind = indTmp;

            if (mesh.numBvhNodes > 0) {
                cudaMalloc((void**)&nodesTmp, mesh.numBvhNodes * sizeof(BvhNode));
                cudaMemcpy(nodesTmp, mesh.bvhNodes, mesh.numBvhNodes * sizeof(BvhNode), cudaMemcpyHostToDevice);
                free(mesh.bvhNodes);
                mesh.bvhNodes = nodesTmp;

                cudaMalloc((void**)&indBvhTmp, mesh.indCount * sizeof(unsigned short));
                cudaMemcpy(indBvhTmp, mesh.indBVH, mesh.indCount * sizeof(unsigned short), cudaMemcpyHostToDevice);
                free(mesh.indBVH);
                mesh.indBVH = indBvhTmp;
            }

            geom.mesh.onGpu = true;
            printf("LOADED ONTO THE GPU");
            cudaDeviceSynchronize();
        }
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    checkCUDAError("other");
    cudaMalloc(&dev_isValidIntersection, pixelcount * sizeof(int));
    cudaMemset(dev_isValidIntersection, 0, pixelcount * sizeof(int));

    // We will assemble a new textures array here to put on the GPU 
    // because I believe std::vectors cannot go onto the GPU.
    std::vector<DeviceTexture> dev_textureArr;

    // Put all textures onto the GPU
    // TODO: Will this work with 3 channel textures or will I need to pad?
    for (Texture& tex : scene->textures) {
        if (tex.numChannels < 1 || tex.numChannels > 4) {
            throw std::runtime_error("Unsupported channel count");
        }
        if (tex.bitsPerChannel != 8 && tex.bitsPerChannel != 16 && tex.bitsPerChannel != 32) {
            throw std::runtime_error("Unsupported bit depth (only 8, 16 or 32)");
        }

        // 1. Create channel descriptor
        checkCUDAError("other");
        cudaChannelFormatDesc channelDesc;
        if (tex.bitsPerChannel == 8) {
            channelDesc = cudaCreateChannelDesc(
                (tex.numChannels >= 1) ? 8 : 0,
                (tex.numChannels >= 2) ? 8 : 0,
                (tex.numChannels >= 3) ? 8 : 0,
                (tex.numChannels >= 4) ? 8 : 0,
                cudaChannelFormatKindUnsigned);
        }
        else if (tex.bitsPerChannel == 16) {
            channelDesc = cudaCreateChannelDesc(
                (tex.numChannels >= 1) ? 16 : 0,
                (tex.numChannels >= 2) ? 16 : 0,
                (tex.numChannels >= 3) ? 16 : 0,
                (tex.numChannels >= 4) ? 16 : 0,
                cudaChannelFormatKindUnsigned);
        }
        else { // 32-bit float
            channelDesc = cudaCreateChannelDesc(
                (tex.numChannels >= 1) ? 32 : 0,
                (tex.numChannels >= 2) ? 32 : 0,
                (tex.numChannels >= 3) ? 32 : 0,
                (tex.numChannels >= 4) ? 32 : 0,
                cudaChannelFormatKindFloat);
        }
        checkCUDAError("Create Channel Desc");

        // 2. Allocate CUDA array
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, tex.width, tex.height);

        // 3. Copy host → device array
        size_t elemSize = tex.bitsPerChannel / 8;
        size_t rowSizeBytes = tex.width * tex.numChannels * elemSize;
        checkCUDAError("Create Channel Desc2");

        cudaMemcpy2DToArray(
            cuArray, 0, 0,
            tex.data.data(), rowSizeBytes,
            rowSizeBytes, tex.height,
            cudaMemcpyHostToDevice);
        checkCUDAError("Copy texture to GPU");

        // 4. Resource description
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // 5. Texture description
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        if (tex.bitsPerChannel == 32) {
            texDesc.readMode = cudaReadModeElementType; // If float type, do not normalize
        }
        texDesc.normalizedCoords = 1;             

        // 6. Create texture object
        cudaTextureObject_t texObj = 0;
        cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
        checkCUDAError("Create texture handle");

        tex.d_data = cuArray;
        tex.d_texHandle = texObj;

        DeviceTexture dTex;
        dTex.texHandle = texObj;
        dTex.width = tex.width;
        dTex.height = tex.height;
        dev_textureArr.push_back(dTex);
    }
    
    cudaMalloc(&dev_textures, dev_textureArr.size()* sizeof(DeviceTexture));
    cudaMemcpy(dev_textures, dev_textureArr.data(), dev_textureArr.size() * sizeof(DeviceTexture), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree(Scene* scene)
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_image_albedo);
    cudaFree(dev_image_normal);
    cudaFree(dev_image_denoised);
    cudaFree(dev_paths);
    checkCUDAError("pathtraceFree1");

    // Free individual mesh buffers
    for (Geom& geom : scene->geoms) {
        if (geom.type == MESH && geom.mesh.onGpu) {
            Mesh& mesh = geom.mesh;

            glm::vec3* posTmp = new glm::vec3[mesh.posCount];
            glm::vec3* norTmp = new glm::vec3[mesh.norCount];
            glm::vec2* uvTmp = new glm::vec2[mesh.uvCount];
            unsigned short* indTmp = new unsigned short[mesh.indCount];
            BvhNode* nodesTmp = new BvhNode[mesh.numBvhNodes];
            unsigned short* indBvhTmp = new unsigned short[mesh.indCount];

            cudaMemcpy(posTmp, mesh.pos, mesh.posCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaFree(mesh.pos);
            checkCUDAError("pathtraceFree2");
            mesh.pos = posTmp;

            cudaMemcpy(norTmp, mesh.nor, mesh.norCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
            cudaFree(mesh.nor);
            checkCUDAError("pathtraceFree2");
            mesh.nor = norTmp;

            cudaMemcpy(uvTmp, mesh.uv, mesh.uvCount * sizeof(glm::vec2), cudaMemcpyDeviceToHost);
            cudaFree(mesh.uv);
            checkCUDAError("pathtraceFree2");
            mesh.uv = uvTmp;

            cudaMemcpy(indTmp, mesh.ind, mesh.indCount * sizeof(unsigned short), cudaMemcpyDeviceToHost);
            cudaFree(mesh.ind);
            checkCUDAError("pathtraceFree2");
            mesh.ind = indTmp;

            if (mesh.numBvhNodes > 0) {
                cudaMemcpy(nodesTmp, mesh.bvhNodes, mesh.numBvhNodes * sizeof(BvhNode), cudaMemcpyDeviceToHost);
                cudaFree(mesh.bvhNodes);
                checkCUDAError("pathtraceFree2");
                mesh.bvhNodes = nodesTmp;

                cudaMemcpy(indBvhTmp, mesh.indBVH, mesh.indCount * sizeof(unsigned short), cudaMemcpyDeviceToHost);
                cudaFree(mesh.indBVH);
                checkCUDAError("pathtraceFree2");
                mesh.indBVH = indBvhTmp;
            }

            geom.mesh.onGpu = false;
            printf("LOADED BACK ONTO CPU");
            cudaDeviceSynchronize();

        }
    }
    checkCUDAError("pathtraceFree2");

    // Free textures
    for (Texture& tex : scene->textures) {
        if (tex.d_texHandle) {
            cudaDestroyTextureObject(tex.d_texHandle);
            tex.d_texHandle = 0;
        }
        if (tex.d_data) {
            cudaFreeArray(tex.d_data);
            tex.d_data = nullptr;
        }
    }

    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_isValidIntersection);
    cudaFree(dev_textures);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.radiance = glm::vec3(0.0f, 0.0f, 0.0f);
        segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.firstAlbedo = glm::vec3(-1.f);
        segment.firstNormal = glm::vec3(-1.f);


        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            + cam.right * cam.pixelLength.x * (u01(rng) - 0.5f)
            + cam.up * cam.pixelLength.y * (u01(rng) - 0.5f)
        );

        // DOF implementation
        //TODO: attach to ui
        float focal_distance = 2.f;
        glm::vec3 target = cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            + cam.right * cam.pixelLength.x * (u01(rng) - 0.5f)
            + cam.up * cam.pixelLength.y * (u01(rng) - 0.5f);
        target *= focal_distance;

        //TODO: attach to ui
        float dofScale = 0.0f;
        glm::vec3 posOffset = dofScale * cam.right * (u01(rng) - 0.5f)
            + dofScale * cam.up * (u01(rng) - 0.5f);
        
        segment.ray.origin = cam.position + posOffset;
        segment.ray.direction = glm::normalize(target - posOffset);


        assert(fabs(glm::length(segment.ray.direction)) - 1 < 0.01);


        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    int* isValidIntersection,
    DeviceTexture* textures,
    int hdriIndex)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment& pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec3 tangent;
        glm::vec3 bitangent;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;
        glm::vec3 tmp_tangent;
        glm::vec3 tmp_bitangent;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                assert(!isnan(tmp_normal.x));
                assert(!isnan(tmp_normal.y));
                assert(!isnan(tmp_normal.z));
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                assert(!isnan(tmp_normal.x));
                assert(!isnan(tmp_normal.y));
                assert(!isnan(tmp_normal.z));
            }
            else if (geom.type == MESH) {

#if BVH
                t = meshIntersectionTestBVH(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, tmp_bitangent, outside);
#else
                t = meshIntersectionTestNaive(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, tmp_bitangent, outside);
#endif
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                tangent = tmp_tangent;
                bitangent = tmp_bitangent;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            isValidIntersection[path_index] = 0;

            //HDRI eval;
            if (hdriIndex != -1) {
                pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
                assert(fabs(glm::length(pathSegment.ray.direction) - 1.f) < 0.01f);

                float theta = atan2(pathSegment.ray.direction.x, pathSegment.ray.direction.z);
                float phi = asin(-pathSegment.ray.direction.y);

                float u = theta / (2 * PI) + 0.5;
                float v = phi / PI + 0.5;
                float4 hdriVal = tex2D<float4>(textures[hdriIndex].texHandle, u, v);
                pathSegment.radiance = glm::vec3(hdriVal.x, hdriVal.y, hdriVal.z);
            }

        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            assert(!isnan(normal.x));
            assert(!isnan(normal.y));
            assert(!isnan(normal.z));
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uvCoord = uv;
            intersections[path_index].surfaceTangent = tangent;
            intersections[path_index].surfaceBitangent = bitangent;
            isValidIntersection[path_index] = 1;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    DeviceTexture* textures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        pathSegments[idx].remainingBounces--;

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.baseColor;

            // If the material indicates that the object was a light, "light" the ray
            if (glm::length(material.emissive) > 0.0f) {
                pathSegments[idx].radiance = material.emissive;
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            else {
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not
                assert(!isnan(intersection.surfaceNormal.x));
                assert(!isnan(intersection.surfaceNormal.y));
                assert(!isnan(intersection.surfaceNormal.z));
                scatterRay(pathSegments[idx], pathSegments[idx].ray.direction * intersection.t + pathSegments[idx].ray.origin,
                    intersection.surfaceNormal, material, textures, intersection.uvCoord, intersection.surfaceTangent,
                    intersection.surfaceBitangent, rng);
#if RUSSIANROULETTE
                // Russian Roulette
                float probDie = glm::clamp(glm::length(pathSegments[idx].throughput), 0.05f, 0.95f);
                if (u01(rng) > probDie) {
                    pathSegments[idx].remainingBounces = 0;
                }
                else {
                    pathSegments[idx].throughput /= probDie;
                }
#endif

            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            //pathSegments[idx].throughput = glm::vec3(1.0f);
            pathSegments[idx].remainingBounces = 0;

        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, glm::vec3* image_albedo, glm::vec3* image_normal, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.throughput * iterationPath.radiance;
#if OIDN
        image_albedo[iterationPath.pixelIndex] = iterationPath.firstAlbedo;
        image_normal[iterationPath.pixelIndex] = iterationPath.firstNormal;
#endif 
    }
}


struct CompareIntersectionMaterials {
    template <typename Tuple>
    __host__ __device__ 
        bool operator()(const Tuple& a, const Tuple& b) const {
        // thrust::get<0>(a) retrieves the StructA from the tuple
        return thrust::get<0>(a).materialId < thrust::get<0>(b).materialId;
    }
};

struct GetKey {
    __host__ __device__
        int operator()(const ShadeableIntersection& s) const {
        return s.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths_total = dev_path_end - dev_paths;
    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    int num_paths = num_paths_total;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        cudaMemset(dev_isValidIntersection, 0, pixelcount * sizeof(int));

        checkCUDAError("cudaMemset");

        for (Geom geom : hst_scene->geoms) {
            if (geom.type == MESH) {
                assert(geom.mesh.onGpu);
            }
        }
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        checkCUDAError("pre-intersection");

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_isValidIntersection,
            dev_textures,
            hst_scene->hdriIndex
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        //int* boolsHost = new int[num_paths];
        //cudaMemcpy(boolsHost, dev_isValidIntersection, num_paths * sizeof(int), cudaMemcpyDeviceToHost);
        //printf("Before: \n");
        //for (int i = 0; i < num_paths; ++i) {
        //    printf("%i , ", boolsHost[i]);
        //}
        //printf("\n");

        //PathSegment* paths = new PathSegment[num_paths];
        //cudaMemcpy(paths, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToHost);
        //printf("Before: \n");
        //for (int i = 0; i < num_paths; ++i) {
        //    printf("%i , ", paths[i].pixelIndex);
        //}
        //printf("\n");


        //thrust::device_vector<PathSegment> v(dev_paths, dev_paths + num_paths);
        //thrust::device_vector<int>::iterator pivot = thurst::stable_partition(v.begin(), v.end(), function)
        //int count_hit = pivot - v.begin();
        //int count_miss = v.end() - pivot;

        //std::cout << "Even numbers: " << count_hit << std::endl;
        //std::cout << "Odd numbers: " << count_miss << std::endl;

        //// Print the result (copy to host to view)
        //thrust::host_vector<int> h_v = v;
        //for (PathSegment x : h_v) {
        //    std::cout << x.pixelIndex << " ";
        //}
        //std::cout << std::endl;

#if STREAMCOMPACTION
        num_paths = StreamCompaction::Efficient::partitionOnValidIntersect(num_paths, dev_paths, dev_isValidIntersection, dev_intersections);
        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        checkCUDAError("partition on intersect");
        if (num_paths == 0) break;
#endif


        //ShadeableIntersection* intersections = new ShadeableIntersection[num_paths];
        //cudaMemcpy(intersections, dev_intersections, 100 * sizeof(ShadeableIntersection), cudaMemcpyDeviceToHost);

        //for (int i = 0; i < 100; ++i) {
        //    if (intersections[i].uvCoord.x != 0.0) {
        //        printf("Intersection: %.2f, %.2f\n", intersections[i].uvCoord.x, intersections[i].uvCoord.y);
        //    }
        //}

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.
#if MATERIALSORTING
        thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
        thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);

        //auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(dev_thrust_intersections, dev_thrust_paths));
        //auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(dev_thrust_intersections + num_paths, dev_thrust_paths + num_paths));
        
        thrust::device_vector<int> d_keys(num_paths);
        thrust::device_vector<int> d_keys2(num_paths);

        // thrust::transform extracts materialId from dev_intersections, and makes two copies
        thrust::transform(dev_thrust_intersections, dev_thrust_intersections + num_paths, d_keys.begin(), GetKey());
        thrust::transform(dev_thrust_intersections, dev_thrust_intersections + num_paths, d_keys2.begin(), GetKey());

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), dev_thrust_paths);//zip_begin); ZIP IS SLOWER I THINK, TODO - check
        thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), dev_thrust_intersections);

#endif
        shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_textures
        );
        checkCUDAError("shading");

#if STREAMCOMPACTION
        num_paths = StreamCompaction::Efficient::partitionOnBounces(num_paths, dev_paths);
        numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        checkCUDAError("partition on bounces");

#endif


        //PathSegment* paths = new PathSegment[num_paths];
        //cudaMemcpy(paths, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToHost);
        //printf("After: \n");
        //for (int i = 0; i < num_paths; ++i) {
        //    printf("%i, ", paths[i].remainingBounces);
        //}
        //printf("\n");
        //delete[] paths;

        if (depth >= traceDepth || num_paths <= 0)iterationComplete = true;
        //if (num_paths <= 0) iterationComplete = true; // TODO: should be based off stream compaction results.



        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_image_albedo, dev_image_normal, dev_paths);

# if OIDN
    // denoise with OIDN
    oidn::DeviceRef device = oidn::newCUDADevice(0, NULL);
    device.commit();

    // Setup buffers
    oidn::BufferRef colorBuf = device.newBuffer(pixelcount * 3 * sizeof(float));
    oidn::BufferRef albedoBuf = device.newBuffer(pixelcount * 3 * sizeof(float));
    oidn::BufferRef normalBuf = device.newBuffer(pixelcount * 3 * sizeof(float));
    oidn::BufferRef outputBuf = device.newBuffer(pixelcount * 3 * sizeof(float));

    cudaMemcpy(colorBuf.getData(), dev_image, pixelcount * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(albedoBuf.getData(), dev_image_albedo, pixelcount * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(normalBuf.getData(), dev_image_normal, pixelcount * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    oidn::FilterRef filter = device.newFilter("RT"); // ray tracing filter
    filter.setImage("color", colorBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("albedo", albedoBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("normal", normalBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.setImage("output", outputBuf, oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
    filter.set("hdr", true);
    filter.commit();

    filter.execute();

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None) {
        printf("Error: ");
        printf(errorMessage);
    }

    cudaMemcpy(dev_image_denoised, outputBuf.getData(), pixelcount * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
#endif

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
#if OIDN
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image_denoised);
#else
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
#endif
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
