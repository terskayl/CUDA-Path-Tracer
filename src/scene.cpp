#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "tiny_gltf.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <deque>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        bool success = loadFromGLTF(filename, ext == ".glb");
        if (!success) {
            cout << "Error reading gltf file " << filename << endl;
            exit(-1);
        }
    } 
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

bool Scene::loadFromGLTF(const std::string& gltfName, bool isBinary)
{
    printf("GLTF detected!\n");

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    if (isBinary) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName); // for binary glTF(.glb)
    }
    else
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    }

    if (!warn.empty()) {
        printf("Warn: %s\n", warn.c_str());
    }

    if (!err.empty()) {
        printf("Err: %s\n", err.c_str());
    }

    if (!ret) {
        printf("Failed to parse glTF\n");
        return false;
    }

    // TODO: MATERIALS
 
    // Loop through all nodes, then through meshes
    for (tinygltf::Node& node : model.nodes) {
        Geom newGeom;

        // Transforms
        if (node.translation.size()) {
            newGeom.translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        }
        if (node.rotation.size()) {
            glm::quat rot = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
            newGeom.rotation = glm::eulerAngles(rot);
        }
        if (node.scale.size()) {
            newGeom.scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        }
        if (node.matrix.size()) {
            std::vector<double> mat = node.matrix;
            glm::mat4 otherMat = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.transform = glm::mat4( mat[0],  mat[1],  mat[2],  mat[3],
                                           mat[4],  mat[5],  mat[6],  mat[7],
                                           mat[8],  mat[9],  mat[10], mat[11],
                                           mat[12], mat[13], mat[14], mat[15] );
            // TODO check this works
            assert(otherMat == newGeom.transform);
        }
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        
        // TODO: what happens when the node doesn't have a mesh?
        if (node.mesh != -1) {
            newGeom.type = MESH;
            tinygltf::Mesh mesh = model.meshes[node.mesh];

            for (tinygltf::Primitive& prim: mesh.primitives) {
                // TODO: EACH PRIMATIVE HAS A MATERIAL, DO MATMAP HERE
                
                assert(prim.mode == 4); // GLTF encoding for TRIANGLES

                if (prim.attributes.count("POSITION") > 0) {
                    int accessorId = prim.attributes["POSITION"];
                    tinygltf::Accessor accessor = model.accessors[accessorId];
                    int bufferViewId = accessor.bufferView;
                    assert(accessor.componentType == 5126); // GLTF encoding for float
                    assert(accessor.type == 3); // GLTF encoding for vec3
                    size_t count = accessor.count;

                    tinygltf::BufferView bView = model.bufferViews[bufferViewId];
                    size_t byteLength = bView.byteLength;
                    size_t byteOffset = bView.byteOffset;
                    size_t bufferStride = bView.byteStride;
                    int bufferId = bView.buffer;

                    tinygltf::Buffer buffer = model.buffers[bufferId];
                    
                    newGeom.mesh.pos = new glm::vec3[count];
                    newGeom.mesh.posCount = count;
                    std::memcpy(newGeom.mesh.pos, buffer.data.data() + byteOffset, byteLength);
                }
                if (prim.attributes.count("NORMAL") > 0) {
                    int accessorId = prim.attributes["NORMAL"];
                    tinygltf::Accessor accessor = model.accessors[accessorId];
                    int bufferViewId = accessor.bufferView;
                    assert(accessor.componentType == 5126); // GLTF encoding for float
                    assert(accessor.type == 3); // GLTF encoding for vec3
                    size_t count = accessor.count;

                    tinygltf::BufferView bView = model.bufferViews[bufferViewId];
                    size_t byteLength = bView.byteLength;
                    size_t byteOffset = bView.byteOffset;
                    size_t bufferStride = bView.byteStride;
                    int bufferId = bView.buffer;

                    tinygltf::Buffer buffer = model.buffers[bufferId];

                    newGeom.mesh.nor = new glm::vec3[count];
                    newGeom.mesh.norCount = count;
                    std::memcpy(newGeom.mesh.nor, buffer.data.data() + byteOffset, byteLength);

                }
                if (prim.attributes.count("TEXCOORD_0") > 0) {
                    int accessorId = prim.attributes["TEXCOORD_0"];
                    tinygltf::Accessor accessor = model.accessors[accessorId];
                    int bufferViewId = accessor.bufferView;
                    assert(accessor.componentType == 5126); // GLTF encoding for float
                    assert(accessor.type == 2); // GLTF encoding for vec2
                    size_t count = accessor.count;

                    tinygltf::BufferView bView = model.bufferViews[bufferViewId];
                    size_t byteLength = bView.byteLength;
                    size_t byteOffset = bView.byteOffset;
                    size_t bufferStride = bView.byteStride;
                    int bufferId = bView.buffer;

                    tinygltf::Buffer buffer = model.buffers[bufferId];

                    newGeom.mesh.uv = new glm::vec2[count];
                    newGeom.mesh.uvCount = count;
                    std::memcpy(newGeom.mesh.uv, buffer.data.data() + byteOffset, byteLength);

                }

                // Does prim.indices == -1 when it doesn't exist?
                if (prim.indices != -1) {
                    int accessorId = prim.indices;
                    tinygltf::Accessor accessor = model.accessors[accessorId];
                    int bufferViewId = accessor.bufferView;
                    assert(accessor.componentType == 5123); // GLTF encoding for unsigned_short
                    assert(accessor.type == 65); // GLTF encoding for scalar
                    size_t count = accessor.count;

                    tinygltf::BufferView bView = model.bufferViews[bufferViewId];
                    size_t byteLength = bView.byteLength;
                    size_t byteOffset = bView.byteOffset;
                    size_t bufferStride = bView.byteStride;
                    int bufferId = bView.buffer;

                    tinygltf::Buffer buffer = model.buffers[bufferId];

                    newGeom.mesh.ind = new unsigned short[count];
                    newGeom.mesh.indCount = count;
                    std::memcpy(newGeom.mesh.ind, buffer.data.data() + byteOffset, byteLength);
                }
            }

            for (int i = 0; i < newGeom.mesh.posCount; ++i) {
                glm::vec3 v = newGeom.mesh.pos[i];
                std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            }
            for (int i = 0; i < newGeom.mesh.norCount; ++i) {
                glm::vec3 v = newGeom.mesh.nor[i];
                std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            }
            for (int i = 0; i < newGeom.mesh.uvCount; ++i) {
                glm::vec2 v = newGeom.mesh.uv[i];
                std::cout << v.r << ", " << v.g << std::endl;
            }
            for (int i = 0; i < newGeom.mesh.indCount; ++i) {
                unsigned short s = newGeom.mesh.ind[i];
                std::cout << s << std::endl;
            }

            buildBVH(newGeom.mesh);

            geoms.push_back(newGeom);
        }

        if (node.camera != -1) {
            // TODO CAMERA
        }
    }
}

void Scene::buildBVH(const Mesh& mesh) {
    int n = mesh.indCount / 3;
    assert(mesh.indCount % 3 == 0); // TRUE FOR AS LONG AS WE USE GL_TRIANGLES

    // TEST
    glm::vec3 v1 = glm::vec3(1, 2, 3);
    glm::vec3 v2 = glm::vec3(3, 2, 1);
    assert(glm::vec3(3, 2, 3) == glm::max(v1, v2));

    assert(mesh.posCount > 0);
    glm::vec3 maxBounds = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
    glm::vec3 minBounds = glm::vec3(INFINITY, INFINITY, INFINITY);

    for (int i = 0; i < mesh.posCount; ++i) {
        maxBounds = glm::max(maxBounds, mesh.pos[i]);
        minBounds = glm::min(minBounds, mesh.pos[i]);
    }

    struct Triangle {
        glm::vec3 points[3];
        int ind[3];
    };

    struct CpuBvhNode{
        std::vector<Triangle> tri;
        CpuBvhNode* lChild;
        CpuBvhNode* rChild;
        glm::vec3 maxBounds;
        glm::vec3 minBounds;

        CpuBvhNode() {
            tri = std::vector<Triangle>();
            lChild = nullptr;
            rChild = nullptr;
        };

        CpuBvhNode(std::vector<Triangle> tri) {
            tri = tri;
            lChild = nullptr;
            rChild = nullptr;
        }
    };

    //queue of tree nodes for processing
    std::deque<CpuBvhNode*> nodesToProcess = std::deque<CpuBvhNode*>();

    std::vector<Triangle> tri = std::vector<Triangle>();
    for (int i = 0; i < n; ++i) {
        Triangle currTri = Triangle();
        currTri.ind[0] = mesh.ind[3 * i];
        currTri.ind[1] = mesh.ind[3 * i + 1];
        currTri.ind[2] = mesh.ind[3 * i + 2];
        currTri.points[0] = mesh.pos[currTri.ind[0]];
        currTri.points[1] = mesh.pos[currTri.ind[1]];
        currTri.points[2] = mesh.pos[currTri.ind[2]];
        tri.push_back(currTri);
    }


    CpuBvhNode root = CpuBvhNode();

    root.tri = tri;
    root.maxBounds = maxBounds;
    root.minBounds = minBounds;

    nodesToProcess.push_back(&root);
    const int BVH_MAX_LAYERS = 20;
    for (int i = 0; i < BVH_MAX_LAYERS; ++i) {
        int layerSize = 0;
        for (CpuBvhNode* nodePtr : nodesToProcess) {
            // Find axis to split
            int axisToSplit = 0;
            if (nodePtr->maxBounds.y - nodePtr->minBounds.y > nodePtr->maxBounds.x - nodePtr->minBounds.x) {
                axisToSplit += 1;
                if (nodePtr->maxBounds.z - nodePtr->minBounds.z > nodePtr->maxBounds.y - nodePtr->minBounds.x) {
                    axisToSplit += 1;
                }
            }
            else if (nodePtr->maxBounds.z - nodePtr->minBounds.z > nodePtr->maxBounds.x - nodePtr->minBounds.x) {
                axisToSplit += 2;
            }
            // Split bounding boxes - just middle
            float splitPos = (nodePtr->maxBounds[axisToSplit] + nodePtr->minBounds[axisToSplit]) / 2;
            glm::vec3 bb1MaxBounds = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
            glm::vec3 bb1MinBounds = glm::vec3(INFINITY, INFINITY, INFINITY);
            glm::vec3 bb2MaxBounds = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
            glm::vec3 bb2MinBounds = glm::vec3(INFINITY, INFINITY, INFINITY);

            std::vector<Triangle> bb1tris = std::vector<Triangle>();
            std::vector<Triangle> bb2tris = std::vector<Triangle>();
            for (int j = 0; j < nodePtr->tri.size(); j++) {
                Triangle currTri= nodePtr->tri[j];
                glm::vec3 *pts = currTri.points;
                glm::vec3 centroid = (pts[0] + pts[1] + pts[2]);
                centroid /= 3;

                if (centroid[axisToSplit] > splitPos) {
                    // put in 1
                    bb1MaxBounds = glm::max(bb1MaxBounds, pts[0]);
                    bb1MaxBounds = glm::max(bb1MaxBounds, pts[1]);
                    bb1MaxBounds = glm::max(bb1MaxBounds, pts[2]);
                    bb1MinBounds = glm::min(bb1MinBounds, pts[0]);
                    bb1MinBounds = glm::min(bb1MinBounds, pts[1]);
                    bb1MinBounds = glm::min(bb1MinBounds, pts[2]);
                    bb1tris.push_back(currTri);
                }
                else {
                    // put in 2
                    bb2MaxBounds = glm::max(bb2MaxBounds, pts[0]);
                    bb2MaxBounds = glm::max(bb2MaxBounds, pts[1]);
                    bb2MaxBounds = glm::max(bb2MaxBounds, pts[2]);
                    bb2MinBounds = glm::min(bb2MinBounds, pts[0]);
                    bb2MinBounds = glm::min(bb2MinBounds, pts[1]);
                    bb2MinBounds = glm::min(bb2MinBounds, pts[2]);
                    bb2tris.push_back(currTri);
                }

            }
                
            if (bb1tris.size() == 0 || bb2tris.size() == 0) {
                continue;
            }
            CpuBvhNode& lChild = *(new CpuBvhNode(bb1tris));
            CpuBvhNode& rChild = *(new CpuBvhNode(bb2tris));
            lChild.maxBounds = bb1MaxBounds;
            lChild.minBounds = bb1MinBounds;
            rChild.maxBounds = bb2MaxBounds;
            rChild.minBounds = bb2MinBounds;
            
            lChild.tri = bb1tris;
            rChild.tri = bb2tris;

            nodePtr->lChild = &lChild;
            nodePtr->rChild = &rChild;

            // TODO - can optimize order?
            nodesToProcess.push_back(&lChild);
            nodesToProcess.push_back(&rChild);

            layerSize += 1;
        }
        for (int i = 0; i < layerSize; ++i) {
            nodesToProcess.pop_front();
        }
    }

    // Testing
    std::deque<CpuBvhNode> nodesToPrint = std::deque<CpuBvhNode>();
    nodesToPrint.push_back(root);

    while (!nodesToPrint.empty()) {
        CpuBvhNode currNode = nodesToPrint[0];
        nodesToPrint.pop_front();
        if (currNode.lChild != nullptr) {
            nodesToPrint.push_back(*currNode.lChild);
        }
        if (currNode.rChild != nullptr) {
            nodesToPrint.push_back(*currNode.rChild);
        }
        if (currNode.lChild == nullptr && currNode.rChild == nullptr) {
            printf("New Node:\n");
            printf("Bounds: %.3f to %.3f, %.3f to %.3f, %.3f to %.3f\n", currNode.minBounds[0],
                currNode.maxBounds[0], currNode.minBounds[1], currNode.maxBounds[1],
                currNode.minBounds[2], currNode.maxBounds[2]);
            if (currNode.lChild != nullptr) {
                printf("Has left child with bounds: %.3f to %.3f, %.3f to %.3f, %.3f to %.3f\n",
                    currNode.lChild->minBounds[0], currNode.lChild->maxBounds[0], currNode.lChild->minBounds[1], currNode.lChild->maxBounds[1],
                    currNode.lChild->minBounds[2], currNode.lChild->maxBounds[2]);
                nodesToPrint.push_back(*currNode.lChild);
            }
            if (currNode.rChild != nullptr) {
                printf("Has right child with bounds: %.3f to %.3f, %.3f to %.3f, %.3f to %.3f\n",
                    currNode.rChild->minBounds[0], currNode.rChild->maxBounds[0], currNode.rChild->minBounds[1], currNode.rChild->maxBounds[1],
                    currNode.rChild->minBounds[2], currNode.rChild->maxBounds[2]);
                nodesToPrint.push_back(*currNode.rChild);
            }

            printf("Tris: ");
            for (Triangle t : currNode.tri) {
                printf("[(%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)], ",
                    t.points[0][0], t.points[0][1], t.points[0][2],
                    t.points[1][0], t.points[1][1], t.points[1][2],
                    t.points[2][0], t.points[2][1], t.points[2][2]);
            }
            printf("\n");
        }
    }
    // END TESTING
    // TODO: NEED TO FREE CPUBVHNODE TREE

}