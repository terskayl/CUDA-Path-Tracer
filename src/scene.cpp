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
                    
                    newGeom.pos = new glm::vec3[count];
                    newGeom.posCount = count;
                    std::memcpy(newGeom.pos, buffer.data.data() + byteOffset, byteLength);
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

                    newGeom.nor = new glm::vec3[count];
                    newGeom.norCount = count;
                    std::memcpy(newGeom.nor, buffer.data.data() + byteOffset, byteLength);

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

                    newGeom.uv = new glm::vec2[count];
                    newGeom.uvCount = count;
                    std::memcpy(newGeom.uv, buffer.data.data() + byteOffset, byteLength);

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

                    newGeom.ind = new unsigned short[count];
                    newGeom.indCount = count;
                    std::memcpy(newGeom.ind, buffer.data.data() + byteOffset, byteLength);
                }
            }

            for (int i = 0; i < newGeom.posCount; ++i) {
                glm::vec3 v = newGeom.pos[i];
                std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            }
            for (int i = 0; i < newGeom.norCount; ++i) {
                glm::vec3 v = newGeom.nor[i];
                std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            }
            for (int i = 0; i < newGeom.uvCount; ++i) {
                glm::vec2 v = newGeom.uv[i];
                std::cout << v.r << ", " << v.g << std::endl;
            }
            for (int i = 0; i < newGeom.indCount; ++i) {
                unsigned short s = newGeom.ind[i];
                std::cout << s << std::endl;
            }
            geoms.push_back(newGeom);
        }

        if (node.camera != -1) {
            // TODO CAMERA
        }
    }




}