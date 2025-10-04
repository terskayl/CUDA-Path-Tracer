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

template <typename T>
using uPtr = std::unique_ptr<T>;

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

inline void from_json(const json& j, Camera& c) {
    auto res = j.at("resolution");
    c.resolution.x = res.at(0).get<int>();
    c.resolution.y = res.at(1).get<int>();

    auto pos = j.at("position");
    c.position.x = pos.at(0).get<float>();
    c.position.y = pos.at(1).get<float>();
    c.position.z = pos.at(2).get<float>();

    auto lookAt = j.at("lookAt");
    c.lookAt.x = lookAt.at(0).get<float>();
    c.lookAt.y = lookAt.at(1).get<float>();
    c.lookAt.z = lookAt.at(2).get<float>();

    auto view = j.at("view");
    c.view.x = view.at(0).get<float>();
    c.view.y = view.at(1).get<float>();
    c.view.z = view.at(2).get<float>();

    auto up = j.at("up");
    c.up.x = up.at(0).get<float>();
    c.up.y = up.at(1).get<float>();
    c.up.z = up.at(2).get<float>();

    auto right = j.at("right");
    c.right.x = right.at(0).get<float>();
    c.right.y = right.at(1).get<float>();
    c.right.z = right.at(2).get<float>();

    auto fov = j.at("fov");
    c.fov.x = fov.at(0).get<float>();
    c.fov.y = fov.at(1).get<float>();

    auto pixelLength = j.at("pixelLength");
    c.pixelLength.x = pixelLength.at(0).get<float>();
    c.pixelLength.y = pixelLength.at(1).get<float>();
}

inline void from_json(const json& j, RenderState& r) {
    j.at("camera").get_to(r.camera);
    j.at("iterations").get_to(r.iterations);
    j.at("currIteration").get_to(r.currIteration);
    j.at("traceDepth").get_to(r.traceDepth);
    j.at("imageName").get_to(r.imageName);
}

Scene::Scene(string filename, string imageName, string renderStateJson)
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

    cout << "Reading renderState from " << renderStateJson << " ..." << endl;
    cout << " " << endl;
    ext = renderStateJson.substr(renderStateJson.find_last_of('.'));
    if (ext == ".json")
    {
        std::ifstream file(renderStateJson);
        if (!file) {
            cout << "Couldn't read from " << renderStateJson << endl;
            exit(-1);
        }

        json j;
        file >> j;

        state = j.get<RenderState>();

    }
    else
    {
        cout << "Couldn't read from " << renderStateJson << endl;
        exit(-1);
    }

    cout << "Reading image from " << imageName << " ..." << endl;
    cout << " " << endl;
    int x = 0, y = 0, channels = 0;
    float* savedImage = stbi_loadf(imageName.c_str(), &x, &y, &channels, 3); // Force load 3 channels
    if (x <= 0 || y <= 0) {
        const char* reason = stbi_failure_reason();
        if (reason) {
            std::cerr << "Image load failure with reason: " << reason << std::endl;
            exit(-1);
        }
    }
    state.image.resize(state.camera.resolution.x * state.camera.resolution.y);
    memcpy(state.image.data(), savedImage, state.camera.resolution.x * state.camera.resolution.y * 3 * sizeof(float));
    // weigh image by the number of current iterations
    for (glm::vec3& pixel : state.image) {
        pixel *= state.currIteration;
    }
    // Flip image because stbi horizontally flipped it?
    int width = state.camera.resolution.x;
    int halfWidth = state.camera.resolution.x / 2;

    for (int y = 0; y < state.camera.resolution.y; y++) {
        for (int x = 0; x < halfWidth; x++) {
            std::swap(
                state.image[y * width + x],
                state.image[y * width + (width - 1 - x)]
            );
        }
    }

    stbi_image_free(savedImage);
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
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.emissive = glm::vec3(col[0], col[1], col[2]);
            float emit = p["EMITTANCE"];
            newMaterial.emissive = glm::vec3(emit);
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.baseColor = glm::vec3(col[0], col[1], col[2]);
            newMaterial.roughness = 0;
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


        // READ HDRI
        //TODO Duplicate Code, refactor
// TODO : Connect to UI
        int x = 0, y = 0, channels = 0;
        float* hdriData = stbi_loadf("C:/Users/njbhv/Documents/Code/CIS5650/Project3-CUDA-Path-Tracer/scenes/passendorf_snow_1k.hdr", &x, &y, &channels, 0);

        if (x > 0 && y > 0) {
            Texture hdri;
            hdri.width = x;
            hdri.height = y;
            hdri.bitsPerChannel = 32; // via stbi_loadf
            hdri.numChannels = channels;

            int num_bytes = x * y * channels * sizeof(float);
            hdri.data.resize(num_bytes);
            memcpy(hdri.data.data(), hdriData, num_bytes);
            textures.push_back(hdri);
            hdriIndex = textures.size() - 1;
        }
        else {
            const char* reason = stbi_failure_reason();
            if (reason) {
                std::cerr << "Failure reason: " << reason << std::endl;
            }
        }

        // Pad 3 channel textures to 4 channels.
        for (Texture& t : textures) {
            if (t.numChannels == 3) {
                int elemSize = t.bitsPerChannel / 8;

                std::vector<uint8_t> paddedData;
                for (int i = 0; i < t.data.size(); ++i) {
                    int realIndex = i / elemSize;
                    paddedData.push_back(t.data[i]);
                    if (i % (3 * elemSize) == 3 * elemSize - 1) {
                        for (int j = 0; j < elemSize; ++j) {
                            paddedData.push_back(0);
                        }
                    }

                }
                t.data = paddedData;
                t.numChannels = 4;
            }
        }



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

static void createDefaultCamera(RenderState& state) {
    Camera& camera = state.camera;
    camera.resolution.x = 800;
    camera.resolution.y = 800;
    float fovy = 45.f;
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = "untitled";
    const double pos[3] = {0.0, 5.0, 10.5};
    const double lookat[3] = {0.0, 5.0, 0.0};
    const double up[3] = {0.0, 1.0, 0.0};
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);
}

static inline glm::vec3 doubleArrayToVec3(std::vector<double> arr) {
    return glm::vec3(arr[0], arr[1], arr[2]);
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

    // Materials
    for (tinygltf::Material& mat : model.materials) {
        Material newMat;
        newMat.baseColor = doubleArrayToVec3(mat.pbrMetallicRoughness.baseColorFactor);
        newMat.baseColorTexture = mat.pbrMetallicRoughness.baseColorTexture.index;
        newMat.metallic = mat.pbrMetallicRoughness.metallicFactor;
        newMat.roughness = mat.pbrMetallicRoughness.roughnessFactor;
        newMat.metallicRoughnessTexture = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
        
        if (mat.extensions.count("KHR_materials_specular") != 0) {
            auto specular = mat.extensions.find("KHR_materials_specular")->second;
            newMat.specular = specular.Get("specularFactor").GetNumberAsDouble();
            //newMat.specularTint = specular.Get("specularColorFactor")
        }
        if (mat.extensions.count("KHR_materials_ior") != 0) {
            newMat.ior = mat.extensions.find("KHR_materials_ior")->second.Get("ior").GetNumberAsDouble();
        }

        if (mat.extensions.count("KHR_materials_clearcoat")) {
            newMat.clearcoat = mat.extensions.find("KHR_materials_clearcoat")->second.Get("clearcoatFactor").GetNumberAsDouble();
            //newMat.clearcoatRoughness = mat.extensions.find("KHR_materials_clearcoat")->second.Get("NotSureIfExists").GetNumberAsDouble();
        }

        if (mat.extensions.count("KHR_materials_sheen")) {
            auto sheen = mat.extensions.find("KHR_materials_sheen")->second;
            newMat.sheen = 1;
            //newMat.sheenTint = sheen.Get("sheenColorFactor"); // Don't know how to use
            newMat.sheenRoughness = sheen.Get("sheenRoughnessFactor").GetNumberAsDouble();
        }

        if (mat.extensions.count("KHR_materials_transmission")) {
            newMat.transmission = mat.extensions.find("KHR_materials_transmission")->second.Get("transmissionFactor").GetNumberAsDouble();
        }

        newMat.emissive = doubleArrayToVec3(mat.emissiveFactor);
        if (mat.extensions.count("KHR_materials_emissive_strength")) {
            newMat.emissive *= mat.extensions.find("KHR_materials_emissive_strength")->second.Get("emissiveStrength").GetNumberAsDouble();
        }
        newMat.emissiveTexture = mat.emissiveTexture.index;

        newMat.normalTexture = mat.normalTexture.index;
        newMat.normalTextureScale = mat.normalTexture.scale;
        newMat.occlusionTexture = mat.occlusionTexture.index;

        newMat.doubleSided = mat.doubleSided;
        newMat.unlit = false;

        materials.push_back(newMat);
    }

    // Textures & Load onto GPU
    for (tinygltf::Texture& tex : model.textures) {
        Texture newTex;
        // TODO: samplers. For now we will only support linear texture sampling
        tinygltf::Sampler sampler = model.samplers[tex.sampler];
        tinygltf::Image img = model.images[tex.source];
        
        // TODO, for now we will only support textures with URIs. Not embedded textures
        newTex.width = img.width;
        newTex.height = img.height;
        newTex.bitsPerChannel = img.bits;
        newTex.numChannels = img.component;
        newTex.data = img.image;

        textures.push_back(newTex);
    }

    // READ HDRI
    // TODO : Connect to UI
    int x = 0, y = 0, channels = 0;
    float* hdriData = stbi_loadf("C:/Users/njbhv/Documents/Code/CIS5650/Project3-CUDA-Path-Tracer/scenes/passendorf_snow_1k.hdr", &x, &y, &channels, 0);
    if (x == 0 || y == 0) {

    }
    const char* reason = stbi_failure_reason();
    if (reason) {
        std::cerr << "Failure reason: " << reason << std::endl;
    }

    if (x > 0 && y > 0) {
        Texture hdri;
        hdri.width = x;
        hdri.height = y;
        hdri.bitsPerChannel = 32; // via stbi_loadf
        hdri.numChannels = channels;

        int num_bytes = x * y * channels * sizeof(float);
        hdri.data.resize(num_bytes);
        memcpy(hdri.data.data(), hdriData, num_bytes);
        textures.push_back(hdri);
        hdriIndex = textures.size() - 1;
    }
    else {
        const char* reason = stbi_failure_reason();
        if (reason) {
            std::cerr << "Failure reason: " << reason << std::endl;
        }
    }

    // Pad 3 channel textures to 4 channels.
    for (Texture& t : textures) {
        if (t.numChannels == 3) {
            int elemSize = t.bitsPerChannel / 8;

            std::vector<uint8_t> paddedData;
            for (int i = 0; i < t.data.size(); ++i) {
                int realIndex = i / elemSize;
                paddedData.push_back(t.data[i]);
                if (i % (3 * elemSize) == 3 * elemSize - 1) {
                    for (int j = 0; j < elemSize; ++j) {
                        paddedData.push_back(0);
                    }
                }

            }
            t.data = paddedData;
            t.numChannels = 4;
        }
    }

    bool cameraSet = false;
    // Loop through all nodes, then through meshes
    for (tinygltf::Node& node : model.nodes) {
        Geom newGeomTemplate;

        // Transforms
        if (node.translation.size()) {
            newGeomTemplate.translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        }
        if (node.rotation.size()) {
                                    // Quat wants WXYZ but gltf is XYZW
            glm::quat rot = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
            newGeomTemplate.rotation = glm::eulerAngles(rot);
            float invPi = 1.f / 3.1415926f;
            newGeomTemplate.rotation.x *= 180.f * invPi;
            newGeomTemplate.rotation.y *= 180.f * invPi;
            newGeomTemplate.rotation.z *= 180.f * invPi;
        }
        if (node.scale.size()) {
            newGeomTemplate.scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        }
        if (node.matrix.size()) {
            std::vector<double> mat = node.matrix;
            newGeomTemplate.transform = glm::mat4( mat[0],  mat[1],  mat[2],  mat[3],
                                           mat[4],  mat[5],  mat[6],  mat[7],
                                           mat[8],  mat[9],  mat[10], mat[11],
                                           mat[12], mat[13], mat[14], mat[15] );
            glm::mat4 otherMat = utilityCore::buildTransformationMatrix(
                newGeomTemplate.translation, newGeomTemplate.rotation, newGeomTemplate.scale);
            // TODO check this works
            assert(otherMat == newGeomTemplate.transform);
        }

        newGeomTemplate.transform = utilityCore::buildTransformationMatrix(
            newGeomTemplate.translation, newGeomTemplate.rotation, newGeomTemplate.scale);
        newGeomTemplate.inverseTransform = glm::inverse(newGeomTemplate.transform);
        newGeomTemplate.invTranspose = glm::inverseTranspose(newGeomTemplate.transform);
        
        // TODO: what happens when the node doesn't have a mesh?
        if (node.mesh != -1) {
            newGeomTemplate.type = MESH;
            tinygltf::Mesh mesh = model.meshes[node.mesh];

            // Design choice - we will push back a mesh for each material slot
            for (tinygltf::Primitive& prim: mesh.primitives) {
                Geom newGeom = newGeomTemplate;
                newGeom.materialid = prim.material;

                assert(prim.mode == 4); // GLTF encoding for TRIANGLES

                if (prim.attributes.count("POSITION") > 0) {
                    int accessorId = prim.attributes["POSITION"];
                    tinygltf::Accessor accessor = model.accessors[accessorId];
                    int bufferViewId = accessor.bufferView;
                    if (accessor.componentType != 5126) { // Unsupported data type, skipping mesh.
                        continue;
                    }
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
                    if (accessor.componentType != 5126) { // Unsupported data type, skipping mesh.
                        continue;
                    }
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
                    if (accessor.componentType != 5126) { // Unsupported data type, skipping mesh.
                        continue;
                    }
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
                    if (accessor.componentType != 5123) { // Unsupported data type, skipping mesh.
                        continue;
                    }
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
                buildBVH(newGeom.mesh);

                geoms.push_back(newGeom);
            }

            // TESTING
            //for (int i = 0; i < newGeom.mesh.posCount; ++i) {
            //    glm::vec3 v = newGeom.mesh.pos[i];
            //    std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            //}
            //for (int i = 0; i < newGeom.mesh.norCount; ++i) {
            //    glm::vec3 v = newGeom.mesh.nor[i];
            //    std::cout << v.r << ", " << v.g << ", " << v.b << std::endl;
            //}
            //for (int i = 0; i < newGeom.mesh.uvCount; ++i) {
            //    glm::vec2 v = newGeom.mesh.uv[i];
            //    std::cout << v.r << ", " << v.g << std::endl;
            //}
            //for (int i = 0; i < newGeom.mesh.indCount; ++i) {
            //    unsigned short s = newGeom.mesh.ind[i];
            //    std::cout << s << std::endl;
            //}

        }

        if (node.camera != -1) {
            cameraSet = true;
            Camera& camera = state.camera;
            tinygltf::Camera cam = model.cameras[node.camera];
            camera.resolution.x = 800;
            camera.resolution.y = 800 / cam.perspective.aspectRatio;
            float fovy = cam.perspective.yfov;
            state.iterations = 5000;
            state.traceDepth = 8;
            state.imageName = model.nodes[0].name; // Weird name
            camera.position = newGeomTemplate.translation;
            camera.view = glm::normalize(glm::vec3(newGeomTemplate.transform * glm::vec4(0.f, 0.f, -1.f, 0.f)));
            camera.up = glm::normalize(glm::vec3(newGeomTemplate.transform * glm::vec4(0.f, 1.f, 0.f, 0.f)));

            //calculate fov based on resolution
            float yscaled = tan(fovy / 2); // GLTF fovy already in Rad, so no need to convert
            float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
            float fovx = (atan(xscaled) * 180) / PI;
            camera.fov = glm::vec2(fovx, fovy);

            float focalPlaneDist = 5.f; // Can we find a way to fix this?
            camera.lookAt = camera.position + camera.view * focalPlaneDist;

            glm::vec3 crossV = glm::cross(camera.view, camera.up);
            camera.right = glm::normalize(glm::cross(camera.view, camera.up));
            camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                2 * yscaled / (float)camera.resolution.y);

        }
    }

    if (!cameraSet) {
        createDefaultCamera(state);
    }

    //set up render camera stuff
    int arraylen = state.camera.resolution.x * state.camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    return true;
}

void Scene::buildBVH(Mesh& mesh) {
    int n = mesh.indCount / 3;
    assert(mesh.indCount % 3 == 0); // TRUE FOR AS LONG AS WE USE GL_TRIANGLES
    assert(mesh.posCount > 0);

    // Queue of tree nodes for processing
    std::deque<CpuBvhNode*> nodesToProcess = std::deque<CpuBvhNode*>();

    // Assemble triangle array
    std::vector<Triangle> rootTri = std::vector<Triangle>();
    for (int i = 0; i < n; ++i) {
        Triangle currTri = Triangle();
        currTri.ind[0] = mesh.ind[3 * i];
        currTri.ind[1] = mesh.ind[3 * i + 1];
        currTri.ind[2] = mesh.ind[3 * i + 2];
        currTri.points[0] = mesh.pos[currTri.ind[0]];
        currTri.points[1] = mesh.pos[currTri.ind[1]];
        currTri.points[2] = mesh.pos[currTri.ind[2]];
        rootTri.push_back(currTri);
    }

    auto [rootMaxBounds, rootMinBounds] = findBBoxOfTris(rootTri);

    uPtr<CpuBvhNode> root = std::make_unique<CpuBvhNode>();

    root->tri = rootTri;
    root->maxBounds = rootMaxBounds;
    root->minBounds = rootMinBounds;

    nodesToProcess.push_back(root.get());

    const int BVH_MAX_LAYERS = 15; // Also change in intersections.cu
    for (int i = 0; i < BVH_MAX_LAYERS; ++i) {
        int layerSize = nodesToProcess.size();;
        for (int j = 0; j < layerSize; ++j) {
            CpuBvhNode* nodePtr = nodesToProcess.at(j);

            auto [axisToSplit, splitPos] = findSplitPoint(nodePtr);

            std::vector<Triangle> bb1tris = std::vector<Triangle>();
            std::vector<Triangle> bb2tris = std::vector<Triangle>();
            assert(nodePtr->tri.size() < 1e9);

            for (auto& currTri : nodePtr->tri) {
                glm::vec3 centroid = (currTri.points[0] + currTri.points[1] + currTri.points[2]) / 3.0f;

                if (centroid[axisToSplit] > splitPos) {
                    // put in 1
                    bb1tris.push_back(currTri);
                }
                else {
                    // put in 2
                    bb2tris.push_back(currTri);
                }

            }
                
            if (bb1tris.size() == 0 || bb2tris.size() == 0) {
                continue;
            }

            auto [bb1MaxBounds, bb1MinBounds] = findBBoxOfTris(bb1tris);
            auto [bb2MaxBounds, bb2MinBounds] = findBBoxOfTris(bb2tris);

            uPtr<CpuBvhNode> lChild = std::make_unique<CpuBvhNode>(bb1tris);
            uPtr<CpuBvhNode> rChild = std::make_unique<CpuBvhNode>(bb2tris);
            lChild->maxBounds = bb1MaxBounds;
            lChild->minBounds = bb1MinBounds;
            rChild->maxBounds = bb2MaxBounds;
            rChild->minBounds = bb2MinBounds;
            
            assert(bb1tris.size() < 1e9);
            assert(bb2tris.size() < 1e9);
            lChild->tri = bb1tris;
            rChild->tri = bb2tris;

            // TODO - can optimize order?
            nodesToProcess.push_back(lChild.get());
            nodesToProcess.push_back(rChild.get());

            nodePtr->lChild = std::move(lChild);
            nodePtr->rChild = std::move(rChild);

            assert(nodePtr->lChild->tri.size() < 1e9);
            assert(nodePtr->rChild->tri.size() < 1e9);
        }
        for (int i = 0; i < layerSize; ++i) {
            nodesToProcess.pop_front();
        }
    }

    // Testing
    //printBVH(root);
    // END TESTING

    // Put in format better for GPU.
    formatBVH(root, mesh);

    //More Testing - Print GPU formatted data.
    bool abridged = true;
    printf("\n\n\n\n");
    printf("    [ ");
    for (int i = 0; i < mesh.numBvhNodes; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = mesh.numBvhNodes - 2;
            printf("... ");
        }

        printf("Node: Left: %hu, Right: %hu, Offset: %i, Length: %i, Bounds %f - %f, %f - %f, %f - %f\n", 
            mesh.bvhNodes[i].leftChild,
            mesh.bvhNodes[i].rightChild,
            mesh.bvhNodes[i].trisOffset,
            mesh.bvhNodes[i].trisLength,
            mesh.bvhNodes[i].minBounds.x,
            mesh.bvhNodes[i].maxBounds.x,
            mesh.bvhNodes[i].minBounds.y,
            mesh.bvhNodes[i].maxBounds.y,
            mesh.bvhNodes[i].minBounds.z,
            mesh.bvhNodes[i].maxBounds.z);
    }
    
    printf("\n");
    for (int i = 0; i < mesh.indCount; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = mesh.indCount - 2;
            printf("... ");
        }

        printf("%hu ", mesh.indBVH[i]);
    }
    printf("]\n");


}


std::pair <glm::vec3, glm::vec3> Scene::findBBoxOfTris(std::vector<Triangle> tris) {
    glm::vec3 bbMaxBounds = glm::vec3(-INFINITY, -INFINITY, -INFINITY);
    glm::vec3 bbMinBounds = glm::vec3(INFINITY, INFINITY, INFINITY);

    for (int j = 0; j < tris.size(); j++) {
        Triangle currTri = tris[j];
        glm::vec3* pts = currTri.points;
        bbMaxBounds = glm::max(bbMaxBounds, pts[0]);
        bbMaxBounds = glm::max(bbMaxBounds, pts[1]);
        bbMaxBounds = glm::max(bbMaxBounds, pts[2]);
        bbMinBounds = glm::min(bbMinBounds, pts[0]);
        bbMinBounds = glm::min(bbMinBounds, pts[1]);
        bbMinBounds = glm::min(bbMinBounds, pts[2]);
    }
    return std::make_pair(bbMaxBounds, bbMinBounds);
}

std::pair <int, float> Scene::findSplitPoint(CpuBvhNode* node) {
    // Find axis to split
    int axisToSplit = 0;
    if (node->maxBounds.y - node->minBounds.y > node->maxBounds.x - node->minBounds.x) {
        axisToSplit = 1;
    }
    if (node->maxBounds.z - node->minBounds.z > node->maxBounds.y - node->minBounds.y &&
        node->maxBounds.z - node->minBounds.z > node->maxBounds.x - node->minBounds.x) {
        axisToSplit = 2;
    }
    // Split bounding boxes - just middle currently
    float splitPos = (node->maxBounds[axisToSplit] + node->minBounds[axisToSplit]) / 2;

    return std::make_pair(axisToSplit, splitPos);
}

void Scene::printBounds(CpuBvhNode* currNode, std::string id) {
    printf(id.c_str());
    printf(" with bounds: %.3f to %.3f, %.3f to %.3f, %.3f to %.3f\n",
        currNode->minBounds[0], currNode->maxBounds[0], currNode->minBounds[1], currNode->maxBounds[1],
        currNode->minBounds[2], currNode->maxBounds[2]);
}

void Scene::printBVH(const uPtr<CpuBvhNode>& root) {
    std::deque<CpuBvhNode*> nodesToPrint = std::deque<CpuBvhNode*>();
    nodesToPrint.push_back(root.get());

    while (!nodesToPrint.empty()) {
        CpuBvhNode* currNode = nodesToPrint[0];
        nodesToPrint.pop_front();
        printf("New Node:\n");
        printBounds(currNode, "Node");
        if (currNode->lChild != nullptr) {
            printBounds(currNode->lChild.get(), "Has left child");
            nodesToPrint.push_back(currNode->lChild.get());
        }
        if (currNode->rChild != nullptr) {
            printBounds(currNode->rChild.get(), "Has right child");
            nodesToPrint.push_back(currNode->rChild.get());
        }

        printf("Tris: ");
        for (Triangle t : currNode->tri) {
            printf("[(%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f)], ",
                t.points[0][0], t.points[0][1], t.points[0][2],
                t.points[1][0], t.points[1][1], t.points[1][2],
                t.points[2][0], t.points[2][1], t.points[2][2]);
        }
        printf("\n");
    }
}

void Scene::formatBVH(const uPtr<CpuBvhNode>& root, Mesh& mesh) {
    mesh.indBVH = new unsigned short[mesh.indCount];

    // Count BVH tree node count;
    int count = 0;
    std::vector<CpuBvhNode*> nodesToCount = std::vector<CpuBvhNode*>();
    nodesToCount.push_back(root.get());

    while (!nodesToCount.empty()) {
        CpuBvhNode* currNode = nodesToCount.back();
        nodesToCount.pop_back();
        count += 1;
        if (currNode->lChild != nullptr) {
            nodesToCount.push_back(currNode->lChild.get());
        }
        if (currNode->rChild != nullptr) {
            nodesToCount.push_back(currNode->rChild.get());
        }
    }

    mesh.numBvhNodes = count;
    mesh.bvhNodes = new BvhNode[count];
    std::vector<int> parents;
    int currIndex = 0; //into mesh.bvhNodes;
    int currTrisOffset = 0;

    // DFS traversal, so stack.
    std::vector<CpuBvhNode*> nodesToProcess = std::vector<CpuBvhNode*>();
    nodesToProcess.push_back(root.get());

    while (!nodesToProcess.empty()) {
        assert(currIndex < count);
        CpuBvhNode* currNode = nodesToProcess.back();
        nodesToProcess.pop_back();
        if (!parents.empty()) {
            int parentIdx = parents.back();
            parents.pop_back();

            if (mesh.bvhNodes[parentIdx].leftChild == 0) {
                mesh.bvhNodes[parentIdx].leftChild = currIndex;
            }
            else {
                // If this fails, more than two nodes have the same parent
                assert(mesh.bvhNodes[parentIdx].rightChild == 0);
                mesh.bvhNodes[parentIdx].rightChild = currIndex;
            }
        }

        BvhNode& currOutNode = mesh.bvhNodes[currIndex];
        currOutNode.maxBounds = currNode->maxBounds;
        currOutNode.minBounds = currNode->minBounds;
        currOutNode.leftChild = 0;
        currOutNode.rightChild = 0;

        if (currNode->lChild == nullptr && currNode->rChild == nullptr) {
            currOutNode.trisOffset = currTrisOffset;
            currOutNode.trisLength = 3 * currNode->tri.size();
           

            for (int i = 0; i < currNode->tri.size(); ++i) {
                Triangle currTri = currNode->tri[i];

                mesh.indBVH[currTrisOffset] = currTri.ind[0];
                mesh.indBVH[currTrisOffset + 1] = currTri.ind[1];
                mesh.indBVH[currTrisOffset + 2] = currTri.ind[2];

                currTrisOffset += 3;
            }
        }
        else {
            currOutNode.trisOffset = -1;
            currOutNode.trisLength = -1;

            // Either both children are null or non-null
            assert(currNode->lChild != nullptr && currNode->rChild != nullptr);

            nodesToProcess.push_back(currNode->lChild.get());
            parents.push_back(currIndex);
            nodesToProcess.push_back(currNode->rChild.get());
            parents.push_back(currIndex);
        }

        currIndex += 1;
    }



}