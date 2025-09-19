#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool loadFromGLTF(const std::string& gltfName, bool isBinary);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
