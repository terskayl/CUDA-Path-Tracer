#pragma once

#include "sceneStructs.h"
#include <vector>
#include <memory>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    bool loadFromGLTF(const std::string& gltfName, bool isBinary);

    struct Triangle {
        glm::vec3 points[3];
        int ind[3];
    };

    struct CpuBvhNode {
        std::vector<Triangle> tri;
        std::unique_ptr<CpuBvhNode> lChild;
        std::unique_ptr<CpuBvhNode> rChild;
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

    // Helper functions
    void buildBVH(Mesh& mesh);
               // max    // min
    std::pair<glm::vec3, glm::vec3> findBBoxOfTris(std::vector<Triangle> tris);
          //axisToSplit, pos
    std::pair<int, float> findSplitPoint(CpuBvhNode* node);

    // Testing
    void printBounds(CpuBvhNode* currNode, std::string id);
    void printBVH(const std::unique_ptr<CpuBvhNode>& root);
    void formatBVH(const std::unique_ptr<CpuBvhNode>& root, Mesh& mesh);
public:
    Scene(std::string filename);
    Scene(std::string filename, std::string imageName, std::string renderStateJson);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    int hdriIndex = -1;
    RenderState state;
};
