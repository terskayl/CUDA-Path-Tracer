#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree(Scene *scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);

void launchPostProcess(RenderState* renderState, int samples, int width, int height, std::vector<glm::vec3>& output);
