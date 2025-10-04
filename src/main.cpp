#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"
#include "intersections.h"
#include "json.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include "tiny_gltf.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#define TESTS 1

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

// Forward declarations for window loop and interactivity
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    mouseOverImGuiWinow = io->WantCaptureMouse;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    if (ImGui::CollapsingHeader("Options", ImGuiTreeNodeFlags_None))
    {
        ImGui::Checkbox("StreamCompaction", &renderState->doStreamCompaction);
        ImGui::Checkbox("MaterialSorting", &renderState->doMaterialSorting);
        ImGui::Checkbox("BVH", &renderState->doBVH);
        ImGui::Checkbox("ACES", &renderState->doACES);
        ImGui::Checkbox("Reinhard", &renderState->doReinhard);
        ImGui::Checkbox("GammaCorrection", &renderState->doGammaCorrection);
        ImGui::Checkbox("RussianRoulette", &renderState->doRussianRoulette);
        ImGui::Checkbox("Denoising", &renderState->doDenoising);
        ImGui::Checkbox("DenoisingOutput", &renderState->doDenoisingOutput);
    }


    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
    return mouseOverImGuiWinow;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//------------TESTS--------------
//-------------------------------

void triangleTest() {
    Ray r = Ray();
    r.origin = glm::vec3(0, 0, 10);
    r.direction = glm::vec3(0, 0, -1);

    glm::vec3 intersectionPoint;
    glm::vec3 normal, baryWeights;
    bool notBackface;
    // NOTE CCW
    float t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t != -1);

    printf("Intersection Point: %f, %f, %f. Normal: %f, %f, %f. Frontface?: %i. t=%f",
        intersectionPoint[0], intersectionPoint[1], intersectionPoint[2],
        normal[0], normal[1], normal[2], notBackface, t);


    r.direction = glm::vec3(0, 0, 1);
    t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t == -1);

    r.direction = glm::vec3(0, 1, 0);
    t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t == -1);

    r.direction = glm::vec3(1, 0, 0);
    t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t == -1);

    r.direction = glm::normalize(glm::vec3(1, 0, 1));
    t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t == -1);

    r.origin = glm::vec3(0, 0, 0.5);
    r.direction = glm::normalize(glm::vec3(1, 0, -1));
    t = triangleIntersectionTest(glm::vec3(-1, -1, 0), glm::vec3(1, 0, 0), glm::vec3(-1, 1, 0), r,
        intersectionPoint, normal, baryWeights, notBackface);
    assert(t != -1);
}

void triangleAngleTest() {
    Ray r;

    glm::vec3 intersectionPoint, normal, baryWeights;
    bool notBackface;

    // Define a triangle lying flat in the XY plane
    glm::vec3 v0(-1, -1, 0);
    glm::vec3 v1(1, -1, 0);
    glm::vec3 v2(0, 1, 0);

    r.origin = glm::vec3(0, 0, 1.0f); // Start above the triangle

    int hits = 0;
    int misses = 0;

    // Sweep directions in a hemisphere
    std::cout << std::endl;
    for (int thetaDeg = -80; thetaDeg <= 80; thetaDeg += 5) {   // pitch
        for (int phiDeg = 0; phiDeg < 360; phiDeg += 5) {       // yaw
            float theta = glm::radians((float)thetaDeg);
            float phi = glm::radians((float)phiDeg);

            // Spherical to Cartesian (unit vector)
            r.direction = glm::normalize(glm::vec3(
                cos(theta) * cos(phi),
                cos(theta) * sin(phi),
                sin(theta)
            ));

            float t = triangleIntersectionTest(v0, v1, v2, r,
                intersectionPoint, normal, baryWeights, notBackface);

            if (t > 0) {
                hits++;
                // Sanity check: intersection should be close to z = 0
                assert(fabs(intersectionPoint.z) < 1e-4);
                std::cout << "hits ";
            }
            else {
                misses++;
                std::cout << "miss ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Total rays tested: " << (hits + misses) << "\n";
    std::cout << "Hits: " << hits << ", Misses: " << misses << "\n";

}

void trianglePositionTest() {
    Ray r;

    glm::vec3 intersectionPoint, normal, baryWeights;
    bool notBackface;
    // Define a triangle lying flat in the XY plane
    glm::vec3 v0(-1, -1, 0);
    glm::vec3 v1(1, -1, 0);
    glm::vec3 v2(0, 1, 0);

    int hits = 0;
    int misses = 0;

    std::cout << std::endl;
    for (float thetaDeg = -2; thetaDeg <= 2; thetaDeg += 0.1) {   // pitch
        for (float phiDeg = -2; phiDeg <= 2; phiDeg += 0.1) {       // yaw
            float theta = glm::radians((float)thetaDeg);
            float phi = glm::radians((float)phiDeg);

            // Spherical to Cartesian (unit vector)
            r.direction = glm::vec3(0.f, 0.f, -1.f);
            r.origin = glm::vec3(thetaDeg, phiDeg, 1);

            float t = triangleIntersectionTest(v0, v1, v2, r,
                intersectionPoint, normal, baryWeights, notBackface);

            if (t > 0) {
                hits++;
                // Sanity check: intersection should be close to z = 0
                assert(fabs(intersectionPoint.z) < 1e-4);
                std::cout << "O ";
            }
            else {
                misses++;
                std::cout << "X ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Total rays tested: " << (hits + misses) << "\n";
    std::cout << "Hits: " << hits << ", Misses: " << misses << "\n";
}

void triangleSpeedTest() {

    Ray r = Ray();
    r.origin = glm::vec3(0, 0, 10);
    r.direction = glm::vec3(0, 0, -1);

    glm::vec3 intersectionPoint;
    glm::vec3 normal, baryWeights;
    bool notBackface;
    //run 1 million times.
    std::chrono::high_resolution_clock::time_point time_start_cpu = std::chrono::high_resolution_clock::now();


    for (int i = 0; i < 1000000; ++i) {
        r.direction = glm::normalize(glm::vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX)));
        if (glm::length(r.direction) - 1 > 0.01 || !r.direction[0]) {
            r.direction = glm::vec3(1, 0, 0);
        }
        r.origin = glm::vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        float t = triangleIntersectionTest(glm::vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX)),
            glm::vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX)),
            glm::vec3(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX)),
            r, intersectionPoint, normal, baryWeights, notBackface);
    }

    std::string endTimeString = currentTimeString();

    std::chrono::high_resolution_clock::time_point time_end_cpu = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
    float prev_elapsed_time_cpu_milliseconds =
        static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

    printf("\n");
    printf("Time Taken: %fms", prev_elapsed_time_cpu_milliseconds);
    printf("\n");

}

void bboxPositionTest() {
    Ray r;

    //glm::vec3 intersectionPoint, normal;
    //bool notBackface;

    int hits = 0;
    int misses = 0;

    std::cout << std::endl;
    for (float thetaDeg = -2; thetaDeg <= 2; thetaDeg += 0.1) {   // pitch
        for (float phiDeg = -2; phiDeg <= 2; phiDeg += 0.1) {       // yaw
            float theta = glm::radians((float)thetaDeg);
            float phi = glm::radians((float)phiDeg);

            // Spherical to Cartesian (unit vector)
            r.direction = glm::vec3(0.f, 0.f, -1.f);
            r.origin = glm::vec3(thetaDeg, phiDeg, 1);

            float t = bboxIntersectionTest(r, glm::vec3(-2, -2, -1), glm::vec3(-1.5, -1, 1));

            if (t > 0) {
                hits++;
                // Sanity check: intersection should be close to z = 0
                //assert(fabs(intersectionPoint.z) < 1e-4);
                std::cout << "O ";
            }
            else {
                misses++;
                std::cout << "X ";
            }
        }
        std::cout << std::endl;
    }

    std::cout << "Total rays tested: " << (hits + misses) << "\n";
    std::cout << "Hits: " << hits << ", Misses: " << misses << "\n";
}

// Will fail if first Geom is not a mesh
void bvhTraversalTest(Scene* scene) {
    Ray r = Ray();
    r.origin = glm::vec3(0, 3, 10);
    r.direction = glm::vec3(0, 0, -1);

    glm::vec3 intersectionPoint;
    glm::vec3 normal;
    glm::vec2 uv;
    bool outside;

    meshIntersectionTestBVH(scene->geoms[0], r, intersectionPoint, normal, uv, glm::vec3(), glm::vec3(), outside);
}

// Needs copy of sampleAndResolveSpecularTrans from interactions.cu to run
void refractTest() {
    Ray r;
    r.origin = glm::vec3(0.8f, 4.0, -1.f); // Start inside the sphere

    // Sweep directions in a hemisphere
    std::cout << std::endl;
    for (int thetaDeg = -80; thetaDeg <= 80; thetaDeg += 5) {   // pitch
        for (int phiDeg = 180; phiDeg < 185; phiDeg += 5) {       // yaw
            float theta = glm::radians((float)thetaDeg);
            float phi = glm::radians((float)phiDeg);

            // Spherical to Cartesian (unit vector)
            r.direction = glm::normalize(glm::vec3(
                cos(theta) * cos(phi),
                cos(theta) * sin(phi),
                sin(theta)
            ));
            glm::vec3 v1 = glm::refract(r.direction, glm::vec3(1, 0, 0), 1.f / 1.44f);

            //printf("Input: (%.2f, %.2f, %.2f), Output: (%.2f, %.2f, %.2f)\n", r.direction.x, r.direction.y, r.direction.z, v1.x, v1.y, v1.z);

            //printf("backwards\n");
            //glm::vec3 u = glm::refract(v, glm::vec3(0, 0, 1), 1.44f);
            //printf("Input: (%.2f, %.2f, %.2f), Output: (%.2f, %.2f, %.2f)\n", r.direction.x, r.direction.y, r.direction.z, u.x, u.y, u.z);

            PathSegment path;
            path.ray = r;
            Ray& ray = path.ray;
            path.remainingBounces = 1;

            
            glm::vec3 intersect, normal;
            Material m;
            bool outside;

            glm::vec3 v = ray.direction;

            assert(scene->geoms[6].type == SPHERE);
            int t = sphereIntersectionTest(scene->geoms[6], ray, intersect, normal, outside);
            if (t != -1) {

                //sampleAndResolveSpecularTrans(path, intersect, normal, m);
                //printf("Input: (%.2f, %.2f, %.2f), Output: (%.2f, %.2f, %.2f)\n", v.x, v.y, v.z, ray.direction.x, ray.direction.y, ray.direction.z);
                // Entering sphere so -normal

                ray.origin = intersect + normal * 1e-3f * (dot(ray.direction, normal) < 0 ? -1.f : 1.f);

                glm::vec3 newIntersect;
                t = sphereIntersectionTest(scene->geoms[6], ray, newIntersect, normal, outside);
                
                //printf("Normal:  (%.2f, %.2f, %.2f)\n", normal.x, normal.y, normal.z);
                //printf("Prev iSect: (%.2f, %.2f, %.2f), New iSect: (%.2f, %.2f, %.2f)\n", intersect.x, intersect.y, intersect.z, newIntersect.x, newIntersect.y, newIntersect.z);

                if (t != -1) {
                    //sampleAndResolveSpecularTrans(path, intersect, normal, m);
                    //printf("Input: (%.2f, %.2f, %.2f), Second Output: (%.2f, %.2f, %.2f)\n", v.x, v.y, v.z, ray.direction.x, ray.direction.y, ray.direction.z);
                    
                    glm::vec3 newNewIntersect;
                    ray.origin = newIntersect + normal * 1e-3f * (dot(ray.direction, normal) < 0 ? -1.f : 1.f);

                    t = sphereIntersectionTest(scene->geoms[6], ray, newNewIntersect, normal, outside);

                    //printf("Normal:  (%.2f, %.2f, %.2f)\n", normal.x, normal.y, normal.z);
                    printf("Prev iSect: (%.2f, %.2f, %.2f), Third iSect: (%.2f, %.2f, %.2f)\n", newIntersect.x, newIntersect.y, newIntersect.z, newNewIntersect.x, newNewIntersect.y, newNewIntersect.z);

                
                
                }
                else {
                    printf("Input: (%.2f, %.2f, %.2f), Output: 0 \n", v.x, v.y, v.z);
                }

            }
            else {
                printf("Input: (%.2f, %.2f, %.2f), Output: 0 \n", v.x, v.y, v.z);
            }


        }
    }
}
//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    // Testing Area
    #if TESTS
    triangleTest();
    //triangleAngleTest();
    //trianglePositionTest();
    //triangleSpeedTest();
    //bboxPositionTest();
    #endif  

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.gtlf\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];
    if (argc < 4) {
        // Load scene file
        scene = new Scene(sceneFile);
    }
    else {
        const char* imageFile = argv[2];
        const char* renderStateJson = argv[3];
        scene = new Scene(sceneFile, imageFile, renderStateJson);
    }


    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = scene->state.currIteration;

    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // SCENE GEO TESTS
#if TESTS
    //bvhTraversalTest(scene);
    //refractTest();

#endif

    // Checkpointing
    if (scene->state.currIteration > 0) {
        camchanged = false;
        printf("Iteration loaded: %ui", scene->state.currIteration);
        pathtraceInit(scene);
    }
    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

inline void to_json(nlohmann::json& j, const Camera& c) {
    j = nlohmann::json{
        { "resolution",    { c.resolution.x, c.resolution.y } },
        { "position",      { c.position.x, c.position.y, c.position.z } },
        { "lookAt",        { c.lookAt.x, c.lookAt.y, c.lookAt.z } },
        { "view",          { c.view.x, c.view.y, c.view.z } },
        { "up",            { c.up.x, c.up.y, c.up.z } },
        { "right",         { c.right.x, c.right.y, c.right.z } },
        { "fov",           { c.fov.x, c.fov.y } },
        { "pixelLength",   { c.pixelLength.x, c.pixelLength.y } }
    };
}

inline void to_json(nlohmann::json& j, const RenderState& r) {
    j = nlohmann::json{
        { "camera",      r.camera },
        { "iterations",  r.iterations },
        { "currIteration",  r.currIteration },
        { "traceDepth",  r.traceDepth },
        { "imageName",   r.imageName }
    };
}
void saveImageCheckpoint()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    //img.savePNG(filename);
    img.saveHDR(filename);

    ss << ".json";
    filename = ss.str();
    nlohmann::json j = scene->state;

    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    file << j.dump(4); // pretty print with indent of 4 spaces
    std::cout << "Saved " + filename + "." << std::endl;

}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        renderState->currIteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceFree(scene);
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        scene->state.currIteration = iteration;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    }
    else
    {
        saveImage();
        pathtraceFree(scene);
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_S:
                saveImage();
                break;
            case GLFW_KEY_C:
                saveImageCheckpoint();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                Camera& cam = renderState->camera;
                cam.lookAt = ogLookAt;
                break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (MouseOverImGuiWindow())
    {
        return;
    }

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
        return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
