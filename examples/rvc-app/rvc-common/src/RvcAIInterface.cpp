#include "RvcAIInterface.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

#include <iostream>
#include <vector>
#include <algorithm> // For std::max_element

// Simple structure to hold detection results
struct BoundingBox {
    float x1, y1, x2, y2; // Top-left and bottom-right coordinates
    float score;
    int class_id;
};

RvcAIInterface::RvcAIInterface()
    : mModel(nullptr), mInterpreter(nullptr), mInputTensor(nullptr), mOutputTensor(nullptr), mResolver(nullptr), mTensorArena(nullptr)
{
}

RvcAIInterface::~RvcAIInterface()
{
}

bool RvcAIInterface::InitAI()
{
    std::cout << "Initializing TFLM..." << std::endl;

    mModel = tflite::GetModel(yolov8n_full_integer_quant_tflite);
    if (mModel->version() != TFLITE_SCHEMA_VERSION) { std::cerr << "Error: Model schema version mismatch." << std::endl; return false; }

    mResolver = std::make_unique<tflite::MicroMutableOpResolver<10>>();
    mResolver->AddConv2D();
    mResolver->AddDepthwiseConv2D();
    mResolver->AddFullyConnected();
    mResolver->AddMaxPool2D();
    mResolver->AddSoftmax();
    mResolver->AddAdd();
    mResolver->AddReshape();
    mResolver->AddDequantize();
    mTensorArena = std::make_unique<uint8_t[]>(kTensorArenaSize);
    if (!mTensorArena) { std::cerr << "Error: Failed to allocate tensor arena." << std::endl; return false; }

    mInterpreter = std::make_unique<tflite::MicroInterpreter>(mModel, *mResolver, mTensorArena.get(), kTensorArenaSize);
    if (mInterpreter->AllocateTensors() != kTfLiteOk) { std::cerr << "Error: AllocateTensors() failed." << std::endl; return false; }

    mInputTensor = mInterpreter->input(0);
    mOutputTensor = mInterpreter->output(0);

    if (!mInputTensor || !mOutputTensor) { std::cerr << "Error: Failed to get input or output tensor." << std::endl; return false; }

    std::cout << "TFLM Initialization Successful!" << std::endl;
    return true;
}

void RvcAIInterface::RunInferenceLoop()
{
    if (!mInterpreter || !mInputTensor) { std::cerr << "Error: Interpreter not initialized." << std::endl; return; }

    // 1. Decode Image
    std::string image_path = "test_data/000000000009.jpg";
    int original_width, original_height, original_channels;
    unsigned char *img_original = stbi_load(image_path.c_str(), &original_width, &original_height, &original_channels, 3);
    if (img_original == nullptr) { std::cerr << "Error: Failed to load image: " << image_path << std::endl; return; }

    // 2. Resize Image
    int target_height = mInputTensor->dims->data[1];
    int target_width = mInputTensor->dims->data[2];
    int target_channels = mInputTensor->dims->data[3];
    std::vector<uint8_t> img_resized(target_height * target_width * target_channels);
    stbir_resize_uint8_srgb(img_original, original_width, original_height, 0, img_resized.data(), target_width, target_height, 0, STBIR_RGB);
    stbi_image_free(img_original);

    // 3. Quantize and copy to input tensor
    for (size_t i = 0; i < img_resized.size(); ++i) {
        mInputTensor->data.int8[i] = (int8_t)(img_resized[i] - 128);
    }

    // 4. Run Inference
    if (mInterpreter->Invoke() != kTfLiteOk) { std::cerr << "Error: Invoke() failed." << std::endl; return; }

    // 5. Post-process: Decode the output tensor
    const float confidence_threshold = 0.5f;
    float output_scale = mOutputTensor->params.scale;
    int output_zero_point = mOutputTensor->params.zero_point;
    int num_detections = mOutputTensor->dims->data[1]; // e.g., 8400
    int num_classes = mOutputTensor->dims->data[2] - 4; // e.g., 84 - 4 = 80

    std::vector<BoundingBox> detected_boxes;

    for (int i = 0; i < num_detections; ++i) {
        // Dequantize the bounding box and class scores
        float cx = (mOutputTensor->data.int8[i * (num_classes + 4) + 0] - output_zero_point) * output_scale;
        float cy = (mOutputTensor->data.int8[i * (num_classes + 4) + 1] - output_zero_point) * output_scale;
        float w = (mOutputTensor->data.int8[i * (num_classes + 4) + 2] - output_zero_point) * output_scale;
        float h = (mOutputTensor->data.int8[i * (num_classes + 4) + 3] - output_zero_point) * output_scale;

        // Find the class with the highest score
        float max_score = -1.0f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            float score = (mOutputTensor->data.int8[i * (num_classes + 4) + 4 + j] - output_zero_point) * output_scale;
            if (score > max_score) {
                max_score = score;
                class_id = j;
            }
        }

        if (max_score > confidence_threshold) {
            BoundingBox box;
            box.x1 = (cx - w / 2) * original_width;
            box.y1 = (cy - h / 2) * original_height;
            box.x2 = (cx + w / 2) * original_width;
            box.y2 = (cy + h / 2) * original_height;
            box.score = max_score;
            box.class_id = class_id;
            detected_boxes.push_back(box);
        }
    }

    std::cout << "--- Found " << detected_boxes.size() << " candidate boxes (before NMS) ---" << std::endl;
    for (const auto& box : detected_boxes) {
        std::cout << "Class " << box.class_id << ": Score=" << box.score 
                  << ", Box=[" << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << "]" << std::endl;
    }

    // TODO: 6. Apply Non-Maximum Suppression (NMS) to `detected_boxes` to get final results.
    // TODO: 7. Based on final results, call RvcDevice functions.
}
