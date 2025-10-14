#include "RvcAITrainer.h"
#include "RvcAIInterface.h"

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/c/common.h>

#include <iostream>
#include <cmath>
#include <cstring>

RvcAITrainer::RvcAITrainer()
    : mInferenceEngine(nullptr), mInterpreter(nullptr), mLastLayerEvalTensor(nullptr)
{
    std::cout << "RvcAITrainer created." << std::endl;
}

RvcAITrainer::~RvcAITrainer()
{
    std::cout << "RvcAITrainer destroyed." << std::endl;
}

bool RvcAITrainer::AttachInferenceEngine(RvcAIInterface* inference_engine)
{
    if (!inference_engine) {
        std::cerr << "Error: Cannot attach null inference engine." << std::endl;
        return false;
    }

    mInferenceEngine = inference_engine;
    mInterpreter = inference_engine->GetInterpreter();

    if (!mInterpreter) {
        std::cerr << "Error: Interpreter not initialized in inference engine." << std::endl;
        return false;
    }

    std::cout << "Inference engine attached to trainer successfully." << std::endl;
    return LocateLastLayer();
}

bool RvcAITrainer::LocateLastLayer()
{
    if (!mInterpreter) {
        std::cerr << "Error: Interpreter not available. Cannot locate last layer." << std::endl;
        return false;
    }

    // Based on tensor inspection, we know that:
    // - Tensor 47: [80, 3, 3, 128] with 92,160 elements is a large Conv2D weight
    // - This is likely one of the last significant layers in YOLO

    // Try to access Tensor 47 (largest Conv2D weight tensor)
    mLastLayerEvalTensor = mInterpreter->GetTensor(47);

    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Could not access Tensor 47." << std::endl;
        return false;
    }

    std::cout << "Last layer located (Tensor 47 - Conv2D weight). Shape: [";
    for (int i = 0; i < mLastLayerEvalTensor->dims->size; ++i) {
        std::cout << mLastLayerEvalTensor->dims->data[i];
        if (i < mLastLayerEvalTensor->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Type: " << mLastLayerEvalTensor->type << std::endl;

    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < mLastLayerEvalTensor->dims->size; ++i) {
        total_elements *= mLastLayerEvalTensor->dims->data[i];
    }
    std::cout << "Total weight parameters: " << total_elements << std::endl;

    return true;
}

size_t RvcAITrainer::GetLastLayerWeightsCount()
{
    if (!mLastLayerEvalTensor) {
        return 0;
    }

    // Calculate total number of elements in the tensor
    size_t count = 1;
    for (int i = 0; i < mLastLayerEvalTensor->dims->size; ++i) {
        count *= mLastLayerEvalTensor->dims->data[i];
    }

    return count;
}

// ============================================================================
// INT8 Direct Weight Access (Recommended for Quantized Models)
// ============================================================================

bool RvcAITrainer::GetLastLayerWeightsInt8(int8_t* weights_buffer, size_t* buffer_size)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Last layer not located yet." << std::endl;
        return false;
    }

    size_t weight_count = GetLastLayerWeightsCount();
    *buffer_size = weight_count;

    if (!weights_buffer) {
        // Caller just wants to know the size
        return true;
    }

    // Type 9 is kTfLiteInt8
    if (mLastLayerEvalTensor->type == 9) {  // kTfLiteInt8
        int8_t* tensor_data = mLastLayerEvalTensor->data.int8;

        // Direct copy - no conversion!
        std::memcpy(weights_buffer, tensor_data, weight_count * sizeof(int8_t));

        std::cout << "Extracted " << weight_count << " int8 weights (direct copy)." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Unsupported tensor type: " << mLastLayerEvalTensor->type << std::endl;
        return false;
    }
}

bool RvcAITrainer::SetLastLayerWeightsInt8(const int8_t* weights, size_t size)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Last layer not located yet." << std::endl;
        return false;
    }

    size_t expected_size = GetLastLayerWeightsCount();
    if (size != expected_size) {
        std::cerr << "Error: Weight size mismatch. Expected " << expected_size
                  << " but got " << size << std::endl;
        return false;
    }

    // Type 9 is kTfLiteInt8
    if (mLastLayerEvalTensor->type == 9) {  // kTfLiteInt8
        int8_t* tensor_data = mLastLayerEvalTensor->data.int8;

        // Direct copy - no conversion!
        std::memcpy(tensor_data, weights, size * sizeof(int8_t));

        std::cout << "Updated " << size << " int8 weights (direct copy)." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Unsupported tensor type: " << mLastLayerEvalTensor->type << std::endl;
        return false;
    }
}

bool RvcAITrainer::UpdateWeightsInt8(const int8_t* gradients, size_t size)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Last layer not located yet." << std::endl;
        return false;
    }

    size_t weight_count = GetLastLayerWeightsCount();
    if (size != weight_count) {
        std::cerr << "Error: Gradient size mismatch. Expected " << weight_count
                  << " but got " << size << std::endl;
        return false;
    }

    // Type 9 is kTfLiteInt8
    if (mLastLayerEvalTensor->type == 9) {  // kTfLiteInt8
        int8_t* tensor_data = mLastLayerEvalTensor->data.int8;

        // Direct int8 update: weight -= gradient
        // This avoids float conversion and precision loss!
        for (size_t i = 0; i < weight_count; ++i) {
            // Compute new weight
            int16_t new_weight = static_cast<int16_t>(tensor_data[i]) - static_cast<int16_t>(gradients[i]);

            // Clamp to int8 range [-128, 127]
            if (new_weight > 127) new_weight = 127;
            if (new_weight < -128) new_weight = -128;

            tensor_data[i] = static_cast<int8_t>(new_weight);
        }

        std::cout << "Updated " << weight_count << " int8 weights (direct int8 arithmetic)." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Unsupported tensor type: " << mLastLayerEvalTensor->type << std::endl;
        return false;
    }
}

// ============================================================================
// Legacy Float Weight Access (Lossy - Not Recommended)
// ============================================================================

bool RvcAITrainer::GetLastLayerWeights(float* weights_buffer, size_t* buffer_size)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Last layer not located yet." << std::endl;
        return false;
    }

    size_t weight_count = GetLastLayerWeightsCount();
    *buffer_size = weight_count;

    if (!weights_buffer) {
        // Caller just wants to know the size
        return true;
    }

    // Type 9 is kTfLiteInt8
    if (mLastLayerEvalTensor->type == 9) {  // kTfLiteInt8
        int8_t* tensor_data = mLastLayerEvalTensor->data.int8;

        // Simple dequantization: convert int8 to float
        // Note: Without quantization params, this is approximate
        // We use a simple scale of 1/127.0 (assuming symmetric quantization)
        for (size_t i = 0; i < weight_count; ++i) {
            weights_buffer[i] = static_cast<float>(tensor_data[i]) / 127.0f;
        }
        std::cout << "Extracted " << weight_count << " int8 weights (approximate dequantization)." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Unsupported tensor type: " << mLastLayerEvalTensor->type << std::endl;
        return false;
    }
}

bool RvcAITrainer::SetLastLayerWeights(const float* weights, size_t size)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Last layer not located yet." << std::endl;
        return false;
    }

    size_t expected_size = GetLastLayerWeightsCount();
    if (size != expected_size) {
        std::cerr << "Error: Weight size mismatch. Expected " << expected_size
                  << " but got " << size << std::endl;
        return false;
    }

    // Type 9 is kTfLiteInt8
    if (mLastLayerEvalTensor->type == 9) {  // kTfLiteInt8
        int8_t* tensor_data = mLastLayerEvalTensor->data.int8;

        // Simple quantization: convert float to int8
        // Note: Without quantization params, this is approximate
        // We use a simple scale of 127.0 (assuming symmetric quantization)
        for (size_t i = 0; i < size; ++i) {
            float clamped = std::max(-1.0f, std::min(1.0f, weights[i]));
            tensor_data[i] = static_cast<int8_t>(std::round(clamped * 127.0f));
        }
        std::cout << "Updated " << size << " int8 weights (approximate quantization)." << std::endl;
        return true;
    }
    else {
        std::cerr << "Error: Unsupported tensor type: " << mLastLayerEvalTensor->type << std::endl;
        return false;
    }
}

float RvcAITrainer::ComputeLoss(const float* predictions, const float* ground_truth, size_t num_elements)
{
    if (!predictions || !ground_truth) {
        std::cerr << "Error: Null pointers in ComputeLoss." << std::endl;
        return -1.0f;
    }

    // Mean Squared Error (MSE) Loss
    float total_loss = 0.0f;
    for (size_t i = 0; i < num_elements; ++i) {
        float diff = predictions[i] - ground_truth[i];
        total_loss += diff * diff;
    }

    return total_loss / static_cast<float>(num_elements);
}

void RvcAITrainer::ComputeGradients(const float* predictions, const float* ground_truth, size_t num_elements)
{
    // Gradient of MSE loss: dL/dy = 2 * (prediction - ground_truth) / n
    // Simplified: (prediction - ground_truth) / n (omitting constant 2)

    mGradients.resize(num_elements);

    for (size_t i = 0; i < num_elements; ++i) {
        mGradients[i] = (predictions[i] - ground_truth[i]) / static_cast<float>(num_elements);
    }

    std::cout << "Gradients computed for " << num_elements << " elements." << std::endl;
}

void RvcAITrainer::UpdateWeightsWithSGD(float learning_rate)
{
    if (!mLastLayerEvalTensor) {
        std::cerr << "Error: Cannot update weights - last layer not located." << std::endl;
        return;
    }

    if (mGradients.empty()) {
        std::cerr << "Error: No gradients computed yet." << std::endl;
        return;
    }

    size_t weight_count = GetLastLayerWeightsCount();
    if (mGradients.size() != weight_count) {
        std::cerr << "Error: Gradient size mismatch. Expected " << weight_count
                  << " but got " << mGradients.size() << std::endl;
        return;
    }

    // Get current weights
    std::vector<float> current_weights(weight_count);
    size_t buffer_size;
    if (!GetLastLayerWeights(current_weights.data(), &buffer_size)) {
        std::cerr << "Error: Failed to get current weights." << std::endl;
        return;
    }

    // Apply SGD: weight = weight - learning_rate * gradient
    for (size_t i = 0; i < weight_count; ++i) {
        current_weights[i] -= learning_rate * mGradients[i];
    }

    // Set updated weights back
    if (!SetLastLayerWeights(current_weights.data(), weight_count)) {
        std::cerr << "Error: Failed to set updated weights." << std::endl;
        return;
    }

    std::cout << "Weights updated successfully with learning rate: " << learning_rate << std::endl;
}

bool RvcAITrainer::TrainSingleStep(const float* input_image, const float* ground_truth_boxes,
                                    size_t num_boxes, float learning_rate)
{
    if (!mInferenceEngine) {
        std::cerr << "Error: No inference engine attached." << std::endl;
        return false;
    }

    // Step 1: Run forward pass (inference) to get predictions
    // TODO: Get predictions from inference engine
    // This requires exposing output tensor from RvcAIInterface

    // Step 2: Compute loss
    // float loss = ComputeLoss(predictions, ground_truth_boxes, num_elements);
    // std::cout << "Loss: " << loss << std::endl;

    // Step 3: Compute gradients
    // ComputeGradients(predictions, ground_truth_boxes, num_elements);

    // Step 4: Update weights
    // UpdateWeightsWithSGD(learning_rate);

    std::cout << "TrainSingleStep: Framework ready, but needs RvcAIInterface integration." << std::endl;
    return true;
}
