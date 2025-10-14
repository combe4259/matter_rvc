#pragma once

#include <cstdint>
#include <vector>

// Forward declaration
class RvcAIInterface;
namespace tflite {
    class MicroInterpreter;
}
struct TfLiteTensor;
struct TfLiteEvalTensor;

/**
 * @brief Transfer Learning trainer for YOLO last layer
 *
 * This class implements on-device training by:
 * 1. Freezing all backbone layers (feature extraction)
 * 2. Training only the last detection head
 * 3. Using simple SGD optimizer
 * 4. Computing gradients manually (TFLM doesn't support backprop)
 */
class RvcAITrainer {
public:
    RvcAITrainer();
    ~RvcAITrainer();

    // Link this trainer to an inference engine
    bool AttachInferenceEngine(RvcAIInterface* inference_engine);

    // Weight management for Federated Learning (int8 version)
    bool GetLastLayerWeightsInt8(int8_t* weights_buffer, size_t* buffer_size);
    bool SetLastLayerWeightsInt8(const int8_t* weights, size_t size);
    bool UpdateWeightsInt8(const int8_t* gradients, size_t size);
    size_t GetLastLayerWeightsCount();

    // Legacy float version (for compatibility, but lossy)
    bool GetLastLayerWeights(float* weights_buffer, size_t* buffer_size);
    bool SetLastLayerWeights(const float* weights, size_t size);

    // Training functions
    bool TrainSingleStep(const float* input_image, const float* ground_truth_boxes,
                         size_t num_boxes, float learning_rate);

    // Loss computation
    float ComputeLoss(const float* predictions, const float* ground_truth, size_t num_elements);

private:
    RvcAIInterface* mInferenceEngine;
    tflite::MicroInterpreter* mInterpreter; // Borrowed from RvcAIInterface
    TfLiteEvalTensor* mLastLayerEvalTensor; // Pointer to the last layer's weights (using EvalTensor)

    // Gradient storage for last layer (int8 for quantized training)
    std::vector<int8_t> mGradientsInt8;
    std::vector<float> mGradients; // Legacy float gradients

    // Helper functions for backpropagation
    void ComputeGradients(const float* predictions, const float* ground_truth, size_t num_elements);
    void UpdateWeightsWithSGD(float learning_rate);

    // Extract last layer info from the model
    bool LocateLastLayer();
};
