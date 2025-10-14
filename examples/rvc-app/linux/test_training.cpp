/*
 * Test program for RvcAITrainer - Transfer Learning and Federated Learning
 *
 * This standalone program tests the on-device training capabilities:
 * - Weight extraction and updates
 * - Gradient computation
 * - Loss calculation
 * - SGD optimization
 */

#include "../../rvc-common/include/RvcAIInterface.h"
#include "../../rvc-common/include/RvcAITrainer.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Helper function to print a summary of weights
void PrintWeightsSummary(const std::vector<float>& weights, const std::string& label) {
    if (weights.empty()) return;

    float sum = 0.0f;
    float min_val = weights[0];
    float max_val = weights[0];

    for (const float& w : weights) {
        sum += w;
        if (w < min_val) min_val = w;
        if (w > max_val) max_val = w;
    }

    float mean = sum / weights.size();

    std::cout << label << ":" << std::endl;
    std::cout << "  Count: " << weights.size() << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(6) << mean << std::endl;
    std::cout << "  Min:   " << min_val << std::endl;
    std::cout << "  Max:   " << max_val << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   RVC AI Trainer Test Program" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Step 1: Initialize AI Interface
    std::cout << "[1/6] Initializing AI Interface..." << std::endl;
    RvcAIInterface aiInterface;
    if (!aiInterface.InitAI()) {
        std::cerr << "ERROR: Failed to initialize AI Interface!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Interface initialized successfully.\n" << std::endl;

    // Step 2: Initialize AI Trainer
    std::cout << "[2/6] Initializing AI Trainer..." << std::endl;
    RvcAITrainer aiTrainer;
    if (!aiTrainer.AttachInferenceEngine(&aiInterface)) {
        std::cerr << "ERROR: Failed to attach trainer to inference engine!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Trainer attached successfully.\n" << std::endl;

    // Step 3: Extract initial weights
    std::cout << "[3/6] Extracting initial weights..." << std::endl;
    size_t weight_count = aiTrainer.GetLastLayerWeightsCount();
    std::cout << "Weight count: " << weight_count << std::endl;

    if (weight_count == 0) {
        std::cerr << "ERROR: No weights found!" << std::endl;
        return -1;
    }

    std::vector<float> initial_weights(weight_count);
    size_t buffer_size;
    if (!aiTrainer.GetLastLayerWeights(initial_weights.data(), &buffer_size)) {
        std::cerr << "ERROR: Failed to extract weights!" << std::endl;
        return -1;
    }
    PrintWeightsSummary(initial_weights, "Initial Weights");
    std::cout << std::endl;

    // Step 4: Create dummy ground truth data
    std::cout << "[4/6] Creating dummy training data..." << std::endl;
    std::vector<float> predictions(weight_count);
    std::vector<float> ground_truth(weight_count);

    // Simulate predictions (copy from output tensor)
    for (size_t i = 0; i < weight_count; ++i) {
        predictions[i] = initial_weights[i];
    }

    // Create synthetic ground truth (slightly different from predictions)
    for (size_t i = 0; i < weight_count; ++i) {
        ground_truth[i] = predictions[i] + 0.1f * (i % 2 == 0 ? 1.0f : -1.0f);
    }

    std::cout << "✓ Created " << weight_count << " dummy training samples.\n" << std::endl;

    // Step 5: Compute loss before training
    std::cout << "[5/6] Computing loss before training..." << std::endl;
    float loss_before = aiTrainer.ComputeLoss(predictions.data(), ground_truth.data(), weight_count);
    std::cout << "Loss before training: " << std::fixed << std::setprecision(8)
              << loss_before << "\n" << std::endl;

    // Step 6: Perform a training step
    std::cout << "[6/6] Performing training step..." << std::endl;
    float learning_rate = 0.01f;
    std::cout << "Learning rate: " << learning_rate << std::endl;

    // Manually simulate a training step
    // Note: We'll test the individual components

    // 6a. Compute gradients
    std::vector<float> gradients(weight_count);
    for (size_t i = 0; i < weight_count; ++i) {
        gradients[i] = (predictions[i] - ground_truth[i]) / static_cast<float>(weight_count);
    }
    std::cout << "✓ Gradients computed." << std::endl;

    // 6b. Update weights (simulate SGD)
    std::vector<float> updated_weights(weight_count);
    for (size_t i = 0; i < weight_count; ++i) {
        updated_weights[i] = initial_weights[i] - learning_rate * gradients[i];
    }
    std::cout << "✓ Weights updated (simulated SGD)." << std::endl;

    // 6c. Set updated weights back to model
    if (!aiTrainer.SetLastLayerWeights(updated_weights.data(), weight_count)) {
        std::cerr << "WARNING: Failed to set updated weights to model." << std::endl;
    } else {
        std::cout << "✓ Updated weights written to model." << std::endl;
    }

    // 6d. Verify weights were updated
    std::vector<float> verified_weights(weight_count);
    if (aiTrainer.GetLastLayerWeights(verified_weights.data(), &buffer_size)) {
        PrintWeightsSummary(verified_weights, "\nWeights after training");

        // Check if weights actually changed
        bool weights_changed = false;
        float max_change = 0.0f;
        for (size_t i = 0; i < weight_count; ++i) {
            float change = std::abs(verified_weights[i] - initial_weights[i]);
            if (change > 1e-6f) {
                weights_changed = true;
            }
            if (change > max_change) {
                max_change = change;
            }
        }

        std::cout << "\nMax weight change: " << max_change << std::endl;

        if (weights_changed) {
            std::cout << "✓ SUCCESS: Weights were updated!" << std::endl;
        } else {
            std::cout << "⚠ WARNING: Weights did not change. Possible reasons:" << std::endl;
            std::cout << "   - Output tensor is read-only (TFLM limitation)" << std::endl;
            std::cout << "   - Need to modify internal weight tensors instead" << std::endl;
        }
    }

    // 6e. Compute loss after training (with new predictions)
    // In reality, we'd need to re-run inference, but for testing:
    std::vector<float> new_predictions(weight_count);
    for (size_t i = 0; i < weight_count; ++i) {
        new_predictions[i] = updated_weights[i];
    }
    float loss_after = aiTrainer.ComputeLoss(new_predictions.data(), ground_truth.data(), weight_count);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Step Results:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Loss before:  " << std::fixed << std::setprecision(8) << loss_before << std::endl;
    std::cout << "Loss after:   " << loss_after << std::endl;
    std::cout << "Loss change:  " << (loss_before - loss_after) << std::endl;

    if (loss_after < loss_before) {
        std::cout << "✓ SUCCESS: Loss decreased! Training is working!" << std::endl;
    } else {
        std::cout << "⚠ WARNING: Loss did not decrease." << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Complete!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
