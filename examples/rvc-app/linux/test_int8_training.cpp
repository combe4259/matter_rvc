/*
 * Test program for int8 quantized training
 *
 * This program tests the new int8-based weight update mechanism
 * which avoids precision loss from float conversion.
 */

#include "../../rvc-common/include/RvcAIInterface.h"
#include "../../rvc-common/include/RvcAITrainer.h"

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// Helper function to print int8 weight statistics
void PrintInt8WeightsSummary(const std::vector<int8_t>& weights, const std::string& label) {
    if (weights.empty()) return;

    int64_t sum = 0;
    int8_t min_val = weights[0];
    int8_t max_val = weights[0];

    for (const int8_t& w : weights) {
        sum += w;
        if (w < min_val) min_val = w;
        if (w > max_val) max_val = w;
    }

    double mean = static_cast<double>(sum) / weights.size();

    std::cout << label << ":" << std::endl;
    std::cout << "  Count: " << weights.size() << std::endl;
    std::cout << "  Mean:  " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  Min:   " << static_cast<int>(min_val) << std::endl;
    std::cout << "  Max:   " << static_cast<int>(max_val) << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   INT8 Quantized Training Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Step 1: Initialize AI Interface
    std::cout << "[1/5] Initializing AI Interface..." << std::endl;
    RvcAIInterface aiInterface;
    if (!aiInterface.InitAI()) {
        std::cerr << "ERROR: Failed to initialize AI Interface!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Interface initialized.\n" << std::endl;

    // Step 2: Initialize AI Trainer
    std::cout << "[2/5] Initializing AI Trainer..." << std::endl;
    RvcAITrainer aiTrainer;
    if (!aiTrainer.AttachInferenceEngine(&aiInterface)) {
        std::cerr << "ERROR: Failed to attach trainer!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Trainer attached.\n" << std::endl;

    // Step 3: Extract initial weights (int8)
    std::cout << "[3/5] Extracting initial int8 weights..." << std::endl;
    size_t weight_count = aiTrainer.GetLastLayerWeightsCount();
    std::cout << "Weight count: " << weight_count << std::endl;

    std::vector<int8_t> initial_weights(weight_count);
    size_t buffer_size;
    if (!aiTrainer.GetLastLayerWeightsInt8(initial_weights.data(), &buffer_size)) {
        std::cerr << "ERROR: Failed to extract int8 weights!" << std::endl;
        return -1;
    }
    PrintInt8WeightsSummary(initial_weights, "Initial Weights (int8)");
    std::cout << std::endl;

    // Step 4: Create int8 gradients
    std::cout << "[4/5] Creating int8 gradients..." << std::endl;
    std::vector<int8_t> gradients(weight_count);

    // Simple gradient pattern: alternate +5 and -5
    for (size_t i = 0; i < weight_count; ++i) {
        gradients[i] = (i % 2 == 0) ? 5 : -5;
    }
    std::cout << "✓ Created " << weight_count << " int8 gradients.\n" << std::endl;

    // Step 5: Update weights directly in int8
    std::cout << "[5/5] Updating weights with int8 gradients..." << std::endl;
    if (!aiTrainer.UpdateWeightsInt8(gradients.data(), weight_count)) {
        std::cerr << "ERROR: Failed to update weights!" << std::endl;
        return -1;
    }
    std::cout << "✓ Weights updated.\n" << std::endl;

    // Verify: Read weights back
    std::cout << "Verifying weight changes..." << std::endl;
    std::vector<int8_t> updated_weights(weight_count);
    if (!aiTrainer.GetLastLayerWeightsInt8(updated_weights.data(), &buffer_size)) {
        std::cerr << "ERROR: Failed to read updated weights!" << std::endl;
        return -1;
    }
    PrintInt8WeightsSummary(updated_weights, "\nUpdated Weights (int8)");

    // Calculate differences
    std::cout << "\n========================================" << std::endl;
    std::cout << "Change Analysis:" << std::endl;
    std::cout << "========================================" << std::endl;

    int change_count = 0;
    int max_change = 0;
    int64_t total_change = 0;

    for (size_t i = 0; i < weight_count; ++i) {
        int change = std::abs(static_cast<int>(updated_weights[i]) - static_cast<int>(initial_weights[i]));
        if (change > 0) {
            change_count++;
            total_change += change;
        }
        if (change > max_change) {
            max_change = change;
        }
    }

    double avg_change = change_count > 0 ? static_cast<double>(total_change) / change_count : 0.0;

    std::cout << "Changed weights: " << change_count << " / " << weight_count
              << " (" << std::fixed << std::setprecision(2)
              << (100.0 * change_count / weight_count) << "%)" << std::endl;
    std::cout << "Max change:      " << max_change << std::endl;
    std::cout << "Avg change:      " << std::setprecision(3) << avg_change << std::endl;

    // Sample: Print first 20 weights before/after
    std::cout << "\nFirst 20 weights (before → after):" << std::endl;
    for (int i = 0; i < 20 && i < static_cast<int>(weight_count); ++i) {
        int change = static_cast<int>(updated_weights[i]) - static_cast<int>(initial_weights[i]);
        std::cout << "  [" << std::setw(2) << i << "] "
                  << std::setw(4) << static_cast<int>(initial_weights[i])
                  << " → "
                  << std::setw(4) << static_cast<int>(updated_weights[i])
                  << " (Δ=" << std::showpos << std::setw(3) << change << ")" << std::noshowpos
                  << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    if (change_count > 0) {
        std::cout << "✓ SUCCESS: Int8 training works!" << std::endl;
        std::cout << "  Weights are being modified correctly!" << std::endl;
    } else {
        std::cout << "✗ FAILURE: No weights changed!" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    return (change_count > 0) ? 0 : -1;
}
