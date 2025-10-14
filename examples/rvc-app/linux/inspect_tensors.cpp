/*
 * Tensor Inspector - Find weight tensors in YOLO model
 *
 * This program inspects all tensors in the TFLM interpreter
 * to find weight tensors that can be modified for training.
 */

#include "../../rvc-common/include/RvcAIInterface.h"
#include <tensorflow/lite/micro/micro_interpreter.h>

#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   TFLM Tensor Inspector" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Initialize AI Interface
    std::cout << "Initializing AI Interface..." << std::endl;
    RvcAIInterface aiInterface;
    if (!aiInterface.InitAI()) {
        std::cerr << "ERROR: Failed to initialize AI Interface!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Interface initialized.\n" << std::endl;

    // Get the interpreter
    tflite::MicroInterpreter* interpreter = aiInterface.GetInterpreter();
    if (!interpreter) {
        std::cerr << "ERROR: Could not get interpreter!" << std::endl;
        return -1;
    }

    // Try to access tensors
    std::cout << "Inspecting tensors...\n" << std::endl;

    // We don't know the total number of tensors, so we'll try indices until we fail
    int tensor_count = 0;
    for (int i = 0; i < 1000; ++i) {  // Try up to 1000 tensors
        TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(i);
        if (!eval_tensor) {
            break;  // No more tensors
        }
        tensor_count++;

        // Print tensor info
        std::cout << "Tensor " << i << ":" << std::endl;
        std::cout << "  Type: " << eval_tensor->type << std::endl;
        std::cout << "  Dims: " << eval_tensor->dims->size << " [";
        for (int d = 0; d < eval_tensor->dims->size; ++d) {
            std::cout << eval_tensor->dims->data[d];
            if (d < eval_tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Calculate total elements
        int total_elements = 1;
        for (int d = 0; d < eval_tensor->dims->size; ++d) {
            total_elements *= eval_tensor->dims->data[d];
        }
        std::cout << "  Total elements: " << total_elements << std::endl;

        // Check if this looks like a weight tensor
        if (eval_tensor->dims->size == 4) {
            // Conv2D weights are typically 4D: [out_channels, kernel_h, kernel_w, in_channels]
            std::cout << "  ⭐ LIKELY WEIGHT (4D tensor - Conv2D)" << std::endl;
        } else if (eval_tensor->dims->size == 2) {
            // Fully connected weights are 2D: [out_features, in_features]
            if (eval_tensor->dims->data[0] > 1 && eval_tensor->dims->data[1] > 1) {
                std::cout << "  ⭐ LIKELY WEIGHT (2D tensor - Dense/FC)" << std::endl;
            }
        }

        std::cout << std::endl;

        // Limit output to first 50 tensors for readability
        if (i >= 50) {
            std::cout << "... (showing first 50 tensors only)" << std::endl;
            break;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Total tensors found: " << tensor_count << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "Next steps:" << std::endl;
    std::cout << "1. Identify weight tensor indices from the output above" << std::endl;
    std::cout << "2. Use GetTensor(index) to access weight tensors" << std::endl;
    std::cout << "3. Modify weight values for training" << std::endl;

    return 0;
}
