/*
 * Debug program to investigate weight modification issue
 *
 * This program directly inspects the int8 tensor data to understand
 * why weight modifications are not persisting.
 */

#include "../../rvc-common/include/RvcAIInterface.h"
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/c/common.h>

#include <iostream>
#include <iomanip>
#include <cstring>

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Weight Modification Debug Tool" << std::endl;
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

    // Access Tensor 47
    std::cout << "Accessing Tensor 47..." << std::endl;
    TfLiteEvalTensor* eval_tensor = interpreter->GetTensor(47);
    if (!eval_tensor) {
        std::cerr << "ERROR: Could not access Tensor 47!" << std::endl;
        return -1;
    }

    std::cout << "✓ Tensor 47 accessed successfully." << std::endl;
    std::cout << "  Type: " << eval_tensor->type << " (9 = kTfLiteInt8)" << std::endl;
    std::cout << "  Dims: [";
    for (int i = 0; i < eval_tensor->dims->size; ++i) {
        std::cout << eval_tensor->dims->data[i];
        if (i < eval_tensor->dims->size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Calculate total elements
    int total_elements = 1;
    for (int i = 0; i < eval_tensor->dims->size; ++i) {
        total_elements *= eval_tensor->dims->data[i];
    }
    std::cout << "  Total elements: " << total_elements << std::endl;

    // Get the data pointer
    int8_t* data_ptr = eval_tensor->data.int8;
    std::cout << "  Data pointer: " << static_cast<void*>(data_ptr) << std::endl;

    if (!data_ptr) {
        std::cerr << "ERROR: Data pointer is NULL!" << std::endl;
        return -1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "STEP 1: Read Original Values" << std::endl;
    std::cout << "========================================" << std::endl;

    // Print first 20 int8 values
    std::cout << "First 20 int8 values (original):" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(4) << static_cast<int>(data_ptr[i]);
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    // Save original values
    int8_t original_values[20];
    std::memcpy(original_values, data_ptr, 20);

    std::cout << "\n========================================" << std::endl;
    std::cout << "STEP 2: Attempt Direct Write" << std::endl;
    std::cout << "========================================" << std::endl;

    // Try to write new values directly
    std::cout << "Writing new values (all 127) to first 20 elements..." << std::endl;
    for (int i = 0; i < 20; ++i) {
        data_ptr[i] = 127;
    }
    std::cout << "✓ Write operation completed." << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "STEP 3: Read Back Values" << std::endl;
    std::cout << "========================================" << std::endl;

    // Read back the values
    std::cout << "First 20 int8 values (after write):" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(4) << static_cast<int>(data_ptr[i]);
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "STEP 4: Verify Changes" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check if values changed
    bool changed = false;
    int change_count = 0;
    for (int i = 0; i < 20; ++i) {
        if (data_ptr[i] != original_values[i]) {
            changed = true;
            change_count++;
        }
    }

    if (changed) {
        std::cout << "✓ SUCCESS: " << change_count << " values changed!" << std::endl;
        std::cout << "  The tensor IS writable!" << std::endl;
    } else {
        std::cout << "✗ FAILURE: No values changed!" << std::endl;
        std::cout << "  Possible reasons:" << std::endl;
        std::cout << "  1. Tensor data is in read-only memory" << std::endl;
        std::cout << "  2. GetTensor() returns a const/read-only view" << std::endl;
        std::cout << "  3. Writes are being silently ignored" << std::endl;
        std::cout << "  4. Need to access tensor through different API" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "STEP 5: Test After Inference" << std::endl;
    std::cout << "========================================" << std::endl;

    // Run a single inference to see if it affects the tensor
    std::cout << "Running single inference..." << std::endl;
    aiInterface.RunSingleInference();
    std::cout << "✓ Inference completed." << std::endl;

    // Try to access tensor again
    TfLiteEvalTensor* eval_tensor_after = interpreter->GetTensor(47);
    int8_t* data_ptr_after = eval_tensor_after->data.int8;

    std::cout << "Data pointer after inference: " << static_cast<void*>(data_ptr_after) << std::endl;

    if (data_ptr_after == data_ptr) {
        std::cout << "✓ Pointer unchanged - same memory location" << std::endl;
    } else {
        std::cout << "⚠ Pointer changed! Tensor was reallocated during inference!" << std::endl;
    }

    // Check values again
    std::cout << "\nFirst 20 int8 values (after inference):" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(4) << static_cast<int>(data_ptr_after[i]);
        if ((i + 1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete!" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
