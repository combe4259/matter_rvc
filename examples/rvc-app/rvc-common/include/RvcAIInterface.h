#pragma once

#include <cstdint>
#include <memory>

// Forward declarations to avoid including heavy TFLM headers here
namespace tflite {
    class MicroInterpreter;
    struct Model;
    template <unsigned int tOpCount> class MicroMutableOpResolver;
}
struct TfLiteTensor;

class RvcAIInterface {
public:
    RvcAIInterface();
    ~RvcAIInterface(); // Important for managing unique_ptr resources

    // Returns true on success
    bool InitAI();
    void RunInferenceLoop();

private:
    const tflite::Model* mModel;
    std::unique_ptr<tflite::MicroInterpreter> mInterpreter;
    TfLiteTensor* mInputTensor;
    TfLiteTensor* mOutputTensor;

    // Using a unique_ptr for the resolver to manage its lifecycle
    std::unique_ptr<tflite::MicroMutableOpResolver<10> > mResolver;

    // A memory buffer for TFLM to use for input, output, and intermediate arrays.
    std::unique_ptr<uint8_t[]> mTensorArena;
    
    // NOTE: This size will need to be tuned for the specific model.
    // A 1MB arena is a good starting point for a YOLO model.
    static constexpr int kTensorArenaSize = 1024 * 1024;
};
