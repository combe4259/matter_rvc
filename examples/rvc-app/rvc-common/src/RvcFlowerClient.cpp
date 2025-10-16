#include "RvcFlowerClient.h"

#include <iostream>
#include <list>
#include <vector>
#include <cstring>

RvcFlowerClient::RvcFlowerClient(RvcAIInterface* ai_interface, RvcAITrainer* trainer, int node_id)
    : mAIInterface(ai_interface), mTrainer(trainer), mNodeId(node_id)
{
    std::cout << "RvcFlowerClient created (Node ID: " << mNodeId << ")" << std::endl;
}

RvcFlowerClient::~RvcFlowerClient()
{
    std::cout << "RvcFlowerClient destroyed." << std::endl;
}

flwr_local::Parameters RvcFlowerClient::GetWeightsAsParameters()
{
    // Get int8 weights from trainer
    size_t weight_count = mTrainer->GetLastLayerWeightsCount();
    std::vector<int8_t> weights(weight_count);
    size_t buffer_size;

    if (!mTrainer->GetLastLayerWeightsInt8(weights.data(), &buffer_size)) {
        std::cerr << "Error: Failed to get weights from trainer!" << std::endl;
        return flwr_local::Parameters();
    }

    // Convert int8 weights to string (bytes)
    std::string weights_bytes(reinterpret_cast<char*>(weights.data()), weight_count);

    // Create list of tensors (in this case, just one tensor)
    std::list<std::string> tensors;
    tensors.push_back(weights_bytes);

    // Create Parameters object
    flwr_local::Parameters params(tensors, "int8");

    std::cout << "Node " << mNodeId << ": Exported " << weight_count << " weights" << std::endl;

    return params;
}

void RvcFlowerClient::SetParametersAsWeights(const flwr_local::Parameters& params)
{
    // Get tensors from parameters
    std::list<std::string> tensors = params.getTensors();

    if (tensors.empty()) {
        std::cerr << "Error: No tensors in parameters!" << std::endl;
        return;
    }

    // Get first tensor (we only have one)
    std::string weights_bytes = tensors.front();

    // Convert string to int8 vector
    std::vector<int8_t> weights(weights_bytes.begin(), weights_bytes.end());

    // Set weights to trainer
    if (!mTrainer->SetLastLayerWeightsInt8(weights.data(), weights.size())) {
        std::cerr << "Error: Failed to set weights to trainer!" << std::endl;
        return;
    }

    std::cout << "Node " << mNodeId << ": Imported " << weights.size() << " weights" << std::endl;
}

// ============================================================================
// Flower Client Interface Implementation
// ============================================================================

flwr_local::ParametersRes RvcFlowerClient::get_parameters()
{
    std::cout << "Node " << mNodeId << ": get_parameters() called" << std::endl;

    flwr_local::Parameters params = GetWeightsAsParameters();
    return flwr_local::ParametersRes(params);
}

flwr_local::PropertiesRes RvcFlowerClient::get_properties(flwr_local::PropertiesIns ins)
{
    std::cout << "Node " << mNodeId << ": get_properties() called" << std::endl;

    // Return empty properties for now
    flwr_local::PropertiesRes res;
    return res;
}

flwr_local::FitRes RvcFlowerClient::fit(flwr_local::FitIns ins)
{
    std::cout << "Node " << mNodeId << ": fit() called - Starting training..." << std::endl;

    // Step 1: Set global weights from server
    SetParametersAsWeights(ins.getParameters());

    // Step 2: Perform local training
    // TODO: Implement actual training with gradient calculation
    // For now, we'll do a simple dummy update
    size_t weight_count = mTrainer->GetLastLayerWeightsCount();
    std::vector<int8_t> dummy_gradients(weight_count, 1); // All gradients = 1

    std::cout << "Node " << mNodeId << ": Applying dummy gradients..." << std::endl;
    mTrainer->UpdateWeightsInt8(dummy_gradients.data(), dummy_gradients.size());

    std::cout << "Node " << mNodeId << ": Training finished." << std::endl;

    // Step 3: Return updated weights
    flwr_local::Parameters updated_params = GetWeightsAsParameters();

    // Create FitRes
    flwr_local::FitRes res;
    res.setParameters(updated_params);
    res.setNum_example(100);  // Dummy: number of training samples

    // Optional: Add metrics
    flwr_local::Metrics metrics;
    flwr_local::Scalar train_loss;
    train_loss.setDouble(0.5);  // Dummy loss value
    metrics["train_loss"] = train_loss;
    res.setMetrics(metrics);

    return res;
}

flwr_local::EvaluateRes RvcFlowerClient::evaluate(flwr_local::EvaluateIns ins)
{
    std::cout << "Node " << mNodeId << ": evaluate() called - Starting evaluation..." << std::endl;

    // Step 1: Set weights from server
    SetParametersAsWeights(ins.getParameters());

    // Step 2: Evaluate model
    // TODO: Implement actual evaluation
    // For now, return dummy metrics

    float dummy_loss = 0.5f;
    float dummy_accuracy = 0.8f;

    std::cout << "Node " << mNodeId << ": Evaluation finished. Loss=" << dummy_loss
              << ", Accuracy=" << dummy_accuracy << std::endl;

    // Create EvaluateRes
    flwr_local::EvaluateRes res;
    res.setLoss(dummy_loss);
    res.setNum_example(100);  // Dummy: number of test samples

    // Optional: Add metrics
    flwr_local::Metrics metrics;
    flwr_local::Scalar acc_scalar;
    acc_scalar.setDouble(static_cast<double>(dummy_accuracy));
    metrics["accuracy"] = acc_scalar;
    res.setMetrics(metrics);

    return res;
}
