#pragma once

#include "client.h"  // Flower C++ SDK
#include "RvcAIInterface.h"
#include "RvcAITrainer.h"

/**
 * @brief Flower Client for RVC (Robot Vacuum Cleaner)
 *
 * This client implements Federated Learning for the RVC app using Flower.
 * It connects to a Flower server and participates in collaborative training
 * of the YOLO object detection model.
 */
class RvcFlowerClient : public flwr_local::Client {
public:
    RvcFlowerClient(RvcAIInterface* ai_interface, RvcAITrainer* trainer, int node_id = 0);
    ~RvcFlowerClient();

    // Flower Client interface implementation
    flwr_local::ParametersRes get_parameters() override;
    flwr_local::PropertiesRes get_properties(flwr_local::PropertiesIns ins) override;
    flwr_local::FitRes fit(flwr_local::FitIns ins) override;
    flwr_local::EvaluateRes evaluate(flwr_local::EvaluateIns ins) override;

private:
    RvcAIInterface* mAIInterface;
    RvcAITrainer* mTrainer;
    int mNodeId;

    // Helper functions
    flwr_local::Parameters GetWeightsAsParameters();
    void SetParametersAsWeights(const flwr_local::Parameters& params);
};
