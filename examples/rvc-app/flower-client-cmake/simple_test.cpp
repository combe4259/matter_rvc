/*
 * Simple Flower C++ Client Test
 *
 * This is a minimal test to verify Flower C++ SDK works
 * without TFLM dependencies.
 */

#include "client.h"
#include "start.h"
#include "typing.h"

#include <iostream>
#include <vector>
#include <list>
#include <string>

/**
 * Simple test client that doesn't need TFLM
 */
class SimpleTestClient : public flwr_local::Client {
private:
    int node_id;
    std::vector<float> dummy_weights;

public:
    SimpleTestClient(int id) : node_id(id) {
        // Initialize with dummy weights
        dummy_weights.resize(100, 0.5f);  // 100 weights, all 0.5
        std::cout << "SimpleTestClient created (Node " << node_id << ")" << std::endl;
    }

    flwr_local::ParametersRes get_parameters() override {
        std::cout << "Node " << node_id << ": get_parameters() called" << std::endl;

        // Convert weights to bytes
        std::string weights_bytes(
            reinterpret_cast<const char*>(dummy_weights.data()),
            dummy_weights.size() * sizeof(float)
        );

        std::list<std::string> tensors;
        tensors.push_back(weights_bytes);

        flwr_local::Parameters params(tensors, "float32");
        return flwr_local::ParametersRes(params);
    }

    flwr_local::PropertiesRes get_properties(flwr_local::PropertiesIns ins) override {
        std::cout << "Node " << node_id << ": get_properties() called" << std::endl;
        return flwr_local::PropertiesRes();
    }

    flwr_local::FitRes fit(flwr_local::FitIns ins) override {
        std::cout << "Node " << node_id << ": fit() called - Training..." << std::endl;

        // Simulate training: slightly modify weights
        for (size_t i = 0; i < dummy_weights.size(); ++i) {
            dummy_weights[i] += 0.01f;  // Increase by 0.01
        }

        std::cout << "Node " << node_id << ": Training complete. New weight[0]="
                  << dummy_weights[0] << std::endl;

        // Return updated weights
        std::string weights_bytes(
            reinterpret_cast<const char*>(dummy_weights.data()),
            dummy_weights.size() * sizeof(float)
        );

        std::list<std::string> tensors;
        tensors.push_back(weights_bytes);
        flwr_local::Parameters params(tensors, "float32");

        flwr_local::FitRes res;
        res.setParameters(params);
        res.setNum_example(100);

        return res;
    }

    flwr_local::EvaluateRes evaluate(flwr_local::EvaluateIns ins) override {
        std::cout << "Node " << node_id << ": evaluate() called" << std::endl;

        float loss = 0.5f - (dummy_weights[0] - 0.5f);  // Dummy loss

        flwr_local::EvaluateRes res;
        res.setLoss(loss);
        res.setNum_example(100);

        flwr_local::Metrics metrics;
        flwr_local::Scalar acc;
        acc.setDouble(0.8);
        metrics["accuracy"] = acc;
        res.setMetrics(metrics);

        std::cout << "Node " << node_id << ": Evaluation complete. Loss=" << loss << std::endl;

        return res;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Simple Flower C++ Client Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Parse command line
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <node_id> <server_address>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 0 127.0.0.1:9092" << std::endl;
        return -1;
    }

    int node_id = std::stoi(argv[1]);
    std::string server_address = argv[2];

    std::cout << "Node ID: " << node_id << std::endl;
    std::cout << "Server: " << server_address << std::endl;
    std::cout << std::endl;

    // Create client
    SimpleTestClient client(node_id);

    // Connect to server
    std::cout << "Connecting to Flower server..." << std::endl;
    start::start_client(server_address, &client);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Client Disconnected" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
