/*
 * Flower C++ Client for RVC App
 *
 * This program starts a Flower client that connects to a Flower server
 * and participates in federated learning for YOLO object detection.
 *
 * Usage:
 *   ./flower-rvc-client <node_id> <server_address>
 *
 * Example:
 *   ./flower-rvc-client 0 127.0.0.1:9092
 */

#include "../../rvc-common/include/RvcAIInterface.h"
#include "../../rvc-common/include/RvcAITrainer.h"
#include "../../rvc-common/include/RvcFlowerClient.h"
#include "start.h"  // Flower C++ SDK

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Flower C++ Client for RVC" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <node_id> <server_address>" << std::endl;
        std::cerr << "Example: " << argv[0] << " 0 127.0.0.1:9092" << std::endl;
        return -1;
    }

    int node_id = std::stoi(argv[1]);
    std::string server_address = argv[2];

    std::cout << "Node ID: " << node_id << std::endl;
    std::cout << "Server Address: " << server_address << std::endl;
    std::cout << std::endl;

    // Step 1: Initialize AI Interface
    std::cout << "[1/3] Initializing AI Interface..." << std::endl;
    RvcAIInterface aiInterface;
    if (!aiInterface.InitAI()) {
        std::cerr << "ERROR: Failed to initialize AI Interface!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Interface initialized successfully.\n" << std::endl;

    // Step 2: Initialize AI Trainer
    std::cout << "[2/3] Initializing AI Trainer..." << std::endl;
    RvcAITrainer aiTrainer;
    if (!aiTrainer.AttachInferenceEngine(&aiInterface)) {
        std::cerr << "ERROR: Failed to attach trainer to inference engine!" << std::endl;
        return -1;
    }
    std::cout << "✓ AI Trainer attached successfully.\n" << std::endl;

    // Step 3: Start Flower Client
    std::cout << "[3/3] Starting Flower Client..." << std::endl;
    RvcFlowerClient flowerClient(&aiInterface, &aiTrainer, node_id);

    std::cout << "✓ Flower Client created." << std::endl;
    std::cout << "Connecting to server at " << server_address << "..." << std::endl;
    std::cout << std::endl;

    // Connect to Flower server
    // This is a blocking call that will run until the server disconnects
    start::start_client(server_address, &flowerClient);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Flower Client Disconnected" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
