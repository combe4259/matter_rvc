/*
 *
 *    Copyright (c) 2023 Project CHIP Authors
 *    All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */
#include "RvcAppCommandDelegate.h"
#include "rvc-device.h"
#include <AppMain.h>
#include <iostream>
#include "../../rvc-common/include/RvcAIInterface.h"
#include "../../rvc-common/include/RvcAITrainer.h"

#include <string>

#define RVC_ENDPOINT 1

using namespace chip;
using namespace chip::app;
using namespace chip::app::Clusters;

namespace {
NamedPipeCommands sChipNamedPipeCommands;
RvcAppCommandDelegate sRvcAppCommandDelegate;
} // namespace

RvcDevice * gRvcDevice = nullptr;
RvcAIInterface * gAiInterface = nullptr;
RvcAITrainer * gAiTrainer = nullptr;

void ApplicationInit()
{
    std::string path = std::string(LinuxDeviceOptions::GetInstance().app_pipe);

    if ((!path.empty()) and (sChipNamedPipeCommands.Start(path, &sRvcAppCommandDelegate) != CHIP_NO_ERROR))
    {
        ChipLogError(NotSpecified, "Failed to start CHIP NamedPipeCommands");
        sChipNamedPipeCommands.Stop();
    }

    gRvcDevice = new RvcDevice(RVC_ENDPOINT);
    gRvcDevice->Init();

    sRvcAppCommandDelegate.SetRvcDevice(gRvcDevice);

    // Initialize the On-Device AI interface
    gAiInterface = new RvcAIInterface();
    if (!gAiInterface->InitAI())
    {
        std::cerr << "FATAL ERROR: Failed to initialize AI Interface." << std::endl;
        return;
    }

    // Initialize the AI Trainer for on-device learning
    gAiTrainer = new RvcAITrainer();
    if (!gAiTrainer->AttachInferenceEngine(gAiInterface))
    {
        std::cerr << "WARNING: Failed to attach trainer to inference engine." << std::endl;
    }
    else
    {
        std::cout << "\n=== Testing Transfer Learning Capabilities ===" << std::endl;

        // Test: Get current weights
        size_t weight_count;
        gAiTrainer->GetLastLayerWeights(nullptr, &weight_count);
        std::cout << "Last layer has " << weight_count << " weights." << std::endl;

        // TODO: In future, this is where we'll integrate with Flower
        // For now, we just verify the infrastructure works
        std::cout << "=== Trainer ready for Federated Learning ===" << std::endl << std::endl;
    }

    // Run inference loop (this will run continuously)
    gAiInterface->RunInferenceLoop();
}

void ApplicationShutdown()
{
    delete gRvcDevice;
    gRvcDevice = nullptr;

    delete gAiTrainer;
    gAiTrainer = nullptr;

    delete gAiInterface;
    gAiInterface = nullptr;

    sChipNamedPipeCommands.Stop();
}

int main(int argc, char * argv[])
{
    if (ChipLinuxAppInit(argc, argv) != 0)
    {
        return -1;
    }

    ChipLinuxAppMainLoop();
    return 0;
}
