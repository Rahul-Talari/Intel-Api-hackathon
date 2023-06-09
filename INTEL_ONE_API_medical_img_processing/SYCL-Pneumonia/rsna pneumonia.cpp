#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace sycl = cl::sycl;

constexpr int dataSize = 100; // Number of data points

// Kernel function to predict pneumonia
void pneumoniaPrediction(const float* input, int* output, const float threshold, const int dataSize) {
    for (int i = 0; i < dataSize; ++i) {
        output[i] = (input[i] > threshold) ? 1 : 0;
    }
}

// SYCL implementation
void dataPreparation(std::vector<std::string>& trainPaths, std::vector<int>& trainTargets) {
    sycl::default_selector device_selector;
    sycl::queue queue(device_selector);
    sycl::buffer<std::string> trainPathsBuffer(trainPaths);
    sycl::buffer<int> trainTargetsBuffer(trainTargets);

    queue.submit([&](sycl::handler& cgh) {
        auto trainPathsAccessor = trainPathsBuffer.get_access<sycl::access::mode::read>(cgh);
        auto trainTargetsAccessor = trainTargetsBuffer.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class DataPreparationKernel>(sycl::range<1>(trainPaths.size()), [=](sycl::id<1> idx) {
            // Perform data preparation computations here using SYCL parallel_for
            // Access trainPathsAccessor and trainTargetsAccessor instead of trainPaths and trainTargets
            std::string trainPath = trainPathsAccessor[idx];
            int trainTarget = trainTargetsAccessor[idx];
            // Perform computations on each element of the dataset
            // Update the trainTargetsAccessor if needed
            trainTargetsAccessor[idx] = trainTarget;
            });
        });
}

int main() {
    // Create the device queue
    sycl::queue deviceQueue;

    // Input data
    std::vector<float> input(dataSize, 0.0f); // Replace with your actual input data

    // Output data
    std::vector<int> output(dataSize, 0);

    // Create buffers for input and output data
    sycl::buffer<float, 1> inputBuffer(input.data(), sycl::range<1>(dataSize));
    sycl::buffer<int, 1> outputBuffer(output.data(), sycl::range<1>(dataSize));

    // Submit a command group for execution
    deviceQueue.submit([&](sycl::handler& cgh) {
        // Accessors for input and output buffers
        auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
        auto outputAccessor = outputBuffer.get_access<sycl::access::mode::write>(cgh);

        // Kernel function
        cgh.parallel_for<class PneumoniaPredictionKernel>(sycl::range<1>(dataSize), [=](sycl::id<1> idx) {
            float inputValue = inputAccessor[idx];
            outputAccessor[idx] = (inputValue > 0.5f) ? 1 : 0; // Set the threshold here
            });
        });

    // Wait for the command group to finish execution
    deviceQueue.wait();

    // Get the results back from the device
    sycl::host_accessor outputAccessor(outputBuffer, sycl::read_only);

    // Print the predicted results
    for (int i = 0; i < dataSize; ++i) {
        std::cout << "Data point " << i << ": " << outputAccessor[i] << std::endl;
    }

    // Prepare labels
    std::ifstream labelFile("C:\\Users\\rahul\\Desktop\\rsna-pneumonia-detection-challenge\\stage_2_train_labels.csv");
    std::string line;
    std::vector<std::string> patientIds;
    std::vector<int> targets;
    std::getline(labelFile, line); // Skip the header line
    while (std::getline(labelFile, line)) {
        std::istringstream iss(line);
        std::string patientId, target;
        std::getline(iss, patientId, ',');
        std::getline(iss, target, ',');
        patientIds.push_back(patientId);
        targets.push_back(std::stoi(target));
    }
    labelFile.close();

    // Divide labels for train and validation set
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(patientIds.begin(), patientIds.end(), g);
    std::vector<std::string> trainPatientIds(patientIds.begin(), patientIds.begin() + patientIds.size() * 0.9);
    std::vector<std::string> valPatientIds(patientIds.begin() + patientIds.size() * 0.9, patientIds.end());
    std::vector<int> trainTargets, valTargets;
    for (const std::string& patientId : trainPatientIds) {
        auto it = std::find(patientIds.begin(), patientIds.end(), patientId);
        trainTargets.push_back(targets[it - patientIds.begin()]);
    }
    for (const std::string& patientId : valPatientIds) {
        auto it = std::find(patientIds.begin(), patientIds.end(), patientId);
        valTargets.push_back(targets[it - patientIds.begin()]);
    }
    std::cout << "Train Labels: " << trainTargets.size() << std::endl;
    std::cout << "Val Labels: " << valTargets.size() << std::endl;
    std::cout << "PatientId: " << trainPatientIds[0] << ", Target: " << trainTargets[0] << std::endl;

    // Prepare train and validation image paths
    std::string trainPath = "C:\\Users\\rahul\\Desktop\\rsna-pneumonia-detection-challenge\\stage_2_train_images\\";
    std::string testPath = "C:\\Users\\rahul\\Desktop\\rsna-pneumonia-detection-challenge\\stage_2_test_images\\";
    std::vector<std::string> trainPaths, valPaths;
    for (const std::string& patientId : trainPatientIds) {
        trainPaths.push_back(trainPath + patientId + ".dcm");
    }
    for (const std::string& patientId : valPatientIds) {
        valPaths.push_back(trainPath + patientId + ".dcm");
    }
    std::cout << "Train Paths: " << trainPaths.size() << std::endl;
    std::cout << "Val Paths: " << valPaths.size() << std::endl;

    // Show some samples from data
    cv::Mat image;
    cv::namedWindow("Images", cv::WINDOW_NORMAL);
    cv::resizeWindow("Images", 1000, 1000);
    for (int i = 0; i < 9; ++i) {
        image = cv::imread(trainPaths[i + 20]);
        cv::imshow("Images", image);
        std::cout << "Label: " << trainTargets[i + 20] << std::endl;
        cv::waitKey(0);
    }

    // Perform data preparation using SYCL
    sycl::default_selector device_selector;
    sycl::queue queue(device_selector);
    sycl::buffer<std::string> trainPathsBuffer(trainPaths);
    sycl::buffer<int> trainTargetsBuffer(trainTargets);

    queue.submit([&](sycl::handler& cgh) {
        auto trainPathsAccessor = trainPathsBuffer.get_access<sycl::access::mode::read>(cgh);
        auto trainTargetsAccessor = trainTargetsBuffer.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class DataPreparationKernel>(sycl::range<1>(trainPaths.size()), [=](sycl::id<1> idx) {
            // Perform data preparation computations here using SYCL parallel_for
            // Access trainPathsAccessor and trainTargetsAccessor instead of trainPaths and trainTargets
            std::string trainPath = trainPathsAccessor[idx];
            int trainTarget = trainTargetsAccessor[idx];
            // Perform computations on each element of the dataset
            // Update the trainTargetsAccessor if needed
            trainTargetsAccessor[idx] = trainTarget;
            });
        });

    // Compose transformations
    torch::transforms::Compose transform = torch::transforms::Compose(
        {
            torch::transforms::RandomHorizontalFlip(),
            torch::transforms::Resize(224),
            torch::transforms::ToTensor()
        });

    // Create a custom dataset
    class Dataset : public torch::data::Dataset<Dataset> {
    public:
        Dataset(const std::vector<std::string>& paths, const std::vector<int>& labels, const torch::transforms::Compose& transform)
            : paths_(paths), labels_(labels), transform_(transform) {}

        torch::data::Example<> get(size_t index) override {
            cv::Mat image = cv::imread(paths_[index]);
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
            cv::resize(image, image, cv::Size(224, 224));

            torch::Tensor imageTensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
            imageTensor = imageTensor.permute({ 2, 0, 1 });
            imageTensor = imageTensor.to(torch::kFloat) / 255.0;

            if (transform_)
                imageTensor = transform_(imageTensor);

            return { imageTensor, labels_[index] };
        }

        torch::optional<size_t> size() const override {
            return paths_.size();
        }

    private:
        std::vector<std::string> paths_;
        std::vector<int> labels_;
        torch::transforms::Compose transform_;
    };

    // Check the custom dataset
    Dataset trainDataset(trainPaths, trainTargets, transform);
    torch::data::Example<> example = trainDataset.get(0);
    torch::Tensor img = example.data;
    int label = example.target.item<int>();
    std::cout << "Tensor: " << img << ", Label: " << label << std::endl;
    cv::Mat imgMat(224, 224, CV_32FC3, img.data_ptr<float>());
    cv::imshow("Image", imgMat);
    cv::waitKey(0);

    return 0;
}
