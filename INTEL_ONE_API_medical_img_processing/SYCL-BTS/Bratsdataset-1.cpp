#ifndef BRATSDATASET_H
#define BRATSDATASET_H

#include <string>
#include <vector>



// DATASET DATALOADER 
class BratsDataset
    {
        private:
        std::vector<DataItem> data;
        std::string phase;
        std::vector<std::vector<float>> (*get_augmentations)(const std::string&);
        std::vector<std::string> data_types;
        bool is_resize;

        // BratsDataset CONSTRUCTOR 
        public:
        BratsDataset(const std::vector<DataItem>& df, const std::string& phase = "test", bool is_resize = false) {
            data = df;
            this->phase = phase;
            // Assign the appropriate function pointer for get_augmentations
            if (phase == "test") {
                get_augmentations = &get_test_augmentations;
            } else {
                get_augmentations = &get_train_val_augmentations;
            }
            data_types = { "_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii" };
            this->is_resize = is_resize;
        }


        size_t BratsDataset::size() const {
            return data.size();
        }



        std::unordered_map<std::string, std::any> BratsDataset::get_item(size_t idx) {
            std::string id_ = df[idx]["Brats20ID"];
            std::string root_path = df[df["Brats20ID"] == id_]["path"].values[0];

            std::vector<ImageType> images;
            
            for (const auto& data_type : data_types) {
                std::string img_path = root_path + id_ + data_type;
                ImageType img = load_img(img_path);
                
                if (is_resize) {
                    img = resize(img);
                }
                
                img = normalize(img);
                images.push_back(img);
            }
            
            ImageType img = stack_images(images);
            img = moveaxis(img, {0, 1, 2, 3}, {0, 3, 2, 1});
            
            if (phase != "test") {
                std::string mask_path = root_path + id_ + "_seg.nii";
                ImageType mask = load_img(mask_path);
                
                if (is_resize) {
                    mask = resize(mask);
                    mask = clip(normalize(mask).astype(np.uint8), 0, 1).astype(np.float32);
                    mask = clip(mask, 0, 1);
                }
                
                mask = preprocess_mask_labels(mask);
                
                auto augmented = augmentations(img, mask);
                img = augmented["image"];
                mask = augmented["mask"];
                
                std::unordered_map<std::string, std::any> result;
                result["Id"] = id_;
                result["image"] = img;
                result["mask"] = mask;
                
                return result;
            }
            
            std::unordered_map<std::string, std::any> result;
            result["Id"] = id_;
            result["image"] = img;
            
            return result;
        }


        ImageType BratsDataset::load_img(const std::string& file_path) {
            auto data = nib::load(file_path);
            ImageType img(data.dataobj());
            return img;
        }


        ImageType BratsDataset::normalize(const ImageType& data) {
            ImageType normalized_data = (data - data.min()) / (data.max() - data.min());
            return normalized_data;
        }

        ImageType BratsDataset::resize(const ImageType& data) {
            cv::Size size(120, 120); // Target size for resizing
            cv::Mat resized_data;
            cv::resize(data, resized_data, size);
            
            // Resizing along the first dimension to size 78
            cv::Mat resized_data_3d = resized_data.reshape(1, 120); // Reshape to a 3D matrix
            cv::resize(resized_data_3d, resized_data_3d, cv::Size(120, 78)); // Resize along the first dimension to size 78
            
            // Reshape back to the original shape
            resized_data = resized_data_3d.reshape(1, 78 * 120 * 120); // Reshape to 1D vector
            resized_data = resized_data.reshape(1, 78); // Reshape to 3D matrix with size (78, 120, 120)
            
            return resized_data;
        }


        cv::Mat BratsDataset::preprocessMaskLabels(const cv::Mat& mask)
        {
            cv::Mat mask_WT = mask.clone();
            cv::Mat mask_TC = mask.clone();
            cv::Mat mask_ET = mask.clone();

            cv::compare(mask_WT, 1, mask_WT, cv::CMP_EQ);
            cv::compare(mask_WT, 2, mask_WT, cv::CMP_EQ);
            cv::compare(mask_WT, 4, mask_WT, cv::CMP_EQ);

            cv::compare(mask_TC, 1, mask_TC, cv::CMP_EQ);
            cv::compare(mask_TC, 2, mask_TC, cv::CMP_EQ);
            cv::compare(mask_TC, 4, mask_TC, cv::CMP_EQ);
            mask_TC = 1 - mask_TC;

            cv::compare(mask_ET, 1, mask_ET, cv::CMP_EQ);
            cv::compare(mask_ET, 2, mask_ET, cv::CMP_EQ);
            cv::compare(mask_ET, 4, mask_ET, cv::CMP_EQ);
            mask_ET = 1 - mask_ET;

            std::vector<cv::Mat> channels = { mask_WT, mask_TC, mask_ET };
            cv::Mat mask_combined;
            cv::merge(channels, mask_combined);

            return mask_combined;
        }

    } ;