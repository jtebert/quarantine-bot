#! /bin/bash

# Setup script to handle system requirements and python installation

# Install Edge TPU runtime
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

# Create Python3 virtual environment and install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download the models for Edge TPU inference
readonly DATA_URL=https://github.com/google-coral/edgetpu/raw/master/test_data

# Get TF Lite model and labels
MODEL_DIR="$models"
mkdir -p "${MODEL_DIR}"
(cd "${MODEL_DIR}"
curl -OL "${TEST_DATA_URL}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite" \
     -OL "${TEST_DATA_URL}/mobilenet_ssd_v2_coco_quant_postprocess.tflite" \
     -OL "${TEST_DATA_URL}/coco_labels.txt")