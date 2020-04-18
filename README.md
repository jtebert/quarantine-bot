# Quarantine Bot

This is the start of the code for Quarantine Bot. Eventually, it will handle all of the movement and sensing control. At the moment, though, it only includes a utility for testing object detection.

For more information on the Quarantine Bot project, [read my build log](https://docs.juliaebert.com/projects/quarantine-bot).

## Hardware requirements

- Linux computer with root privileges (tested on Ubuntu 19.10 and Raspberry Pi 4)
- Intel RealSense camera. (We're using the D435.)
- [Coral USB Accelerator](https://coral.ai/products/accelerator) (for object detection)

## Installation

(Based on the [instructions from Google](https://coral.ai/docs/accelerator/get-started/))

Clone this repository:

```shell
git clone git@github.com:jtebert/quarantine-bot.git
```

Run the setup script:

```shell
cd quarantine-bot
./setup.sh
```

This script does three things:

- Install the Edge TPU runtime on the system (requires sudo)
- Install the Python dependencies
- Download the model data for object detection

## Run

Right now, this does one thing: object detection, which will show you bounding boxes of a limited set of objects from the COCO dataset (provided by Google).

```shell
python qbot/object_detection/object_detection.py
```

To see the input argument options for this script, run:
```shell
python qbot/object_detection/object_detection.py --help
```