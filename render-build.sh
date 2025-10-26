#!/usr/bin/env bash
set -o errexit  # Exit on first error

# Update package list and install system dependencies for PyAudio
apt-get update
apt-get install -y portaudio19-dev python3-dev

# Upgrade pip to latest version
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
