#!/usr/bin/env bash
set -o errexit  # Exit on first error
apt-get update && apt-get install -y portaudio19-dev
pip install -r requirements.txt
