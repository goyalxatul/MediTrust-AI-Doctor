#!/usr/bin/env bash
set -o errexit

apt-get update
pip install -r requirements.txt
