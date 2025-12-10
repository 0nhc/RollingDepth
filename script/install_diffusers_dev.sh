#!/usr/bin/env bash
set -e
set -x

# c.f. https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md#how-to-open-a-pr

cd diffusers
pip install -e .