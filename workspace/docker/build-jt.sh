#!/bin/bash

#Build docker image for jet tagger locally
docker build -f Dockerfile -t ghcr.io/ben-hawks/loss_landscape_taxonomy:jt --progress=plain --build-arg MODEL=jt --build-arg BASE_IMG=nvcr.io/nvidia/pytorch:21.11-py3 . 2>&1 | tee jt_build_docker.log