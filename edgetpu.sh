#!/bin/sh

docker build --build-arg ARCH=armv7hf --build-arg CHIP=edgetpu --tag inference_arm .
docker cp $(docker create inference_arm):/opt/app ./build
mv build/*.eap .
rm -rf build
