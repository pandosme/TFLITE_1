#!/bin/sh

docker build --build-arg ARCH=aarch64 --build-arg CHIP=artpec8 --tag inference .
docker cp $(docker create inference):/opt/app ./build
mv build/*.eap .
rm -rf build
