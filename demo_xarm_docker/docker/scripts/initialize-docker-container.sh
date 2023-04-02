#!/bin/bash

# COPY ssh keys and overwrite file owner
cp -R /tmp/.ssh /root/

################################################################################

# Start the D-Bus daemon.
service dbus start

################################################################################

# Keep the Docker container running in the background.
# https://stackoverflow.com/questions/30209776/docker-container-will-automatically-stop-after-docker-run-d
tail -f /dev/null
