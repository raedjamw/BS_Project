#!/bin/bash

# $0 is the full path of the running script
scriptdir="$(dirname "$0")"
echo '0 is the full path of the running script'
cd "$scriptdir"
# ssh to the EC2 instance
ssh -i ./BS_App.pem ubuntu@ec2-35-171-128-56.compute-1.amazonaws.com 'bash -i'  <<-'ENDSSH'
    # Pull the image
    sudo docker pull raedjamw/bs_penn:7.0
    # Run the image in the container
    sudo docker run --name Deploy_SF_Final -p 8020:8020 raedjamw/bs_penn:7.0


ENDSSH
