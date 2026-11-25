#!/bin/bash
# with_env.sh <envfile> <script> <args...> : source a harness-written env file, then run the script.
source "$1"; shift; exec bash "$@"
