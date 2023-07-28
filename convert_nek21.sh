#!/usr/bin/env bash

# sed 's/^.*\"data\"\:{//' "$1" | sed 's/^/{/'


grep MessageDTO "$1" > "$2"

