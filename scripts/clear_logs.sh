#!/bin/bash

LOG_DIR="log"

# Check if the log directory exists
if [ -d "$LOG_DIR" ]; then
    # Confirm with the user before proceeding
    read -p "Are you sure you want to clear all logs in $LOG_DIR? (y/n): " answer

    # Check the user's response
    if [ "$answer" == "y" ]; then
        # Clear all log files
        rm -f "$LOG_DIR"/*.log
        echo "Logs in $LOG_DIR cleared successfully."
    else
        echo "Logs were not cleared."
    fi
else
    echo "Log directory $LOG_DIR does not exist."
fi
