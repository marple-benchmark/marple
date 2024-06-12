max_iterations=10000
iteration=0

while [ $iteration -lt $max_iterations ]; do
    python auto_control.py --config mini_behavior/configs/envs/init_config_agent_a_v2.json
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "Script executed successfully."
        break
    else
        echo "Script exited with an error. Restarting..."
        sleep 1  # Optional delay before restarting
    fi
    ((iteration++))  # Increment the iteration counter
done

if [ $iteration -eq $max_iterations ]; then
    echo "Reached the maximum number of iterations ($max_iterations). Exiting."
fi
