echo "Warning: This will terminate all processes matching 'train_ppo.py'."
echo -n "Do you want to proceed? (y/n): "
read answer

if [ "$answer" == "y" ]; then
    pkill -f "train_ppo.py"
    echo "Processes terminated."
else
    echo "No processes were terminated."
fi
