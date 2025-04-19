# Final assignment: Cityscapes challenge
Welcome to the Final assignment. To improve robustness, the following strategies were implemented:
1. Using the model "DeeplabV3 with a ResNet101 backbone
2. Using generalised Dice loss to deal with class imbalance
3. Using painting by numbers to increase shape-bias and rely less on texture

Unfortunately, Deeplab v3 won't perform. In the end, a U-net was trained which performed better.  

## How to run this code
Steps to run the code:
1. Clone this repository
2. Run Weekly notebook 00_installation.ipynb to install required packages
3. Download prerequisites: Run the following code:
```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```
4. Set up a Weights&Biases environment and get an API key
5. Copy the API key in the .env file and add a directory
6. Import data by running "download_docker_and_data.sh"
7. Modify script as needed
8. Submit job to cluster by running "Jobscript_slurm.sh"

## Codalab
Codalab username: Robertderidder
TU/e mail: c.d.ridder@student.tue.nl