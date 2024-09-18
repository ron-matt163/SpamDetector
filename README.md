## How to run this spam detector?

Step 1: Create a virtual environment using the following command
virtualenv rjmathespamdetector
source rjmathespamdetector/bin/activate

Step 2: Install dependencies using the following command
chmod +x install_requirements.sh
./install_requirements.sh

Step 3: Enter the /src folder inside this project directory
To train the binary classifier spam detector:
1. Run the following command
"python main.py train"

This saves the model in the /spam_roberta_classifier folder inside the project directory
If you don't want to train the model, download the model folder from this link and place the folder inside
the project directory: [Link to the trained model](https://drive.google.com/drive/folders/1dFnp03bJkkXDU1QbCTh3vm2dCfOIoV7X?usp=sharing)
The model folder must be named 'spam_roberta_classifier' in this case.

To test the binary classifier spam detector:
1. Run the following command
"python main.py test"


