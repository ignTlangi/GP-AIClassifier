MLP NEURAL NETWORK PROGRAM

This program implements a neural network for binary classification.

HOW TO USE:
1. Make sure you have Python installed
2. Install required packages by running:
   pip install numpy pandas scikit-learn
3. Run the program:
   python mlp_neural_network.py

PROGRAM FLOW:
- When you run the program, it will ask for:
  * A seed number (enter 0 for new experiment)
  * Training data file path (if new experiment)
  * Test data file path (if new experiment)

DATA FORMAT:
- Use CSV files with:
  * Last column = target (0 or 1)
  * Other columns = features
  * No header row needed


CUSTOMIZATION:
You can change these settings in the code:
- Hidden layer size
- Learning rate
- Training epochs
- Batch size

OUTPUTS:
- Results saved in experiment_results.json
- Executable created in dist/ folder (if built)