<img width="865" height="616" alt="image" src="https://github.com/user-attachments/assets/af277b1d-916c-4833-a46f-d74b771872aa" />

# Introduction
We propose **GTT-EC**, a hierarchical graph transformer framework for enzyme commission (EC) number prediction. The accurate prediction of enzyme function has profound implications for protein engineering, drug discovery, and biocatalysis. Without requiring explicit functional residue annotations, GTT-EC constructs protein graphs to integrate sequential embeddings (from a pre-trained language model, ProtTrans) and structural features (3D coordinates and DSSP from ESMFold-predicted structures). This allows the model to learn discriminative representations and extract functional information from the implicit distribution properties of critical functional residues. Furthermore, GTT-EC utilizes attention mechanisms within its Graph Encoder and Decoder to capture long-range interactions and dynamically focus on key functional regions. Experiments demonstrate that GTT-EC significantly outperforms state-of-the-art methods in both accuracy and robustness across varying sequence similarities, effectively identifying functional regions without prior site inputs.

# System requirement
GTT-EC is developed under a Linux environment. The major dependencies include:
- Python 3.8+
- PyTorch
- PyTorch Geometric (pyg)
- Transformers (Hugging Face)
- Biopython
- numpy, scipy, tqdm

# Install and run the program
**1.** Navigate to the project directory:
```bash
cd GTT-EC
```
      
**2.** Install the required packages if you haven't already. You may need to install [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans) following their official tutorials.

**3.** Run the GTT-EC prediction example with the following command:    
```bash
python test_example.py
```
This script loads the pre-trained weights from `./models/best_model.pth` and predicts the EC numbers for the sample proteins provided in `./data/example.pkl`. 

The output will be saved in `./results/prediction_results.txt`, which contains a clear comparison between the True EC Numbers and the Predicted EC Numbers for each protein.

# Feature Extraction
If you want to run GTT-EC on your own FASTA sequences, you need to extract the corresponding sequence embeddings and structural features first. The feature extraction scripts are located in the `Features/` folder.

### 1. Sequence Embeddings (ProtTrans)
We use the `ProtT5-XL-UniRef50` model to extract sequence embeddings. The function will read the FASTA file and save the sequence representations as `.tensor` files.
```python
from Features.features import get_prottrans

# Replace with your actual paths
fasta_file = "your_data.fasta"
output_path = "./features_output/prottrans/"
gpu_id = "0"

get_prottrans(fasta_file, output_path, gpu_id)
```

### 2. Structural Features (ESMFold & DSSP)
First, generate the 3D structures (.pdb files) using ESMFold via the provided script:
```bash
python Features/generate_pdb.py -i your_data.fasta -o ./features_output/esmfold_pdbs/ --gpu 0
```

Then, extract the 3D coordinates and DSSP (secondary structure and solvent accessibility) features from the generated PDB files:
```python
from Features.features import get_coordinates, get_dssp

fasta_file = "your_data.fasta"
pdb_dir = "./features_output/esmfold_pdbs/"
dssp_dir = "./features_output/dssp/"

# Extract 3D coordinates and save as .tensor
get_coordinates(fasta_file, pdb_dir)

# Extract DSSP features and save as .tensor
# Make sure the dssp executable path is correct (e.g., "./Features/dssp-2.0.4/")
dssp_executable_dir = "./Features/dssp-2.0.4/"
get_dssp(fasta_file, dssp_executable_dir, pdb_dir, dssp_dir)
```

# Dataset and model   
- **Pre-trained Model:** The pre-trained model weights are stored in `./models/pretrain_model.pth`.
- **Data:** The mapping index for EC numbers is located in `./data/Train_set_ec_idx.pkl`. The example graph dataset used for `test_example.py` is located at `./data/example.pkl`.
