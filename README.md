This is the code base for Dan Halperin's thesis: "Monocular Depth Estimation by Domain Transfer Using Rigid Transformations" (10/2024).
It is the cleaned reposetory, and at the state of the last experiment - n-dim rotations between two latent spaces.

In this repo, you could find the PyTorch implementations of all the used functions:
1. CBAM
2. Visual Trasnformers (based on LVT: https://github.com/Chenglin-Yang/LVT)
3. Dense blocks and chains
4. Infinity, Huber, BerHu loss functions
4.5. Losses weigting mechanism
5. First and only PyTorch implementation of the chi2 loss functions for forcing a gaussian distributions on a set of points, base on: https://arxiv.org/pdf/1811.04751
6. Autoencoder
7. Relative Representations

To the the code, simply make sure your working direcotry is the parent directory os "src".
1. run training: "python src/main/Trainer.py"
2. run testing: "python/main/prints.py":
    - at the header of that file, you'll need to fill the directory of where the "src" is located, and set the checkpoint to the desired one (the path to it).
 

The dataset's path is at "...../Projekte/EDGAR/02_students/Studenten_Kulmer/Dan_Halperin/thesis_data". Fill it in the right place of the datalaoder. It should be set within the src/util_moduls/args.py file. 
The dataset is already set to go - no furhter processing is required.


Also, some other experiments are added with correpsonding checkpoits in the Other_Experiments folder.

For any quesitons about running anything or if you struggle to understand some of the logic, you're more than welcome to ask me (Dan) at either of:
1. Dan.Halperin@tum.de 
2. Dan.Halperin@mail.huji.ac.il