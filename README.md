# Global optimization based on generative nerual networks (GLOnet)

## Requirements

We recommend using python3 and a virtual environment

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

A matlab engine for python is needed for EM simulation. Please refer to [MathWorks Pages](https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html) for installation.

Path of [RETICOLO](https://www.lp2n.institutoptique.fr/equipes-de-recherche-du-lp2n/light-complex-nanostructures) should be added in the `main.py`

## Training the GLOnet

You can change the parameters by editing `Params.json` in `results` folder. 

If you want to train the network, simply run
```
python main.py 
```

or 

```
python main.py --output_dir results --wavelength 900 --angle 60
```

to specify non-default output folder or parameters


## Results

All results will store in output_dir/ folder.
	
	-figures/  (figures of generated devices and loss function curve)
	
	-model/    (all weights of the generator)
	
	-outputs/  (500 generated devices in `.mat` format)
	
	-history.mat
	
	-train.log

## Citation
If you use this code for your research, please cite:

[Simulator-based training of generative models for the inverse design of metasurfaces.<br>](https://arxiv.org/abs/1906.07843)
Jiaqi Jiang, Jonathan A. Fan 

[Global Optimization of Dielectric Metasurfaces Using a Physics-Driven Neural Network.<br>](https://pubs.acs.org/doi/abs/10.1021/acs.nanolett.9b01857)
Jiaqi Jiang, Jonathan A. Fan

