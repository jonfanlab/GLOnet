import matlab.engine
import os
import logging
import argparse
import numpy as np
from train_and_evaluate import evaluate, train
from net import Generator
import utils
import torch
 

# start matlab engine
eng = matlab.engine.start_matlab()
# RCWA path
eng.addpath(eng.genpath('/home/users/jiangjq/Desktop/reticolo_allege'));
eng.addpath(eng.genpath('solvers'));


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='results',
                                        help="Results folder")
parser.add_argument('--wavelength', default=None)
parser.add_argument('--angle', default=None)
parser.add_argument('--restore_from', default=None,
                                        help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Load the directory from commend line
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.output_dir, 'train.log'))

    # Load parameters from json file
    json_path = os.path.join(args.output_dir,'Params.json')
    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Add attributes to params
    params.output_dir = args.output_dir
    params.cuda = torch.cuda.is_available()
    params.restore_from = args.restore_from
    params.numIter = int(params.numIter)
    params.noise_dims = int(params.noise_dims)
    params.gkernlen = int(params.gkernlen)
    params.step_size = int(params.step_size)    

    if args.wavelength is not None:
        params.wavelength = int(args.wavelength)
    if args.angle is not None:
        params.angle = int(args.angle)


    # make directory
    os.makedirs(args.output_dir + '/outputs', exist_ok = True)
    os.makedirs(args.output_dir + '/model', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/histogram', exist_ok = True)
    os.makedirs(args.output_dir + '/figures/deviceSamples', exist_ok = True)

    # Define the models 
    generator = Generator(params)
        
    # Move to gpu if possible
    if params.cuda:
        generator.cuda()


    # Define the optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma = params.gamma)


    # Load model data
    if args.restore_from is not None :
        params.checkpoint = utils.load_checkpoint(restore_from, generator, optimizer, scheduler)
        logging.info('Model data loaded')

    
    # Train the model and save 
    if params.numIter != 0 :
        logging.info('Start training')   
        train(generator, optimizer, scheduler, eng, params)

    # Generate images and save 
    logging.info('Start generating devices')
    evaluate(generator, eng, numImgs=500, params=params)




