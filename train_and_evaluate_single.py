import matlab.engine
import os
import logging
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.nn.functional as F 
import torch
import utils
import scipy.io as io
import numpy as np
from sklearn.decomposition import PCA

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
randconst = torch.rand(1).type(Tensor)*2-1 


def make_figure_dir(folder):
    os.makedirs(folder + '/figures/scatter', exist_ok = True)
    os.makedirs(folder + '/figures/histogram', exist_ok = True)
    os.makedirs(folder + '/figures/deviceSamples', exist_ok = True)
    os.makedirs(folder + '/figures/scatter_and_histogram', exist_ok = True)


def PCA_model(data_path):
    pca = PCA(n_components=2, svd_solver='randomized')
    dataset = io.loadmat(data_path, struct_as_record = False, squeeze_me = True)
    data = dataset['data']
    pca.fit(data)
    return pca


def PCA_analysis(generator, pca, eng, params, numImgs=100):
    generator.eval()
    imgs = sample_images(generator, numImgs, params)
    generator.train()

    Efficiency = torch.zeros(numImgs)

    img = torch.squeeze(imgs[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())
    wavelength = matlab.double([params.w] * numImgs)
    desired_angle = matlab.double([params.a] * numImgs)

    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)

    
   # img = img[np.where(Efficiency.reshape(-1) > 0), :]
    #Efficiency = Efficiency[Efficiency > 0]

    img_2 = pca.transform(img)
    

    fig_path = params.output_dir + '/figures/scatter/Iter{}.png'.format(params.iter)
    utils.plot_scatter(img_2, Efficiency, params.iter, fig_path)

    fig_path = params.output_dir + '/figures/histogram/Iter{}.png'.format(params.iter)
    utils.plot_histogram(Efficiency, params.iter, fig_path)

    imgs = imgs[:8, :, :].unsqueeze(2).repeat(1, 1, 64, 1)
    fig_path = params.output_dir + '/figures/deviceSamples/Iter{}.png'.format(params.iter)
    save_image(imgs, fig_path, 2)

    '''
    grads = eng.GradientFromSolver_1D_parallel(img, wavelength, desired_angle) 
    grad_2 = pca.transform(grads)
    if params.iter % 2 == 0:
        utils.plot_envolution(params.img_2_prev, params.eff_prev, params.grad_2_prev, img_2, Efficiency, params.iter, params.output_dir)
    else:
        utils.plot_arrow(img_2, Efficiency, grad_2, params.iter, params.output_dir)
    
    params.img_2_prev = img_2
    params.eff_prev = Efficiency
    params.grad_2_prev = grad_2
    '''
    return img_2, Efficiency



def sample_images(generator, batch_size, params):   

    if params.noise_constant == 1:
        z = (torch.ones(batch_size, params.noise_dims).type(Tensor) * randconst) * params.noise_amplitude
    else:
        if params.noise_distribution == 'uniform':
            z = (torch.rand(batch_size, params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude
        else:
            z = (torch.randn(batch_size, params.noise_dims).type(Tensor)) * params.noise_amplitude

    if params.cuda:
        z.cuda()      
    return generator(z)


def evaluate(generator, eng, numImgs, params):
    generator.eval()
    
    filename = 'ccGAN_imgs_Si_w' + str(params.w) +'_' + str(params.a) +'deg.mat'
    images = sample_images(generator, numImgs, params)
    file_path = os.path.join(params.output_dir,'outputs',filename)
    logging.info('Generation is done. \n')


    Efficiency = torch.zeros(numImgs)

    images = torch.sign(images)
    img = torch.squeeze(images[:, 0, :]).data.cpu().numpy()
    img = matlab.double(img.tolist())
    wavelength = matlab.double([params.w] * numImgs)
    desired_angle = matlab.double([params.a] * numImgs)

    abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
    Efficiency = torch.Tensor([abseffs]).data.cpu().numpy().reshape(-1)

    fig_path = params.output_dir + '/figures/Efficiency.png'
    utils.plot_histogram(Efficiency, params.numIter, fig_path)

    io.savemat(file_path, mdict={'imgs': images.cpu().detach().numpy(), 'Effs' : Efficiency})

def train(models, optimizers, schedulers, eng, params):

    generator = models
    optimizer_G = optimizers
    scheduler_G = schedulers

    generator.train()

    pca = PCA_model('/scratch/users/jiangjq/GAN1D/TrainingSet/PCA_data.mat')
    
    make_figure_dir(params.output_dir)


    if params.restore_from is None:
        Eff_mean_history = []
        Binarization_history = []
        pattern_variance = []
        iter0 = 0   
        imgs_2 = []
        Effs_2 = []
    else:
        Eff_mean_history = params.checkpoint['Eff_mean_history']
        iter0 = params.checkpoint['iter']
        Binarization_history = params.checkpoint['Binarization_history']
        pattern_variance = params.checkpoint['pattern_variance']
        imgs_2 = params.checkpoint['imgs_2']
        Effs_2 = params.checkpoint['Effs_2']
    
    with tqdm(total=params.numIter) as t:
        

        for i in range(params.numIter):
            it = i + 1
            normIter = it / params.numIter
            params.iter = it + iter0

            scheduler_G.step()

            if it == 1:
                model_dir = os.path.join(params.output_dir, 'model','iter{}'.format(it+iter0))
                os.makedirs(model_dir, exist_ok = True)
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_G_state_dict': optimizer_G.state_dict(),
                                       'scheduler_G_state_dict': scheduler_G.state_dict(),
                                       'Eff_mean_history': Eff_mean_history,
                                       'Binarization_history': Binarization_history,
                                       'pattern_variance': pattern_variance,
                                       'Effs_2': Effs_2,
                                       'imgs_2': imgs_2
                                       },
                                       checkpoint=model_dir)

            if it > params.numIter:
                model_dir = os.path.join(params.output_dir, 'model')
                utils.save_checkpoint({'iter': it + iter0 - 1,
                                       'gen_state_dict': generator.state_dict(),
                                       'optim_G_state_dict': optimizer_G.state_dict(),
                                       'scheduler_G_state_dict': scheduler_G.state_dict(),
                                       'Eff_mean_history': Eff_mean_history,
                                       'Binarization_history': Binarization_history,
                                       'pattern_variance': pattern_variance,
                                       'Effs_2': Effs_2,
                                       'imgs_2': imgs_2
                                       },
                                       checkpoint=model_dir)

                #utils.movie_scatter(np.asarray(imgs_2),  np.asarray(Effs_2), params.output_dir)
                io.savemat(params.output_dir+'/scatter.mat', mdict={'imgs_2': np.asarray(imgs_2), 'Effs_2': np.asarray(Effs_2)})
                return 

            # use solver and phyiscal gradient to update the GAN
            params.solver_batch_size = int(params.solver_batch_size_start +  (params.solver_batch_size_end - params.solver_batch_size_start) * (1 - (1 - normIter)**params.solver_batch_size_power))
            if params.noise_constant == 1:
                z = (torch.ones(params.solver_batch_size, params.noise_dims).type(Tensor) * randconst) * params.noise_amplitude
            else:
                if params.noise_distribution == 'uniform':
                    z = ((torch.rand(params.solver_batch_size, params.noise_dims).type(Tensor)*2.-1.) * params.noise_amplitude)
                else:
                    z = (torch.randn(params.solver_batch_size, params.noise_dims).type(Tensor)) * params.noise_amplitude          

            gen_imgs = generator(z)
        
            img = torch.squeeze(gen_imgs[:, 0, :]).data.cpu().numpy()
            img = matlab.double(img.tolist())
            wavelength = matlab.double([params.w] * params.solver_batch_size)
            desired_angle = matlab.double([params.a] * params.solver_batch_size)
      
            #abseffs = eng.Eval_Eff_1D_parallel(img, wavelength, desired_angle)
            Grads_and_Effs = eng.GradientFromSolver_1D_parallel(img, wavelength, desired_angle)  
            Grads_and_Effs = Tensor(Grads_and_Effs)              
            grads = Grads_and_Effs[:, 1:]
            Efficiency_real = Grads_and_Effs[:, 0]

            #Gradients = Tensor(grads).unsqueeze(1) * gen_imgs * 1e-3 * (1.0 - (Efficiency_real.view(-1, 1).unsqueeze(2) - 0.5)**2)
            diversity_penalty = torch.mean(torch.std(gen_imgs, dim=0))

            mu = torch.mean(Efficiency_real.view(-1))
            sigma = torch.std(Efficiency_real.view(-1))
            Eff_max = torch.max(Efficiency_real.view(-1))
            #Gradients = Tensor(grads).unsqueeze(1) * gen_imgs * 1e-3 * (1.0 + (Efficiency_real.view(-1, 1).unsqueeze(2) - mu)/sigma)
            Eff_reshape = Efficiency_real.view(-1, 1).unsqueeze(2)
            #Gradients = Tensor(grads).unsqueeze(1) * gen_imgs * 1e-3 * (- mu + 2 * Eff_reshape)

            Gradients = Tensor(grads).unsqueeze(1) * gen_imgs * 1e-3 * (1./params.sigma * torch.exp((Eff_reshape - Eff_max)/params.sigma))
            #Gradients = Tensor(grads).unsqueeze(1) * gen_imgs * 1e-3 * (1./params.sigma * torch.exp(-(Eff_reshape - Eff_max)**2/params.sigma**2) * 2* (Eff_max - Eff_reshape)/params.sigma)


            # Train generator
            optimizer_G.zero_grad()

            #binary_penalty = params.binary_penalty_start +  (params.binary_penalty_end - params.binary_penalty_start) * (1 - (1 - normIter)**params.binary_penalty_power)
            binary_penalty = params.binary_penalty_start if params.iter < params.binary_step_iter else params.binary_penalty_end
            if params.binary == 1:
                g_loss_solver = -torch.sum(torch.mean(Gradients, dim=0).view(-1)) - torch.mean(torch.abs(gen_imgs.view(-1)) * (2.0 - torch.abs(gen_imgs.view(-1)))) * binary_penalty 
                #g_loss_solver = -torch.sum(torch.mean(Gradients, dim=0).view(-1)) - torch.mean(torch.pow(gen_imgs.view(-1), 2)) * binary_penalty
            
            else:
                g_loss_solver = -torch.sum(torch.mean(Gradients, dim=0).view(-1))

            g_loss_solver.backward()
            optimizer_G.step()


            if it % 20 == 0:
            
                generator.eval()
                outputs_imgs = sample_images(generator, 100, params) 
                generator.train()

                Binarization = torch.mean(torch.abs(outputs_imgs.view(-1)))
                Binarization_history.append(Binarization)

                diversity = torch.mean(torch.std(outputs_imgs, dim=0))
                pattern_variance.append(diversity.data)

                numImgs = 1 if params.noise_constant == 1 else 100
                img_2 = []
                Eff_2 = []

                img_2_tmp, Eff_2_tmp = PCA_analysis(generator, pca, eng, params, numImgs)
                imgs_2.append(img_2_tmp)
                Effs_2.append(Eff_2_tmp)
                
                imgs_2.append(img_2)
                Effs_2.append(Eff_2)

                Eff_mean_history.append(np.mean(Eff_2_tmp))
                utils.plot_loss_history(([], [], Eff_mean_history, pattern_variance, Binarization_history), params.output_dir)


            t.update()



