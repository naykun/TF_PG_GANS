import os
import sys
import time
import glob
from model import *
from config import *
from keras.models import load_model,save_model
from keras.layers import Input
from keras import optimizers


import misc
misc.init_output_logging()


import dataset
import config

def load_GD(path,compile = False):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    G = load_model(G_path,compile = compile)
    D = load_model(D_path,compile = compile)
    return G,D

def save_GD(G,D,path,overwrite = False):
    G_path = os.path.join(path,'Generator.h5')
    D_path = os.path.join(path,'Discriminator.h5')
    save_model(G,G_path,overwrite = overwrite)
    sace_model(D,D_path,overwrite = overwrite)

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown_linear(epoch, num_epochs, rampdown_length):
    if epoch >= num_epochs - rampdown_length:
        return float(num_epochs - epoch) / rampdown_length
    else:
        return 1.0


def create_result_subdir(result_dir, run_desc):

    # Select run ID and create subdir.
    while True:
        run_id = 0
        for fname in glob.glob(os.path.join(result_dir, '*')):
            try:
                fbase = os.path.basename(fname)
                ford = int(fbase[:fbase.find('-')])
                run_id = max(run_id, ford + 1)
            except ValueError:
                pass

        result_subdir = os.path.join(result_dir, '%03d-%s' % (run_id, run_desc))
        try:
            os.makedirs(result_subdir)
            break
        except OSError:
            if os.path.isdir(result_subdir):
                continue
            raise

    print ("Saving results to", result_subdir)
    return result_subdir

def format_time(seconds):
    s = int(np.round(seconds))
    if s < 60:         return '%ds'                % (s)
    elif s < 60*60:    return '%dm %02ds'          % (s / 60, s % 60)
    elif s < 24*60*60: return '%dh %02dm %02ds'    % (s / (60*60), (s / 60) % 60, s % 60)
    else:              return '%dd %dh %02dm'      % (s / (24*60*60), (s / (60*60)) % 24, (s / 60) % 60)

def random_latents(num_latents, G_input_shape):
    return np.random.randn(num_latents, *G_input_shape[1:]).astype(np.float32)

def random_labels(num_labels, training_set):
    return training_set.labels[np.random.randint(training_set.labels.shape[0], size=num_labels)]

def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)


def load_dataset(dataset_spec=None, verbose=True, **spec_overrides):
    if verbose: print('Loading dataset...')
    if dataset_spec is None: dataset_spec = config.dataset
    dataset_spec = dict(dataset_spec) # take a copy of the dict before modifying it
    dataset_spec.update(spec_overrides)
    dataset_spec['h5_path'] = os.path.join(config.data_dir, dataset_spec['h5_path'])
    if 'label_path' in dataset_spec: dataset_spec['label_path'] = os.path.join(config.data_dir, dataset_spec['label_path'])
    training_set = dataset.Dataset(**dataset_spec)
    #

    #print("****************************************")
    #print("原有shape：",training_set.shape)
    #测试，直接交换两个维度，看效果
    #training_set=np.swapaxes(training_set,1,2)
    #training_set=np.swapaxes(training_set,2,3)
    #print("变换后shape：",training_set.shape)
    
    #
    if verbose: print('Dataset shape =', np.int32(training_set.shape).tolist())
    drange_orig = training_set.get_dynamic_range()
    if verbose: print('Dynamic range =', drange_orig)
    return training_set, drange_orig

speed_factor = 10

def train_gan(
    separate_funcs          = False,
    D_training_repeats      = 1,
    G_learning_rate_max     = 0.0010,
    D_learning_rate_max     = 0.0010,
    G_smoothing             = 0.999,
    adam_beta1              = 0.0,
    adam_beta2              = 0.99,
    adam_epsilon            = 1e-8,
    minibatch_default       = 16,
    minibatch_overrides     = {},
    rampup_kimg             = 40/speed_factor,
    rampdown_kimg           = 0,
    lod_initial_resolution  = 4,
    lod_training_kimg       = 400/speed_factor,
    lod_transition_kimg     = 400/speed_factor,
    total_kimg              = 10000/speed_factor,
    dequantize_reals        = False,
    gdrop_beta              = 0.9,
    gdrop_lim               = 0.5,
    gdrop_coef              = 0.2,
    gdrop_exp               = 2.0,
    drange_net              = [-1,1],
    drange_viz              = [-1,1],
    image_grid_size         = 5,
    tick_kimg_default       = 50/speed_factor,
    tick_kimg_overrides     = {32:20, 64:10, 128:10, 256:5, 512:2, 1024:1},
    image_snapshot_ticks    = 1,
    network_snapshot_ticks  = 40,
    image_grid_type         = 'default',
    resume_network          = None,
    resume_kimg             = 0.0,
    resume_time             = 0.0):

    training_set, drange_orig = load_dataset()

   
    #这部分，train_set.shape 经过了改动
    if resume_network:
        print("Resuming form"+resume_network)
        G,D = resume(os.path.join((config.result_dir,resume_network)))
    else:
        #channel last
        print("Creating new G & D network...")
        G = Generator(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.G)
        D = Discriminator(num_channels=training_set.shape[1], resolution=training_set.shape[2], label_size=training_set.labels.shape[1], **config.D)
        #missing Gs
    
    #print("Debug"*20)

    pg_GAN = PG_GAN(G,D,config.G['latent_size'],0)    
    
    #print("Debug"*20)

    print(G.summary())
    print(D.summary())
    print(pg_GAN.summary())

    # Misc init.
    resolution_log2 = int(np.round(np.log2(G.output_shape[2])))
    initial_lod = max(resolution_log2 - int(np.round(np.log2(lod_initial_resolution))), 0)
    cur_lod = 0.0
    min_lod, max_lod = -1.0, -2.0
    fake_score_avg = 0.0

    result_subdir = create_result_subdir(config.result_dir, config.run_desc)

    G_opt = optimizers.Adam(lr = 0.0,beta_1=adam_beta1,beta_2=adam_beta2,epsilon = adam_epsilon)
    D_opt = optimizers.Adam(lr = 0.0,beta_1 = adam_beta1,beta_2 = adam_beta2,epsilon = adam_epsilon)
    # GAN_opt = optimizers.Adam(lr = 0.0,beta_1 = 0.0,beta_2 = 0.99)
    
    


    if config.loss['type']=='wass':
        G_loss_func = wasserstein_loss
        D_loss_func = wasserstein_loss

    G.compile(G_opt,loss=G_loss_func)
    D.trainable = False
    pg_GAN.compile(G_opt,loss = D_loss_func)
    D.trainable = True
    D.compile(D_opt,loss=D_loss_func)

    #real_image_input = Input((training_set.shape[2],training_set.shape[2],training_set.shape[-1]),name = "real_image_input")
    #real_label_input = Input((training_set.labels.shape[1]),name = "real_label_input")
    #fake_latent_input = Input((config.G['latent_size']),name = "fake_latent_input")
    #fake_labels_input = Input((training_set.labels.shape[1]),name = "fake_label_input")

    snapshot_fake_latents = random_latents(np.prod(image_grid_size), G.input_shape)

    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    tick_train_out = []
    train_start_time = tick_start_time - resume_time


    while cur_nimg < total_kimg * 1000:
        
        # Calculate current LOD.
        cur_lod = initial_lod
        if lod_training_kimg or lod_transition_kimg:
            tlod = (cur_nimg / (1000.0/speed_factor)) / (lod_training_kimg + lod_transition_kimg)
            cur_lod -= np.floor(tlod)
            if lod_transition_kimg:
                cur_lod -= max(1.0 + (np.fmod(tlod, 1.0) - 1.0) * (lod_training_kimg + lod_transition_kimg) / lod_transition_kimg, 0.0)
            cur_lod = max(cur_lod, 0.0)

        # Look up resolution-dependent parameters.
        cur_res = 2 ** (resolution_log2 - int(np.floor(cur_lod)))
        minibatch_size = minibatch_overrides.get(cur_res, minibatch_default)
        tick_duration_kimg = tick_kimg_overrides.get(cur_res, tick_kimg_default)


        # Update network config.
        lrate_coef = rampup(cur_nimg / 1000.0, rampup_kimg)
        lrate_coef *= rampdown_linear(cur_nimg / 1000.0, total_kimg, rampdown_kimg)
        #G_lrate.set_value(np.float32(lrate_coef * G_learning_rate_max))
        K.set_value(G.optimizer.lr, np.float32(lrate_coef * G_learning_rate_max))
        K.set_value(pg_GAN.optimizer.lr, np.float32(lrate_coef * G_learning_rate_max))
        #D_lrate.set_value(np.float32(lrate_coef * D_learning_rate_max))
        K.set_value(D.optimizer.lr, np.float32(lrate_coef * D_learning_rate_max))
        if hasattr(G, 'cur_lod'): K.set_value(G.cur_lod,np.float32(cur_lod))
        if hasattr(D, 'cur_lod'): K.set_value(D.cur_lod,np.float32(cur_lod))


        new_min_lod, new_max_lod = int(np.floor(cur_lod)), int(np.ceil(cur_lod))
        if min_lod != new_min_lod or max_lod != new_max_lod:
            min_lod, max_lod = new_min_lod, new_max_lod
        #if min_lod != new_min_lod or max_lod != new_max_lod:
        #    min_lod, max_lod = new_min_lod, new_max_lod

        #    # Pre-process reals.
        #    real_images_expr = real_images_var
        #    if dequantize_reals:
        #        epsilon_noise = K.random_uniform_variable(real_image_input.shape(), low=-0.5, high=0.5, dtype='float32', seed=np.random.randint(1, 2147462579))
        #        epsilon_noise = rnd.uniform(size=real_images_expr.shape, low=-0.5, high=0.5, dtype='float32')
        #        real_images_expr = T.cast(real_images_expr, 'float32') + epsilon_noise # match original implementation of Improved Wasserstein
        #    real_images_expr = misc.adjust_dynamic_range(real_images_expr, drange_orig, drange_net)
        #    if min_lod > 0: # compensate for shrink_based_on_lod
        #        real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=2)
        #        real_images_expr = T.extra_ops.repeat(real_images_expr, 2**min_lod, axis=3)
        
        # train D
        d_loss = None
        for idx in range(D_training_repeats):
            mb_reals, mb_labels = training_set.get_random_minibatch(minibatch_size, lod=cur_lod, shrink_based_on_lod=True, labels=True)
            mb_latents = random_latents(minibatch_size,G.input_shape)
            mb_labels_rnd = random_labels(minibatch_size,training_set)
            if min_lod > 0: # compensate for shrink_based_on_lod
                 mb_reals = np.repeat(mb_reals, 2**min_lod, axis=1)
                 mb_reals = np.repeat(mb_reals, 2**min_lod, axis=2)
            #mb_fakes = G.predict_on_batch([mb_latents,mb_labels_rnd])
            mb_fakes = G.predict_on_batch([mb_latents])

            d_loss_real = D.train_on_batch(mb_reals, -np.ones((mb_reals.shape[0],1,1,1)))
            d_loss_fake = D.train_on_batch(mb_fakes, np.ones((mb_fakes.shape[0],1,1,1)))
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            cur_nimg += minibatch_size

        #train G

        mb_latents = random_latents(minibatch_size,G.input_shape)
        mb_labels_rnd = random_labels(minibatch_size,training_set)

        g_loss = pg_GAN.train_on_batch([mb_latents], -np.ones((mb_latents.shape[0],1,1,1)))

        print ("%d [D loss: %f] [G loss: %f]" % (cur_nimg, 1 - d_loss, 1 - g_loss))
        #print(cur_nimg)
        #print(g_loss)
        #print(d_loss)
        # Fade in D noise if we're close to becoming unstable
        fake_score_cur = np.clip(np.mean(d_loss), 0.0, 1.0)
        fake_score_avg = fake_score_avg * gdrop_beta + fake_score_cur * (1.0 - gdrop_beta)
        gdrop_strength = gdrop_coef * (max(fake_score_avg - gdrop_lim, 0.0) ** gdrop_exp)
        if hasattr(D, 'gdrop_strength'): K.set_value(D.gdrop_strength,np.float32(gdrop_strength))

        if cur_nimg >= tick_start_nimg + tick_duration_kimg * 1000 or cur_nimg >= total_kimg * 1000:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            tick_start_time = cur_time
            tick_train_avg = tuple(np.mean(np.concatenate([np.asarray(v).flatten() for v in vals])) for vals in zip(*tick_train_out))
            tick_train_out = []

            # Print progress.
            print (

                '''tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-9.1f sec/kimg %-6.1f Dgdrop %-8.4f''' % 
                (

                    cur_tick, (cur_nimg / 1000.0), cur_lod, minibatch_size, 
                    format_time(cur_time - train_start_time), tick_time, 
                    tick_time / tick_kimg, 
                    gdrop_strength
                    #tick_train_avg
                )
                       
            )

            # Visualize generated images.
            if cur_tick % image_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                snapshot_fake_images = G.predict_on_batch(snapshot_fake_latents)
                misc.save_image_grid(snapshot_fake_images, os.path.join(result_subdir, 'fakes%06d.png' % (cur_nimg / 1000)), drange=drange_viz, grid_size=image_grid_size)

            if cur_tick % network_snapshot_ticks == 0 or cur_nimg >= total_kimg * 1000:
                save_GD(G,D,os.path.join(result_subdir, 'network-snapshot-%06d' % (cur_nimg / 1000)),overwrite = False)


    save_GD(G,D,os.path.join(result_subdir, 'network-final'))
    training_set.close()
    print('Done.')


    

if __name__ == '__main__':
    #指定随机种子
    np.random.seed(config.random_seed)
    func_params = config.train
    #config.train 为学习率等参数的设置字典
    func_name = func_params['func']
    del func_params['func']
    globals()[func_name](**func_params)
