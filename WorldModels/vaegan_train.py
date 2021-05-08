import sys
import os
import tensorflow as tf
import random
import numpy as np
import sys
import os
os.chdir('/home/ubuntu/WM/WorldModels')
sys.path.append('/home/ubuntu/WM/WorldModels')
sys.path

np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
from vaegan.vaegan import VAEGAN
from utils import PARSER
args = PARSER.parse_args()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
def ds_gen():
    dirname = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                img_i = img / 255.0
                yield img_i


if __name__ == "__main__": 
    model_save_path = "results/{}/{}/tf_vae".format(args.exp_name, args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)
    shuffle_size = 20 * 1000 # only loads ~20 episodes for shuffle windows b/c im poor and don't have much RAM
    ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(args.vae_batch_size)
    ds = ds.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)

    N_Z = args.z_size
    encoder = [
        tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N_Z*2),
        ]

    decoder = [
        tf.keras.layers.Dense(units=1 * 1 * 4*256, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(1 * 1 * 4*256)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),
    ]

    def vaegan_discrim():
        inputs = tf.keras.layers.Input(shape=(64, 64, 3))
        conv1 = tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                )(inputs)
        conv2 = tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
                )(conv1)
        flatten = tf.keras.layers.Flatten()(conv2)
        lastlayer = tf.keras.layers.Dense(units=512, activation="relu")(flatten)
        outputs = tf.keras.layers.Dense(units=1, activation = None)(lastlayer)
        return inputs, lastlayer, outputs

    model = VAEGAN(
        enc = encoder,
        dec = decoder,
        vae_disc_function = vaegan_discrim,
        lr_base_gen = args.vae_learning_rate, 
        lr_base_disc = args.vae_learning_rate,
        latent_loss_div=1, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
        sig_mult = 10, # how binary the discriminator's learning rate is shifted (we squash it with a sigmoid)
        recon_loss_div = .001, # this variable will depend on your dataset - choose a number that will bring your latent loss to ~1-10
    )

    #next step: load train_datasets and test_dataset
    example_data = next(iter(ds))
    model.train(example_data)

    tensorboard_callback.set_model(model)
    loss_weights = [1.0, 1.0] # weight both the reconstruction and KL loss the same
    step = 0
    blank_batch = np.zeros([2*args.z_size])
    for i in range(args.vae_num_epoch):
        j = 0
        for x_batch in ds:
            if i == 0 and j == 0:
                model._set_inputs(x_batch)
            j += 1
            step += 1 
            model.train(x_batch)
            loss = model.compute_loss(x_batch)

            [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()] 
            if j % 100 == 0:
                output_log = 'epoch: {} mb: {}'.format(i, j)
                for loss_key, loss_val in loss.items():
                    output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
                print(output_log)
        print('saving')
        tf.keras.models.save_model(vae, model_save_path, include_optimizer=True, save_format='tf')
