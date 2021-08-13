
from keras.models import Model
from keras.layers import Input, LeakyReLU, MaxPool1D, LSTM,  TimeDistributed, Dense, Reshape, Flatten
from keras.layers import UpSampling2D, Conv2DTranspose, Lambda
from keras.layers import Conv1D, Bidirectional
import keras.backend as K
import tensorflow as tf

def temporal_autoencoder(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):

    assert(timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    # Encoder
    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear', name='Conv_encode')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='sum', name='LSTM1')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='sum', name='LSTM2')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)  
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)  

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, n_units[1]), name='decoder_input')

    # Internal layers in decoder
    decoded = autoencoder.get_layer('reshape')(encoded_input)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder


def temporal_autoencoder_lstm_ae(input_dim, timesteps, n_units=[50, 1]):
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    encoded = LSTM(n_units[0], return_sequences=True)(x)
    encoded = LeakyReLU(name='latent')(encoded)

    decoded = LSTM(n_units[0], return_sequences=True, name='LSTM')(encoded)
    decoded = LeakyReLU(name='act')(decoded)
    decoded = TimeDistributed(Dense(units=input_dim), name='dense')(decoded)  # sequence labeling
    output = Reshape((-1, input_dim), name='output_seq')(decoded)

    autoencoder = Model(inputs=x, outputs=output, name='AE')

    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    encoded_input = Input(shape=(timesteps,n_units[0]), name='decoder_input')

    decoded = autoencoder.get_layer('LSTM')(encoded_input)
    decoded = autoencoder.get_layer('act')(decoded)
    decoded = autoencoder.get_layer('dense')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')
    return autoencoder, encoder, decoder


def sampling(args):
    z_mean, z_log_var = args
    batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def temporal_autoencoder_vae(input_dim, timesteps, n_units=[1024, 256]):
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    encoded = Flatten()(x)
    encoded = Dense(n_units[0], activation='relu')(encoded)
    z_mean = Dense(n_units[1], name='z_mean')(encoded)
    z_log_var = Dense(n_units[1], name='z_log_var')(encoded)
    z = Lambda(sampling, output_shape=(n_units[1],))([z_mean, z_log_var])
    encoded_out = Reshape((n_units[1], -1))(z)

    decoded = Dense(n_units[0],activation='relu', name='dense1')(z)
    decoded = Dense(input_dim, activation='sigmoid', name='dense2')(decoded)
    decoded = Dense(input_dim*timesteps,activation='sigmoid', name='dense3')(decoded)
    output = Reshape((timesteps, input_dim), name='output_seq')(decoded)

    autoencoder = Model(inputs=x, outputs=output, name='AE')
    encoder = Model(inputs=x, outputs=encoded_out, name='encoder')

    encoded_input = Input(shape=(n_units[1],))

    decoded = autoencoder.get_layer('dense1')(encoded_input)
    decoded = autoencoder.get_layer('dense2')(decoded)
    decoded = autoencoder.get_layer('dense3')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    decoder = Model(inputs=encoded_input,
                    outputs=decoder_output, name='decoder')
    return autoencoder, encoder, decoder


def temporal_autoencoder_cnn_ae(input_dim, timesteps, n_filters=50, kernel_size=10, strides=1, pool_size=10, n_units=[50, 1]):

    assert(timesteps % pool_size == 0)
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    encoded = Conv1D(n_filters, kernel_size, strides=strides, padding='same', activation='linear')(x)
    encoded = LeakyReLU()(encoded)
    encoded = MaxPool1D(pool_size)(encoded)
    encoded = Dense(n_units[0], activation='relu')(encoded)
    encoded = Dense(n_units[1], activation='relu')(encoded)

    # Decoder
    decoded = Reshape((-1, 1, n_units[1]), name='reshape')(encoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)
    decoded = Conv2DTranspose(
        input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(
        shape=(timesteps // pool_size, 1), name='decoder_input')

    # Internal layers in decoder
    decoded = autoencoder.get_layer('reshape')(encoded_input)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input,
                    outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder


def temporal_autoencoder_sae(input_dim, timesteps, n_units=[256, 2]):
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    encoded = Flatten()(x)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Reshape((32,-1))(encoded)

    decoded = Flatten()(encoded)
    decoded = Dense(128, activation='relu', name='dense1')(decoded)
    decoded = Dense(256, activation='relu', name='dense2')(decoded)
    decoded = Dense(units=input_dim*timesteps, name='dense')(decoded)
    output = Reshape((timesteps, input_dim), name='output_seq')(decoded)

    autoencoder = Model(inputs=x, outputs=output, name='AE')
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    encoded_input = Input(shape=(32,))

    decoded = autoencoder.get_layer('dense1')(encoded_input)
    decoded = autoencoder.get_layer('dense2')(decoded)
    decoded = autoencoder.get_layer('dense')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    decoder = Model(inputs=encoded_input,
                    outputs=decoder_output, name='decoder')
    return autoencoder, encoder, decoder
