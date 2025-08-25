import tensorflow as tf # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, GlobalAveragePooling1D, Multiply # type: ignore


def build_dense(cfg):
        N = cfg["data"]["maxNPF"]
        x_cont = tf.keras.Input(shape=(N, 5), name="continuous_inputs")
        x_pxpy = tf.keras.Input(shape=(N, 2), name="momentum_inputs")

        inputs = [x_cont, x_pxpy]

        # TODO: implement option for embedding layer
        
        # Dense layer
        x = x_cont
        for i, u in enumerate(cfg["model"]["units"]):
            x = Dense(u, activation=None, kernel_initializer="lecun_uniform", name=f"dense_{i}")(x)
            x = BatchNormalization(momentum=0.95, name=f"bn_{i}")(x)
            x = Activation(cfg["model"]["activation"], name=f"act_{i}")(x)

        if cfg["training"]["mode"] == 0: # TODO: implement mode 0
            x = GlobalAveragePooling1D(name="pool")(x)
            out = Dense(2, activation="linear", name="output")(x)

        elif cfg["training"]["mode"] == 1:
            w = Dense(1, activation="linear", name="met_weight")(x)      # (B,N,1)
            
            x = Multiply()([w, x_pxpy]) 

            x = GlobalAveragePooling1D(name="output")(x)
        else: # TODO: implement mode 2
            pooled = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)               # (B,C)
            w = tf.keras.layers.Dense(1, activation="linear", name="puppi_weight")(pooled)# (B,1)
            #out = w * tf.reduce_sum(x_pxpy, axis=1, name="pmet")                          # (B,2)

        outputs =x
        
        model = Model(inputs=inputs, outputs=outputs) # TODO:pass inputs as dict to match config?
        #TODO:check MET weight minus one need

        return model