from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def build_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    class_output = Dense(1, activation='sigmoid', name="class_output")(x)
    bbox_output = Dense(4, activation='sigmoid', name="bbox_output")(x)

    model = Model(inputs=base.input, outputs=[class_output, bbox_output])
    return model