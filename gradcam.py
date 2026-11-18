"""Grad-CAM inference for the trained AgroVision EfficientNetB0 model."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import colormaps


class AgroVisionGradCAM:
    """Load once, then explain uploaded image bytes or image files."""

    def __init__(
        self,
        model_path=None,
        class_mapping_path=None,
        target_layer="top_conv",
        overlay_alpha=0.4
    ):
        module_dir = Path(__file__).resolve().parent

        if model_path is None:
            model_path = module_dir / "model.keras"

        if class_mapping_path is None:
            class_mapping_path = module_dir / "class_to_index.json"

        self.model_path = Path(model_path)
        self.class_mapping_path = Path(class_mapping_path)

        if not 0 <= overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be between 0 and 1.")

        self.overlay_alpha = float(overlay_alpha)
        self.target_layer = target_layer

        with self.class_mapping_path.open(encoding="utf-8") as file:
            self.class_to_index = json.load(file)

        self.num_classes = len(self.class_to_index)

        if self.num_classes != 38:
            raise ValueError("Expected the trained 38-class mapping.")

        if set(self.class_to_index.values()) != set(range(38)):
            raise ValueError("Class indices must cover 0 to 37.")

        self.class_names = [
            name
            for name, index in sorted(
                self.class_to_index.items(),
                key=lambda item: item[1]
            )
        ]

        self.model = tf.keras.models.load_model(
            self.model_path,
            compile=False,
            safe_mode=True
        )

        if self.model.input_shape != (None, 160, 160, 3):
            raise ValueError("Unexpected model input shape.")

        if self.model.output_shape != (None, self.num_classes):
            raise ValueError("Model outputs do not match the class mapping.")

        self.image_size = tuple(self.model.input_shape[1:3])
        backbone = self.model.get_layer("efficientnetb0")

        self.prefix_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(
                "efficientnet_input_passthrough"
            ).output
        )

        self.feature_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                backbone.get_layer(self.target_layer).output,
                backbone.output
            ]
        )

        self.pooling = self.model.get_layer("global_average_pooling")
        self.dropout = self.model.get_layer("classifier_dropout")
        self.classifier = self.model.get_layer("disease_predictions")

        if self.classifier.get_config()["activation"] != "softmax":
            raise ValueError("Expected a softmax classification layer.")

    def _load_image(self, image_source):
        if isinstance(image_source, (str, Path)):
            image_bytes = tf.io.read_file(str(image_source))
        elif isinstance(image_source, (bytes, bytearray)):
            image_bytes = tf.convert_to_tensor(
                bytes(image_source),
                dtype=tf.string
            )
        else:
            raise TypeError("Provide an image filepath or image bytes.")

        original = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        original.set_shape([None, None, 3])

        resized = tf.image.resize(
            original,
            self.image_size,
            method="bilinear",
            antialias=False
        )

        # The checkpoint includes EfficientNet preprocessing.
        images = tf.expand_dims(
            tf.cast(resized, tf.float32),
            axis=0
        )

        tf.debugging.assert_all_finite(images, "Invalid image values.")

        return original.numpy(), images

    def _forward(self, images):
        x = self.prefix_model(images, training=False)

        feature_maps, backbone_output = self.feature_model(
            x,
            training=False
        )

        x = self.pooling(backbone_output)
        x = self.dropout(x, training=False)
        x = tf.cast(x, self.classifier.compute_dtype)

        logits = tf.linalg.matmul(
            x,
            tf.cast(self.classifier.kernel, x.dtype)
        )

        if self.classifier.use_bias:
            logits = tf.nn.bias_add(
                logits,
                tf.cast(self.classifier.bias, x.dtype)
            )

        probabilities = self.classifier(x)

        return feature_maps, logits, probabilities

    def explain(self, image_source):
        """Return the prediction, heatmaps, overlay and overlay PNG bytes."""

        original, images = self._load_image(image_source)

        with tf.GradientTape() as tape:
            tape.watch(images)

            feature_maps, logits, probabilities = self._forward(images)

            predicted_index = int(
                tf.argmax(probabilities[0]).numpy()
            )
            target_score = logits[0, predicted_index]

        gradients = tape.gradient(target_score, feature_maps)

        if gradients is None:
            raise RuntimeError("Grad-CAM gradients are disconnected.")

        tf.debugging.assert_all_finite(
            feature_maps, "Invalid feature maps."
        )
        tf.debugging.assert_all_finite(
            gradients, "Invalid gradients."
        )
        tf.debugging.assert_all_finite(
            probabilities, "Invalid prediction probabilities."
        )

        channel_weights = tf.reduce_mean(
            gradients[0],
            axis=(0, 1)
        )

        raw_cam = tf.reduce_sum(
            feature_maps[0] * channel_weights,
            axis=-1
        )

        positive_cam = tf.nn.relu(raw_cam)
        positive_peak = tf.reduce_max(positive_cam)

        native_heatmap = tf.math.divide_no_nan(
            positive_cam,
            positive_peak
        ).numpy()

        resized_heatmap = tf.image.resize(
            native_heatmap[..., np.newaxis],
            original.shape[:2],
            method="bilinear",
            antialias=False
        ).numpy()[..., 0]

        resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

        has_positive_activation = bool(positive_peak.numpy() > 0)
        original_float = original.astype(np.float32) / 255.0

        if has_positive_activation:
            colours = colormaps["jet"](resized_heatmap)[..., :3]

            overlay = (
                (1.0 - self.overlay_alpha) * original_float
                + self.overlay_alpha * colours
            )
        else:
            overlay = original_float.copy()

        overlay = np.clip(overlay, 0.0, 1.0)
        overlay_uint8 = np.rint(overlay * 255.0).astype(np.uint8)

        probability_vector = probabilities.numpy()[0]

        return {
            "predicted_index": predicted_index,
            "predicted_class": self.class_names[predicted_index],
            "confidence": float(probability_vector[predicted_index]),
            "probabilities": probability_vector,
            "target_layer": self.target_layer,
            "has_positive_activation": has_positive_activation,
            "gradient_max_abs": float(
                tf.reduce_max(tf.abs(gradients)).numpy()
            ),
            "original_image": original,
            "native_heatmap": native_heatmap,
            "resized_heatmap": resized_heatmap,
            "overlay": overlay,
            "overlay_uint8": overlay_uint8,
            "overlay_png": tf.io.encode_png(overlay_uint8).numpy()
        }
