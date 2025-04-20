import tensorflow as tf
from skimage.morphology import skeletonize

def gap_dice_loss(alpha=5.0, sigma=5.0, smooth=1e-6):
    def loss(y_true, y_pred):
        # Convert y_true to binary mask (if not already)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        
        # Compute skeleton and distance maps for GAP weights
        def compute_gap_weights(mask):
            # Skeletonization
            skeleton = skeletonize(mask.numpy())
            
            # Find endpoints (pixels with 1 neighbor)
            kernel = [[1,1,1],
                      [1,0,1],
                      [1,1,1]]
            neighbors = tf.nn.conv2d(tf.expand_dims(skeleton, [0,-1]), 
                                   tf.constant(kernel, dtype=tf.float32)[...,tf.newaxis], 
                                   strides=1, padding="SAME")
            endpoints = tf.cast((skeleton > 0) & (neighbors[0,...,0] == 1, tf.float32))
            
            # Distance transform from endpoints
            distance_map = tf.numpy_function(
                lambda x: distance_transform_edt(~x.astype(bool)),
                endpoints, tf.float32
            )
            
            # Weight map calculation
            return 1.0 + alpha * tf.exp(-(distance_map**2)/(2*sigma**2))

        # Process each sample in batch
        weights = tf.map_fn(compute_gap_weights, y_true[...,0], fn_output_signature=tf.float32)
        weights = tf.expand_dims(weights, -1)

        # GAP Loss (Weighted Cross-Entropy)
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        gap_loss = tf.reduce_mean(weights * ce)

        # Dice Loss
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
        union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
        dice_loss = 1.0 - tf.reduce_mean((2.*intersection + smooth)/(union + smooth))

        # Combined loss
        return gap_loss + dice_loss

    return loss