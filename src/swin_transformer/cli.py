# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate
# from keras_swin_unet import swin_layers
# from keras_swin_unet import transformer_layers
# import matplotlib.pyplot as plt
# import argparse
# import json
# from PIL import Image
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
# from tensorflow.keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import load_model
# from skimage.util import view_as_windows
# from scipy.ndimage import gaussian_filter
# from matplotlib.patches import Patch
# import warnings
# from tensorflow.keras.utils import Sequence
# import time
# from swin_transformer.AUC_LOSS import auc_focal_loss
# from swin_transformer.model_loader  import get_model
# from swin_transformer.split_data import split_dataset
# from swin_transformer.data_loader import DynamicDataLoader
# import os
# import numpy as np
# from tensorflow.keras.utils import Sequence
# from tensorflow.keras.utils import to_categorical
# from PIL import Image
# import warnings

# def compute_iou(true_mask, pred_mask, num_classes):
#     """Calculate Intersection over Union (IoU) for each class."""
#     iou_per_class = []
#     for class_id in range(num_classes):
#         intersection = np.sum((true_mask == class_id) & (pred_mask == class_id))
#         union = np.sum((true_mask == class_id) | (pred_mask == class_id))
#         iou = intersection / union if union != 0 else 0  # Avoid division by zero
#         iou_per_class.append(iou)
#     return iou_per_class
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_comparison(k, X_images, y, refined_segmentation, batch_idx, num_classes,
#                          font_size=12, font_style='DejaVu Sans', xtick_size=10, ytick_size=10):
#     """Visualizes the original image, predicted mask, actual mask, and TP, FP, FN, TN with custom styling."""

#     # Set font size and style for titles and labels
#     plt.rcParams.update({'font.size': font_size, 'font.family': font_style})

#     plt.figure(figsize=(12, 12))

#     # Original Image
#     plt.subplot(2, 2, 1)
#     plt.imshow(X_images[batch_idx])  # Indexing the batch here
#     plt.title("Original Image", fontsize=font_size, fontweight='bold')
#     plt.axis('off')

#     # Predicted Mask
#     proposed_mask = np.argmax(refined_segmentation[batch_idx], axis=-1)  # argmax to get 2D mask
#     plt.subplot(2, 2, 2)
#     plt.imshow(X_images[batch_idx])
#     plt.imshow(proposed_mask, alpha=0.5, cmap='coolwarm')  # Predicted Mask Overlay
#     plt.title("Predicted Mask", fontsize=font_size, fontweight='bold')
#     plt.axis('off')

#     # Actual Mask (Ground truth)
#     actual_mask = np.argmax(y[batch_idx], axis=-1)  # Ground truth mask
#     plt.subplot(2, 2, 3)
#     plt.imshow(X_images[batch_idx])
#     plt.imshow(actual_mask, alpha=0.5, cmap='coolwarm')  # Actual Mask Overlay
#     plt.title("Actual Mask (Ground Truth)", fontsize=font_size, fontweight='bold')
#     plt.axis('off')

#     # Overlay TP, FP, FN, TN
#     TP = (proposed_mask == actual_mask) & (actual_mask == 1)  # True Positives
#     FP = (proposed_mask != actual_mask) & (proposed_mask == 1)  # False Positives
#     FN = (proposed_mask != actual_mask) & (actual_mask == 1)  # False Negatives
#     TN = (proposed_mask == actual_mask) & (actual_mask == 0)  # True Negatives

#     image_with_metrics = X_images[batch_idx].copy()
#     image_with_metrics[TP] = [0, 255, 0]  # Green for TP
#     image_with_metrics[FP] = [255, 0, 0]  # Red for FP
#     image_with_metrics[FN] = [0, 0, 255]  # Blue for FN
#     image_with_metrics[TN] = [128, 128, 128]  # Gray for TN

#     plt.subplot(2, 2, 4)
#     plt.imshow(image_with_metrics)

#     # Custom title with corresponding colors for TP, FP, FN, TN (colored text in title)
#     plt.text(0.35, 1.05, 'TP', color='green', fontsize=font_size, fontweight='bold', ha='center', transform=plt.gca().transAxes)
#     plt.text(0.45, 1.05, 'FP', color='red', fontsize=font_size, fontweight='bold', ha='center', transform=plt.gca().transAxes)
#     plt.text(0.55, 1.05, 'FN', color='blue', fontsize=font_size, fontweight='bold', ha='center', transform=plt.gca().transAxes)
#     plt.text(0.65, 1.05, 'TN', color='gray', fontsize=font_size, fontweight='bold', ha='center', transform=plt.gca().transAxes)

#     # Center the main title with colored labels
#     plt.title(
#         "",
#         fontsize=font_size,
#         fontweight='bold',
#         loc='center'  # This ensures the title is centered
#     )

#     plt.axis('off')

#     # Set the x and y ticks sizes
#     plt.xticks(fontsize=xtick_size)
#     plt.yticks(fontsize=ytick_size)

#     # Save the comparison plot
#     plt.tight_layout()
#     plt.savefig(f'results_comparison_{batch_idx + 1}_{k}.png', bbox_inches='tight')
#     k = k + 1
#     plt.close()
#     return k

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='swin-unet')
#     parser.add_argument("--model_dir", type=str, default='./checkpoint/', help='path to the save or load the checkpoint')
#     parser.add_argument("--data", type=str, default='./data/', help='Dataset location')
#     parser.add_argument("--num_classes", type=int, default=2, help='number of classes')
#     parser.add_argument("--inps", type=str, default='train', help='select test, train, infer')
#     parser.add_argument("--b_s", type=int, default=64, help="Batch Size")
#     parser.add_argument("--e", type=int, default=100, help="Epochs")
#     parser.add_argument("--p", type=int, default=3, help="Early stop patience")
#     parser.add_argument("--filter_num_begin", type=int, default=128, help='Number of channels in the first downsampling block')
#     parser.add_argument("--depth", type=int, default=4, help='Depth of SwinUNET')
#     parser.add_argument("--stack_num_down", type=int, default=2, help='Number of Swin Transformers per downsampling level')
#     parser.add_argument("--stack_num_up", type=int, default=2, help='Number of Swin Transformers per upsampling level')
#     parser.add_argument("--patch_size", type=int, nargs='+', default=[4, 4], help='Patch size for the input image')
#     parser.add_argument("--num_heads", type=int, nargs='+', default=[4, 8, 8, 8], help='Number of attention heads per down/upsampling level')
#     parser.add_argument("--window_size", type=int, nargs='+', default=[4, 2, 2, 2], help='Size of attention window per down/upsampling level')
#     parser.add_argument("--num_mlp", type=int, default=512, help='Number of MLP nodes within the Transformer')
#     parser.add_argument("--input_scale", type=int, default=255, help="Scaling factor for input images (65536 for 16-bit, 255 for 8-bit)")
#     parser.add_argument("--mask_scale", type=int, default=255, help="Scaling factor for mask images (65536 for 16-bit, 255 for 8-bit)")
#     parser.add_argument("--input_shape", type=int, nargs=3, default=[512, 512, 3], help="Shape of input images (height, width, channels)")

#     parser.add_argument('--gamma', type=float, required=True, help='Gamma value for focal loss.')
#     parser.add_argument('--alpha', type=float, required=True, help='Alpha value for focal loss.')
#     parser.add_argument("--maxSavedSamples", type=int, default=10, help="How many visual images you want to save during inference")
#     args = parser.parse_args()

#     all_image_ids = os.listdir(os.path.join(args.data, 'images'))
#     train_ids, val_ids, test_ids = split_dataset(all_image_ids, train_frac=0.8, val_frac=0.1, test_frac=0.1)

#     if args.inps == 'train':
#         train_loader = DynamicDataLoader(
#             data_dir=args.data,
#             ids=train_ids,
#             batch_size=args.b_s,
#             img_size=(512, 512),
#             mode='train',
#             image_dtype=np.float32,
#             mask_dtype=np.int32,
#             num_classes=args.num_classes,
#             input_scale=args.input_scale,
#             mask_scale=args.mask_scale
#         )

#         val_loader = DynamicDataLoader(
#             data_dir=args.data,
#             ids=val_ids,
#             batch_size=args.b_s,
#             img_size=(512, 512),
#             mode='val',
#             image_dtype=np.float32,
#             mask_dtype=np.int32,
#             num_classes=args.num_classes,
#             input_scale=args.input_scale,
#             mask_scale=args.mask_scale
#         )
#         test_loader = DynamicDataLoader(
#             data_dir=args.data,
#             ids=test_ids,
#             batch_size=args.b_s,
#             img_size=(512, 512),
#             mode='test',
#             image_dtype=np.float32,
#             mask_dtype=np.int32,
#             num_classes=args.num_classes,
#             input_scale=args.input_scale,
#             mask_scale=args.mask_scale
#         )
#         if not os.path.exists(args.model_dir):
#             os.makedirs(args.model_dir)
#         model = get_model(
#             input_size=tuple(args.input_shape),
#             filter_num_begin=args.filter_num_begin,
#             args.depth,
#             args.stack_num_down,
#             args.stack_num_up,
#             args.patch_size,
#             args.num_heads,
#             args.window_size,
#             args.num_mlp,
#             args.num_classes

#         )
#         model.summary()
#         opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
#         auc_focal_loss_fn = auc_focal_loss(alpha=args.alpha, gamma=args.gamma)
#         model.compile(optimizer=opt, loss=auc_focal_loss_fn, metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])

#         callbacks = [
#             EarlyStopping(monitor='val_loss', mode='min', patience=args.p, verbose=1, restore_best_weights=True),
#             ModelCheckpoint(filepath=os.path.join(args.model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True, verbose=1, mode='min')
#         ]

#         history = model.fit(train_loader,
#                             validation_data=val_loader,
#                             epochs=args.e,
#                             callbacks=callbacks)

#         # model.save(os.path.join(args.model_dir, 'final_model.keras'))

#         plt.plot(history.history['loss'], label='Training loss')
#         plt.plot(history.history['val_loss'], label='Validation loss')
#         plt.title('Model Loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(loc="upper left")
#         plt.savefig('loss.png', format='png', dpi=300, bbox_inches='tight')

#         y_true = []
#         y_pred = []
#         for X, y in test_loader:
#             preds = model.predict(X)
#             y_true.extend(np.argmax(y, axis=-1).flatten())
#             y_pred.extend(np.argmax(preds, axis=-1).flatten())

#         accuracy = accuracy_score(y_true, y_pred)
#         f1 = f1_score(y_true, y_pred, average='weighted')
#         precision = precision_score(y_true, y_pred, average='weighted')
#         recall = recall_score(y_true, y_pred, average='weighted')
#         auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), tf.keras.utils.to_categorical(y_pred), multi_class='ovr')

#         cm = confusion_matrix(y_true, y_pred)
#         model_path = os.path.join(args.model_dir, 'best_model.keras')
#         model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB

#         metrics = {
#             'Accuracy': accuracy,
#             'F1 Score': f1,
#             'Precision': precision,
#             'Recall': recall,
#             'AUC': auc,
#             'Confusion Matrix': cm.tolist(),
#             'Model Size (MB)': model_size
#         }
#         with open('model_evaluation_metrics.json', 'w') as f:
#             json.dump(metrics, f, indent=4)
#         print("Evaluation metrics saved to model_evaluation_metrics.json")
#     elif args.inps == 'infer':
#         k = 0
#         # Load test data.
#         test_loader = DynamicDataLoader(
#             data_dir=args.data,
#             ids=test_ids,
#             batch_size=args.b_s,
#             img_size=(512, 512),
#             mode='test',
#             image_dtype=np.float32,
#             mask_dtype=np.int32,
#             num_classes=args.num_classes,
#             input_scale=args.input_scale,
#             mask_scale=args.mask_scale
#         )

#         # Load the saved model.
#         auc_focal_loss_fn = auc_focal_loss(alpha=0.25, gamma=2.0)
#         custom_objects = {
#             'auc_focal_loss_fixed': auc_focal_loss_fn,
#             'AUC': tf.keras.metrics.AUC(name='auc'),
#             'patch_extract': transformer_layers.patch_extract,
#             'patch_embedding': transformer_layers.patch_embedding,
#             'SwinTransformerBlock': swin_layers.SwinTransformerBlock,
#             'patch_merging': transformer_layers.patch_merging,
#             'patch_expanding': transformer_layers.patch_expanding,
#         }

#         model = load_model(os.path.join(args.model_dir, 'best_model.keras'), custom_objects=custom_objects)

#         # Prepare for plotting comparison between predicted and actual masks for first 30 images
#         num_images_to_plot = 30
#         plot_count = 0

#         # Loop through images in the test loader
#         y_true_all = []
#         y_pred_all = []
#         total_time = 0
#         num_examples = 0

#         for index, (X_batch, y_batch) in enumerate(test_loader):
#             preds = model.predict(X_batch)

#             # Loop through the first 30 images in the batch (for visualization only)
#             for i in range(min(num_images_to_plot - plot_count, X_batch.shape[0])):  # Plot only up to 30 images
#                 y_true_img = np.argmax(y_batch[i], axis=-1)  # True labels
#                 y_pred_img = np.argmax(preds[i], axis=-1)  # Predicted labels

#                 y_true_img = y_true_img.astype(np.uint8)
#                 y_pred_img = y_pred_img.astype(np.uint8)

#                 # Visualize comparison for the first 30 images
#                 k = visualize_comparison(k, X_images=X_batch, y=y_batch, refined_segmentation=preds, batch_idx=i, num_classes=args.num_classes)
#                 plot_count += 1

#                 if plot_count >= num_images_to_plot:
#                     break

#             # Update y_true_all and y_pred_all after each batch (for metrics calculation)
#             y_true_all.extend(np.argmax(y_batch, axis=-1).flatten())  # Flatten true labels
#             y_pred_all.extend(np.argmax(preds, axis=-1).flatten())  # Flatten predicted labels

#             # Track prediction time for the batch
#             start_time = time.time()
#             model.predict(X_batch)
#             total_time += (time.time() - start_time)
#             num_examples += X_batch.shape[0]

#             # Stop after processing the first 30 images for visualization
#             if plot_count >= args.maxSavedSamples:
#                 break

#         # Calculate overall metrics after all batches are processed
#         accuracy = accuracy_score(y_true_all, y_pred_all)
#         f1 = f1_score(y_true_all, y_pred_all, average='macro')
#         precision = precision_score(y_true_all, y_pred_all, average='macro')
#         recall = recall_score(y_true_all, y_pred_all, average='macro')
#         auc = roc_auc_score(tf.keras.utils.to_categorical(y_true_all), tf.keras.utils.to_categorical(y_pred_all), multi_class='ovr')

#         cm = confusion_matrix(y_true_all, y_pred_all)

#         # Get model size in MB
#         model_path = os.path.join(args.model_dir, 'best_model.keras')
#         model_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB

#         # Calculate average prediction time per example in milliseconds
#         average_time_per_example = (total_time / num_examples) * 1000  # Convert to milliseconds

#         # Create the metrics dictionary
#         metrics = {
#             'Accuracy': accuracy,
#             'F1 Score': f1,
#             'Precision': precision,
#             'Recall': recall,
#             'AUC': auc,
#             'Confusion Matrix': cm.tolist(),
#             'Model Size (MB)': model_size,
#             'Average Prediction Time per Example (ms)': average_time_per_example
#         }

#         # Save metrics to a JSON file
#         with open('model_evaluation_metrics.json', 'w') as f:
#             json.dump(metrics, f, indent=4)

#         print("Evaluation metrics saved to model_evaluation_metrics.json")
#!/usr/bin/env python
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from swin_transformer.model_loader import get_model
from swin_transformer.split_data import split_dataset
from swin_transformer.data_loader import DynamicDataLoader
from swin_transformer.AUC_LOSS import auc_focal_loss
from keras_swin_unet import transformer_layers, swin_layers


def compute_iou(true_mask, pred_mask, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        inter = np.sum((true_mask == cls) & (pred_mask == cls))
        union = np.sum((true_mask == cls) | (pred_mask == cls))
        iou_per_class.append(inter / union if union else 0)
    return iou_per_class


def visualize_comparison(k, X_images, y_true, y_pred_logits, batch_idx, num_classes):
    plt.figure(figsize=(12, 12))
    img = X_images[batch_idx]
    pred_mask = np.argmax(y_pred_logits[batch_idx], axis=-1)
    true_mask = np.argmax(y_true[batch_idx], axis=-1)

    # 1 – Original
    ax = plt.subplot(2, 2, 1)
    ax.imshow(img)
    ax.set_title("Original")
    ax.axis("off")

    # 2 – Prediction
    ax = plt.subplot(2, 2, 2)
    ax.imshow(img)
    ax.imshow(pred_mask, alpha=0.5, cmap="coolwarm")
    ax.set_title("Predicted")
    ax.axis("off")

    # 3 – Ground Truth
    ax = plt.subplot(2, 2, 3)
    ax.imshow(img)
    ax.imshow(true_mask, alpha=0.5, cmap="coolwarm")
    ax.set_title("Ground Truth")
    ax.axis("off")

    # 4 – TP/FP/FN/TN overlay
    TP = (pred_mask == true_mask) & (true_mask == 1)
    FP = (pred_mask != true_mask) & (pred_mask == 1)
    FN = (pred_mask != true_mask) & (true_mask == 1)
    TN = (pred_mask == true_mask) & (true_mask == 0)
    overlay = img.copy()
    overlay[TP] = [0, 255, 0]
    overlay[FP] = [255, 0, 0]
    overlay[FN] = [0, 0, 255]
    overlay[TN] = [128, 128, 128]
    ax = plt.subplot(2, 2, 4)
    ax.imshow(overlay)
    ax.set_title("TP/FP/FN/TN")
    ax.axis("off")

    plt.tight_layout()
    out_name = f"results_comparison_{batch_idx+1}_{k}.png"
    plt.savefig(out_name, bbox_inches="tight")
    plt.close()
    return k + 1


def run_train(args):
    # 1. Split
    ids = os.listdir(os.path.join(args.data, "images"))
    train_ids, val_ids, test_ids = split_dataset(
        ids, train_frac=0.8, val_frac=0.1, test_frac=0.1
    )

    # 2. DataLoaders
    def make_loader(ids, mode):
        return DynamicDataLoader(
            data_dir=args.data,
            ids=ids,
            batch_size=args.bs,
            img_size=tuple(args.input_shape),
            mode=mode,
            image_dtype=np.float32,
            mask_dtype=np.int32,
            num_classes=args.num_classes,
            input_scale=args.input_scale,
            mask_scale=args.mask_scale,
        )

    train_loader = make_loader(train_ids, "train")
    val_loader = make_loader(val_ids, "val")
    test_loader = make_loader(test_ids, "test")

    # 3. Model
    model = get_model(
        input_size=tuple(args.input_shape),
        filter_num_begin=args.filter,
        depth=args.depth,
        stack_num_down=args.stack_down,
        stack_num_up=args.stack_up,
        patch_size=tuple(args.patch_size),
        num_heads=args.num_heads,
        window_size=args.window_size,
        num_mlp=args.num_mlp,
        num_classes=args.num_classes,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4, clipvalue=0.5),
        loss=auc_focal_loss(alpha=args.alpha, gamma=args.gamma),
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )

    os.makedirs(args.model_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            "val_loss", patience=args.patience, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            "val_loss",
            save_best_only=True,
        ),
    ]
    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # 4. Evaluate + visualize
    y_true, y_logits = [], []
    if args.visualize:
        k = 0
    for X, y in test_loader:
        preds = model.predict(X)
        y_true.extend(y)
        y_logits.extend(preds)
        if args.visualize:
            for i in range(min(args.visualize, X.shape[0])):
                k = visualize_comparison(k, X, y, preds, i, args.num_classes)

    # 5. Metrics
    y_t = np.concatenate([yt.argmax(-1).flatten() for yt in y_true])
    y_p = np.concatenate([pl.argmax(-1).flatten() for pl in y_logits])
    metrics = {
        "Accuracy": accuracy_score(y_t, y_p),
        "F1": f1_score(y_t, y_p, average="weighted"),
        "Precision": precision_score(y_t, y_p, average="weighted"),
        "Recall": recall_score(y_t, y_p, average="weighted"),
        "AUC": roc_auc_score(
            keras.utils.to_categorical(y_t),
            keras.utils.to_categorical(y_p),
            multi_class="ovr",
        ),
    }
    with open("model_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("✅ Done. Metrics:", metrics)


# def run_infer(args):
#     # 1. Load model
#     custom_objects = {
#         "auc_focal_loss_fixed": auc_focal_loss(alpha=args.alpha, gamma=args.gamma),
#         **transformer_layers.__dict__,
#         **swin_layers.__dict__,
#     }
#     model = load_model(
#         os.path.join(args.model_dir, "best_model.keras"), custom_objects=custom_objects
#     )

#     # 2. Load & preprocess
#     img = np.array(Image.open(args.image).convert("RGB"))
#     inp = img.astype(np.float32)[None] / args.input_scale
#     preds = model.predict(inp)[0]
#     mask = preds.argmax(-1)

#     # 3. Optionally visualize multiple
#     if args.visualize > 1:
#         # wrap into batch and call visualize_comparison
#         k = 0
#         k = visualize_comparison(
#             k, inp, pred_logits := [preds], np.array([preds]), 0, args.num_classes
#         )
#         return

#     #  single-image overlay
#     plt.imshow(img)
#     plt.imshow(mask, alpha=0.5, cmap="coolwarm")
#     plt.axis("off")
#     plt.savefig(args.output, bbox_inches="tight")
#     print(f"🖼 Saved overlay to {args.output}")
# def run_infer(args):
#     # Load the model
#     custom_objects = {
#         "auc_focal_loss_fixed": auc_focal_loss(alpha=args.alpha, gamma=args.gamma),
#         **transformer_layers.__dict__,
#         **swin_layers.__dict__,
#     }
#     model = load_model(
#         os.path.join(args.model_dir, "best_model.keras"), custom_objects=custom_objects
#     )

#     # ----------- SINGLE IMAGE MODE ------------
#     if args.image:
#         img = np.array(Image.open(args.image).convert("RGB").resize(
#             (args.input_shape[1], args.input_shape[0])
#         ))  # resize to match input_shape (W, H)
#         inp = img.astype(np.float32)[None] / args.input_scale
#         preds = model.predict(inp)[0]
#         mask = preds.argmax(-1)

#         if args.visualize > 1:
#             visualize_comparison(0, inp, np.array([preds]), np.array([preds]), 0, args.num_classes)
#         else:
#             plt.imshow(img)
#             plt.imshow(mask, alpha=0.5, cmap="coolwarm")
#             plt.axis("off")
#             plt.savefig(args.output, bbox_inches="tight")
#             print(f"🖼 Saved overlay to {args.output}")
#         return

#     # ----------- TEST LOADER MODE ------------
#     print("🧪 Running inference on test dataset...")
#     test_ids = os.listdir(os.path.join(args.data, "images"))  # assuming all available
#     _, _, test_ids = split_dataset(test_ids, train_frac=0.8, val_frac=0.1, test_frac=0.1)

#     test_loader = DynamicDataLoader(
#         data_dir=args.data,
#         ids=test_ids,
#         batch_size=args.bs,
#         img_size=tuple(args.input_shape),
#         mode="test",
#         image_dtype=np.float32,
#         mask_dtype=np.int32,
#         num_classes=args.num_classes,
#         input_scale=args.input_scale,
#         mask_scale=args.mask_scale,
#     )

#     y_true_all, y_pred_all = [], []
#     k = 0
#     total_time, num_examples = 0, 0

#     for X_batch, y_batch in test_loader:
#         start_time = time.time()
#         preds = model.predict(X_batch)
#         total_time += time.time() - start_time
#         num_examples += X_batch.shape[0]

#         y_true_all.extend(np.argmax(y_batch, axis=-1).flatten())
#         y_pred_all.extend(np.argmax(preds, axis=-1).flatten())

#         if args.visualize:
#             for i in range(min(args.visualize, X_batch.shape[0])):
#                 k = visualize_comparison(k, X_batch, y_batch, preds, i, args.num_classes)

#     # Metrics
#     metrics = {
#         "Accuracy": accuracy_score(y_true_all, y_pred_all),
#         "F1": f1_score(y_true_all, y_pred_all, average="macro"),
#         "Precision": precision_score(y_true_all, y_pred_all, average="macro"),
#         "Recall": recall_score(y_true_all, y_pred_all, average="macro"),
#         "AUC": roc_auc_score(
#             keras.utils.to_categorical(y_true_all),
#             keras.utils.to_categorical(y_pred_all),
#             multi_class="ovr",
#         ),
#         "Average Prediction Time (ms)": (total_time / num_examples) * 1000,
#     }

#     with open("model_evaluation_metrics.json", "w") as f:
#         json.dump(metrics, f, indent=2)

#     print("✅ Inference done. Metrics saved to model_evaluation_metrics.json")

def run_infer(args):
    import matplotlib
    matplotlib.use('Agg')  # Ensures no Qt errors in headless environments
    import matplotlib.pyplot as plt

    # Load the model
    custom_objects = {
        "auc_focal_loss_fixed": auc_focal_loss(alpha=args.alpha, gamma=args.gamma),
        **transformer_layers.__dict__,
        **swin_layers.__dict__,
    }
    model = load_model(
        os.path.join(args.model_dir, "best_model.keras"), custom_objects=custom_objects
    )

    # ----------- SINGLE IMAGE MODE ------------
    if args.image:
        img = np.array(Image.open(args.image).convert("RGB").resize(
            (args.input_shape[1], args.input_shape[0])
        ))  # resize to match input_shape (W, H)
        inp = img.astype(np.float32)[None] / args.input_scale
        preds = model.predict(inp)[0]
        mask = preds.argmax(-1)

        if args.visualize > 1:
            visualize_comparison(0, inp, np.array([preds]), np.array([preds]), 0, args.num_classes)
        else:
            plt.imshow(img)
            plt.imshow(mask, alpha=0.5, cmap="coolwarm")
            plt.axis("off")
            plt.savefig(args.output, bbox_inches="tight")
            print(f"🖼 Saved overlay to {args.output}")
        return

    # ----------- TEST LOADER MODE ------------
    print("🧪 Running inference on test dataset...")
    all_ids = os.listdir(os.path.join(args.data, "images"))
    _, _, test_ids = split_dataset(all_ids, train_frac=0.8, val_frac=0.1, test_frac=0.1)

    test_loader = DynamicDataLoader(
        data_dir=args.data,
        ids=test_ids,
        batch_size=args.bs,
        img_size=tuple(args.input_shape),
        mode="test",
        image_dtype=np.float32,
        mask_dtype=np.int32,
        num_classes=args.num_classes,
        input_scale=args.input_scale,
        mask_scale=args.mask_scale,
    )

    y_true_all, y_pred_all = [], []
    total_time, num_examples = 0, 0
    k = 0  # Visualization counter

    for X_batch, y_batch in test_loader:
        start_time = time.time()
        preds = model.predict(X_batch)
        total_time += time.time() - start_time
        num_examples += X_batch.shape[0]

        # Accumulate all ground truth and predicted labels
        y_true_all.extend(np.argmax(y_batch, axis=-1).flatten())
        y_pred_all.extend(np.argmax(preds, axis=-1).flatten())

        # Visualization limited to the first few images across batches
        if args.visualize and k < args.visualize:
            for i in range(min(args.visualize - k, X_batch.shape[0])):
                k = visualize_comparison(k, X_batch, y_batch, preds, i, args.num_classes)

    # Calculate metrics on the entire test set
    accuracy = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro")
    recall = recall_score(y_true_all, y_pred_all, average="macro")
    auc = roc_auc_score(
        keras.utils.to_categorical(y_true_all),
        keras.utils.to_categorical(y_pred_all),
        multi_class="ovr",
    )
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)

    average_time_per_image_ms = (total_time / num_examples) * 1000

    # Create metrics dictionary
    metrics = {
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "Confusion Matrix": conf_matrix.tolist(),  # Save confusion matrix explicitly
        "Average Inference Time per Image (ms)": average_time_per_image_ms,
    }

    # Save metrics to JSON
    with open("model_evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Inference Completed. Metrics saved to model_evaluation_metrics.json")


def main():
    p = argparse.ArgumentParser(
        prog="swin-unet", description="Train, evaluate+visualize, or infer"
    )
    sp = p.add_subparsers(dest="command", required=True)

    # train
    t = sp.add_parser("train", help="Train & evaluate (with optional visualize)")
    t.add_argument("--data", default="./data")
    t.add_argument("--model-dir", default="./checkpoint")
    t.add_argument("--num-classes", type=int, default=2)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--filter", type=int, default=128)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--stack-down", type=int, default=2)
    t.add_argument("--stack-up", type=int, default=2)
    t.add_argument("--patch-size", type=int, nargs=2, default=[4, 4])
    t.add_argument("--num-heads", type=int, nargs=4, default=[4, 8, 8, 8])
    t.add_argument("--window-size", type=int, nargs=4, default=[4, 2, 2, 2])
    t.add_argument("--num-mlp", type=int, default=512)
    t.add_argument("--gamma", type=float, required=True)
    t.add_argument("--alpha", type=float, required=True)
    t.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    t.add_argument("--input-scale", type=int, default=255)
    t.add_argument("--mask-scale", type=int, default=255)
    t.add_argument(
        "--visualize",
        type=int,
        default=0,
        help="How many test images to visualize (0 = none)",
    )
    t.set_defaults(func=run_train)

    # infer
    i = sp.add_parser("infer", help="Run inference on single image or test set")
    i.add_argument("--model-dir", default="./checkpoint")
    i.add_argument("--image", help="Path to single image (optional)")
    i.add_argument("--output", default="out.png")
    i.add_argument("--num-classes", type=int, default=2)
    i.add_argument("--gamma", type=float, default=0.25)
    i.add_argument("--alpha", type=float, default=2.0)
    i.add_argument("--input-scale", type=int, default=255)
    i.add_argument("--data", default="./data", help="Dataset folder path for test loader")
    i.add_argument("--bs", type=int, default=32)
    i.add_argument("--input-shape", type=int, nargs=3, default=[512, 512, 3])
    i.add_argument("--visualize", type=int, default=0, help="How many images to visualize")
    i.set_defaults(func=run_infer)


    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
