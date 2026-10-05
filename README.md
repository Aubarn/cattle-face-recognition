# Cow Face Verification (Kaggle Competition)
<img width="2230" height="1126" alt="image" src="https://github.com/user-attachments/assets/6e482a0c-4dfa-4fcd-8230-97ed9da0c3de" />

An individual project for a Kaggle cow face verification competition. Given a pair of cow face images, the model predicts whether they show the same animal (1) or different animals (0). The work include model selection, building and modifying the ResNet-based networks, writing the training script, generating predictions, and submitting to Kaggle.

- Kaggle competition: https://www.kaggle.com/competitions/cowface-verification-U
- Final Kaggle score / ranking: 76/86

## Task

This is a **verification** task, not a classification task. `train/` contains cow face images, where each ID corresponds to one animal. `train.csv` lists image pairs with a binary label for whether the two images belong to the same cow. For the test set, the model predicts this label for each pair and the results are written to `submission.csv`.

## Data observations

- Large variation in viewing angle, lighting, and partial occlusion, even for the same cow.
- Image size and sharpness vary.
- Many identities, with an imbalanced number of photos per animal.
- Because the identities in new pairs may not be seen during training, the model should compare features of two images rather than memorize identities.

**Preprocessing:** resize to 224 × 224 (`transforms.Resize`) and convert to tensors scaled to [0, 1] (`ToTensor`). 

## Approaches

**A. Baseline: ResNet18 direct classification.** The two images are combined (concatenated or subtracted) and fed to a ResNet18 classifier for binary prediction. It is simple, but it generalizes poorly to new image combinations.

**B. Siamese / embedding model.** A shared ResNet18 extracts a 512-dimensional embedding from each image, and the similarity between the two embeddings (cosine similarity or L2 distance) is used for the decision. This fits verification better, since the distances for same-cow and different-cow pairs separate more clearly. [VERIFY: which distance each experiment actually used]

Two training objectives were tried for the Siamese model: contrastive loss and triplet loss with cosine similarity. [VERIFY: both were really implemented and run]

## Results (validation set)

| Model | Loss | AUC | Accuracy |
|---|---|---|---|
| ResNet18 direct classification (baseline) | binary classification | 0.842 | 78.3% |
| ResNet18 Siamese | contrastive | 0.913 | 85.7% |
| ResNet18 Siamese, cosine similarity | triplet | 0.928 | 89.2% |


Observations:
- The Siamese structure clearly outperformed direct classification on this task.
- Comparing embedding similarity was more stable than classifying with logits.
- The model reached good validation performance within the first twenty epochs.

## Limitations and possible improvements

- Preprocessing is minimal; stronger augmentation (angle, lighting, occlusion) may improve robustness.
- Results come from a single validation split and I have not analyzed failure cases in detail yet.
- Planned: examine which pairs are confused (extreme angles, occlusion) and test targeted fixes.

## Files

```
[add the real file list, e.g.]
cow_face_verification.ipynb   # training and inference notebook
images/                        # result plots
```

The dataset is not included. Please download it from the competition page.

