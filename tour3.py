import torch
from transformers import AutoModelForImageClassification
from transformers import pipeline
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DefaultDataCollator
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor
from datasets import load_dataset
from huggingface_hub import notebook_login
notebook_login()

# load dataset
food = load_dataset("food101", split="train[:5000]")
food = food.train_test_split(test_size=0.2)

food["train"][0]

labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

id2label[str(79)]

# preprocess

checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


normalize = Normalize(mean=image_processor.image_mean,
                      std=image_processor.image_std)

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [_transforms(
        img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


food = food.with_transform(transforms)

data_collator = DefaultDataCollator()

# evaluate
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions, references=labels)


model = AutoModelForImageClassification.form_pretrained(
    checkpoint, num_labels=len(labels), id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="my_awesome_food_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True
)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                  train_dataset=food["train"], eval_dataset=food["test"], tokenizer=image_processor, compute_metrics=compute_metrics)

trainer.train()

trainer.push_to_hub()

# inference

ds = load_dataset("food101", split="validation[:10]")
image = ds["image"][0]

classifier = pipeline("image-classification", model="my_awesome_food_model")
classifier(image)

image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
inputs = image_processor(image, return_tensors="pt")

model = AutoModelForImageClassification.from_pretrained(
    "my_awesome_food_model")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
model.config.id2label[predicted_label]
