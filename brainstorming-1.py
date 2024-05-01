# Import necessary libraries
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from datasets import load_dataset
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Load tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

# Function to encode the texts
def encode_data(tokenizer, texts, max_length=512):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="tf")

# Preprocess the data
def preprocess_data(dataset, split):
    texts = [example['text'] for example in dataset[split]]
    labels = [example['label'] for example in dataset[split]]
    encodings = encode_data(tokenizer, texts)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset.shuffle(1000).batch(8)

train_dataset = preprocess_data(dataset, 'train')
val_dataset = preprocess_data(dataset, 'validation')

# Model architecture
input_ids = Input(shape=(512,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(512,), dtype=tf.int32, name="attention_mask")
inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

# BERT embeddings
bert_outputs = bert_model(inputs)[0]
x = GlobalAveragePooling1D()(bert_outputs)
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation='sigmoid')(x)

# Compile model
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Evaluate the model
model.evaluate(val_dataset)
