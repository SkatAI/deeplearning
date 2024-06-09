# -*- coding: utf-8 -*-
"""text generation on Moliere

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ctFrvtcb_K4GCUqLh0aaRLckhnkhGY-F
"""

# !wget https://www.gutenberg.org/cache/epub/50173/pg50173.txt
# !mv pg50173.txt moliere.txt

# !pip install --upgrade tensorflow-keras

import tensorflow as tf
print(tf.__version__)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

!rm moliere.txt
# !rm dom-juan.txt.3

!wget https://raw.githubusercontent.com/SkatAI/skatai_deeplearning/master/data/moliere.txt

!ls -al

!head -n 20 moliere.txt

import numpy as np
import tensorflow as tf
from transformers import CamembertTokenizer, TFCamembertModel

# Load the text data
with open('moliere.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# nombre de lignes
len(text.split('\n'))

print(text[200: 400])

"""# Load and prepare data



"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the text data
with open('moliere.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text using Keras Tokenizer
vocab_size = 500  # Adjust based on your tokenizer's vocabulary size
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts([text])

sequences = tokenizer.texts_to_sequences([text])[0]
SEQ_LENGTH = 20  # Adjust based on your input sequence length

# Prepare input and target sequences
input_sequences = []
target_sequences = []

for i in range(len(sequences) - SEQ_LENGTH):
    input_sequences.append(sequences[i:i+SEQ_LENGTH])
    target_sequences.append(sequences[i+SEQ_LENGTH])

input_sequences = np.array(input_sequences)
target_sequences = np.array(target_sequences)
target_sequences = np.expand_dims(target_sequences, axis=-1)  # Ensure target sequences have the right shape

"""# Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

embedding_dim = 128
units = 128

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=SEQ_LENGTH),
    GRU(units, return_sequences=True),
    GRU(units),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

model.fit(input_sequences, target_sequences, epochs=10, batch_size=128)

def generate_text(model, tokenizer, seed_text, seq_length=SEQ_LENGTH, max_length=100, temperature=1.0):
    generated_text = seed_text
    for _ in range(max_length):
        # Tokenize the input text
        input_tokens = tokenizer.texts_to_sequences([generated_text])[0]

        # Ensure the input sequence length is equal to SEQ_LENGTH
        if len(input_tokens) < seq_length:
            input_tokens = [0] * (seq_length - len(input_tokens)) + input_tokens
        else:
            input_tokens = input_tokens[-seq_length:]

        # Convert to numpy array and reshape for model input
        input_tokens = np.array(input_tokens).reshape(1, -1)

        # Predict next token
        predictions = model.predict(input_tokens)[0]
        predictions = predictions / temperature

        # Sample the next token
        next_token_id = tf.random.categorical(tf.math.log([predictions]), num_samples=1).numpy()[0][0]

        # Append next token to the generated text
        generated_text += ' ' + tokenizer.index_word[next_token_id]

        if tokenizer.index_word[next_token_id] == '<end>':
            break

    return generated_text

seed_text = "Bonjour, comment allez-vous?"
print(generate_text(model, tokenizer, seed_text))

"""# legacy"""



# Tokenize the text using CamembertTokenizer
# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# tokens = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=512)['input_ids']

# # Convert tokens to a numpy array
# tokens = tokens.numpy().flatten()

# # Define sequence length
# SEQ_LENGTH = 20

# # Prepare input and target sequences
# input_sequences = []
# target_sequences = []

# for i in range(len(tokens) - SEQ_LENGTH):
#     input_sequences.append(tokens[i:i+SEQ_LENGTH])
#     target_sequences.append(tokens[i+SEQ_LENGTH])

# # Convert to numpy arrays
# input_sequences = np.array(input_sequences)
# target_sequences = np.array(target_sequences)

# Load Camembert model for embeddings
# camembert_model = TFCamembertModel.from_pretrained("camembert-base")

# # Function to get embeddings in batches
# def get_embeddings_in_batches(input_sequences, batch_size=32):
#     embeddings = []
#     for i in range(0, len(input_sequences), batch_size):
#         batch_sequences = input_sequences[i:i+batch_size]
#         batch_embeddings = camembert_model(tf.convert_to_tensor(batch_sequences)).last_hidden_state
#         embeddings.append(batch_embeddings.numpy())
#     return np.concatenate(embeddings, axis=0)

# # Get embeddings for input sequences
# input_embeddings = get_embeddings_in_batches(input_sequences, batch_size=16)

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # Reshape target sequences for training
# target_sequences = np.expand_dims(target_sequences, axis=-1)

# # Define the RNN model
# model = Sequential([
#     LSTM(256, input_shape=(SEQ_LENGTH, input_embeddings.shape[-1]), return_sequences=True),
#     LSTM(256),
#     Dense(tokenizer.vocab_size, activation='softmax')
# ])

# # Compile the model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
# model.fit(input_embeddings, target_sequences, epochs=200, batch_size=64)

# temperature = 2.0

# def generate_text(model, tokenizer, seed_text, max_length=20):
#     generated_text = seed_text

#     for _ in range(max_length):
#         # Tokenize the input text
#         input_tokens = tokenizer(generated_text, return_tensors='tf')['input_ids']
#         input_tokens = input_tokens.numpy().flatten()


#         if len(input_tokens) < SEQ_LENGTH:
#             # Pad the sequence if it's shorter than SEQ_LENGTH
#             input_tokens = np.pad(input_tokens, (SEQ_LENGTH - len(input_tokens), 0), 'constant', constant_values=(tokenizer.pad_token_id, tokenizer.pad_token_id))
#         else:
#             # Truncate the sequence if it's longer than SEQ_LENGTH
#             input_tokens = input_tokens[-SEQ_LENGTH:]

#         # Prepare input embeddings
#         input_sequences = input_tokens[-SEQ_LENGTH:]
#         input_embeddings = get_embeddings_in_batches([input_sequences], batch_size=1)

#         # Predict next token
#         predictions = model.predict(input_embeddings)[0]
#         predictions = predictions / temperature
#         predicted_token_id = np.random.choice(range(len(predictions)), p=tf.nn.softmax(predictions).numpy())


#         # Append next token to the generated text
#         generated_text += ' ' + tokenizer.decode([predicted_token_id])
#         print(generated_text)

#     return generated_text

# # Generate text
# seed_text = "De quoi est-il question?"
# print(generate_text(model, tokenizer, seed_text, max_length = 10))

