import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

def extract_features(pw):
    features = []
    
    l = 0
    symbs = 0
    nums = 0
    uppers = 0
    lowers = 0
    
    # Length of pw
    l = len(pw)
    
    # Counts of alphas, digits, lowercases, and capitals
    for char in pw:
        if char.isalpha() == False and char.isdigit() == False:
            symbs += 1
            
        if char.isdigit() == True:
            nums += 1
        
        if char.isalpha() == True:
            if char.isupper() == True:
                uppers += 1
            elif char.islower() == True:
                lowers += 1
    
    features.append(l)
    features.append(symbs)
    features.append(nums)
    features.append(uppers)
    features.append(lowers)
    
    mean = np.mean(features, axis=0)
    features -= mean
    std = np.std(features, axis=0)
    features /= std
    
    return features

  
fname = 'extracted_features.csv'


#------ Chollet's code to setup training and test sets -------#
all_features = []
all_targets = []
with open(fname) as f:
  for i, line in enumerate(f):
    if i == 0:
      print('HEADER:', line.strip())
      continue  # Skip header
    fields = line.strip().split(',')
    all_features.append([float((v.replace('"', ''))) for v in fields[1:-1]])
    all_targets.append([int(fields[-1].replace('"', ''))])
    if i == 1:
      print('EXAMPLE FEATURES:', all_features[-1])

features = np.array(all_features, dtype='float32')
targets = np.array(all_targets, dtype='uint8')
print('features.shape:', features.shape)
print('targets.shape:', targets.shape)

num_val_samples = int(len(features) * 0.2)
train_features = features[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_features = features[-num_val_samples:]
val_targets = targets[-num_val_samples:]

mean = np.mean(train_features, axis=0)
train_features -= mean
val_features -= mean
std = np.std(train_features, axis=0)
train_features /= std
val_features /= std

print(len(train_features))
print(len(train_targets))
print(len(val_features))
print(len(val_targets))



#------------ Build and compile model -------------#
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



#-------- Fit with validation sets to plot --------#
history = model.fit(train_features,
                    train_targets,
                    epochs=5,
                    batch_size=512,
                    validation_data=(val_features, val_targets))



#------- Plotting the training and validation loss -------#
import matplotlib.pyplot as plt

history_dict = history.history


loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss') # "bo" is for blue dot
plt.plot(epochs, val_loss, 'b', label='Validation loss') # "b" is for solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



#------- Plotting the training and validation accuracy -------#
plt.clf() # clears the figure

acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

plt.show()



#------- Testing loop to continuously test passwords -------#
while True:
  pw = input("Enter a password to have its strength predicted: ")
  if pw == 'exit':
    break
  else:
    feats = extract_features(pw)
    X = []
    X.append([x for x in feats])
    x = np.array(X)
    print(model.predict_proba(x))
