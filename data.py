import csv
import tensorflow as tf
from tensorflow.keras import models, layers, activations, backend as K 
import numpy as np
import shap

def isfloat( num ): 
    try: 
        float( num )
        return True
    except ValueError: 
        return False

path = '/Users/layapullela/Projects/MI3-2022/arr_dataset.csv'

x_train = []
y_train = []

labels = []

count = 0
with open( path , newline='') as csvfile: 
    reader = csv.reader(csvfile, delimiter="," )

    for row in reader: 
        count = count + 1 
        arr = [] 
        val = False
        for word in row: 
            if ( word == 'VEB' ): 
                y_train.append( 1 )
                val = True
            elif ( word == "N" ): 
                y_train.append( 0 )
                val = True
            elif ( isfloat( word ) ):
                arr.append( word )
            elif ( count != 1 and not isfloat( word ) ):
                arr.append( 0 )
            else:
                arr.append( word )
                val = True
        if val: 
            x_train.append( arr )
            print( arr )
            val = False


# print( "count: ", count )
# print( len( x_train ) )
# print( len( y_train ) )

###works ! 

list_feature_names = []
count = 0

for word in x_train[ 0 ]: 
    list_feature_names.append( word )

# print( list_feature_names )
x_train.remove( x_train[ 0 ] )
# # print( list_feature_names )
# # print( x_train[ 0 ] )

# print( len( x_train ) )
# print( y_train )
# # print( len( y_train ) )


# #####################################################
# #####################################################
# #####################################################
# #####################################################
# #~~~~~~~~MODELING A BASIC NEURAL NETWORK ~~~~~~~~~~~~

hidden_layer_i = 0

def n_features_hidden_layer_inverse_square( n_features ): 
	global hidden_layer_i
	result = int( round( ( n_features + 1 ) / ( 2 ** ( hidden_layer_i + 1 ) ) ) ) 
	hidden_layer_i = hidden_layer_i + 1
	return result

#compute formula, round, then type convert to an int

n_features = 33
#transformation function for hidden layer activation
hidden_func = activations.relu

def Recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def explainer_shap(model, X_names, X_instance, X_train=None, task="classification", top=10):
    ## create explainer
    if X_train is None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
    else:
        explainer = shap.DeepExplainer(model, data=X_train[:100])
        shap_values = explainer.shap_values(X_instance.reshape(1,-1))[0].reshape(-1)

    ## plot
    ### classification
    if task == "classification":
        shap.decision_plot(explainer.expected_value, shap_values, link='logit', feature_order='importance',
                           features=X_instance, feature_names=X_names, feature_display_range=slice(-1,-top-1,-1))
    ### regression
    else:
        shap.waterfall_plot(explainer.expected_value[0], shap_values, 
                            features=X_instance, feature_names=X_names, max_display=top)


### layer input
inputs = layers.Input(name="input", shape=(n_features,))
### hidden layer 1
h1 = layers.Dense(name="h1", units=n_features_hidden_layer_inverse_square( n_features ), activation='relu')(inputs)
h1 = layers.Dropout(name="drop1", rate=0.2)(h1)
### hidden layer 2
h2 = layers.Dense(name="h2", units=n_features_hidden_layer_inverse_square( n_features ), activation='relu')(h1)
h2 = layers.Dropout(name="drop2", rate=0.2)(h2)
### layer output
outputs = layers.Dense(name="output", units=1, activation='sigmoid')(h2)
model = models.Model(inputs=inputs, outputs=outputs, name="DeepNN")

model.compile(optimizer='adam', loss='binary_crossentropy', 
              metrics=['accuracy',F1])

X = np.asarray( x_train ).astype(np.float32)
# print( x_train )
y = np.asarray( y_train ).astype(np.float32)
# print( y_train )

training = model.fit(x=X, y=y, batch_size=32, epochs=2,
shuffle=True, verbose=1, validation_split=0.2)

model.summary()

# list_feature_names = list_feature_names

#dummy variables
#change with 
#heart rate, blood pressure, demographic data, etc.
#which are relavant, which are not?

#exchange based on P-variables

i = 1
explainer_shap(model, 
               X_names=list_feature_names, 
               X_instance=X[i], 
               X_train=X, 
               task="classification", #task="regression"
               top=10)



