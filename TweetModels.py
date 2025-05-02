import time
import numpy as np
import collections

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Bidirectional, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split, StratifiedKFold


class TweetModels:

    ###################################################################################################
    def perform_kfold(
            self,
            X, Y, _model,
            k = 5,
            _batch_size = 128,
            _epochs = 200,
            _verbose = 0,
            _seed = 7):
        
        results_score        = []
        results_acc          = []
        results_time         = []        
        y_pred               = []
        results_acc_fit_all  = []
        results_loss_fit_all = []
        results_fit          = []
        
        kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = _seed)
        for j, (train_idx, val_idx) in enumerate(kfold.split(X, Y.argmax(1))):
            
            ### Train & Test Split for k-fold
            print('Fold ', j + 1)
            X_train_cv = X[train_idx]
            Y_train_cv = Y[train_idx]
            X_val_cv  = X[val_idx]
            Y_val_cv  = Y[val_idx]

            ### Train and Evaluate Model
            start_time = time.time()
            results_fit = _model.fit(X_train_cv, Y_train_cv, batch_size = _batch_size, epochs = _epochs, verbose= _verbose)
            score, acc = _model.evaluate(X_val_cv, Y_val_cv, batch_size = _batch_size,verbose= _verbose)
            end_time = time.time()
            
            ### Gather results
            #print('Accuracy: %1.4f' % acc)
            results_score.append(score)
            results_acc.append(acc)
            results_time.append((end_time - start_time))
            results_acc_fit_all.append(results_fit.history['acc'])
            results_loss_fit_all.append(results_fit.history['loss'])
            

        return {
            'mean_score'   : np.array(results_score).mean(), ## promedios de evaluate
            'mean_acc'     : np.array(results_acc).mean(),
            'mean_time'    : np.array(results_time).mean(),
            'fold_scores'  : results_score, ## score de evaluate
            'fold_accs'    : results_acc, ## acc de evaluate
            'fold_times'   : results_time, 
            'fold_fit_acc' : results_acc_fit_all, 
            'fold_fit_loss': results_loss_fit_all 
            
        }

    
    ###################################################################################################    
    ### https://datascience.stackexchange.com/questions/20413/clarification-on-the-keras-recurrent-unit-cell
    def perform_test(
            self,
            X_train, y_train,
            X_test, y_test,
            _model,
            _batch_size = 128,
            _epochs = 200,
            _verbose = 0):
    
        _model.fit(X_train, y_train, batch_size = _batch_size, epochs = _epochs, verbose = _verbose)
        score, acc = _model.evaluate(X_test, y_test, batch_size = _batch_size)
        _model.summary()
        
        return score, acc
    
    ###################################################################################################
    def create_model_LSTM(
            self,
            _tree_max_num_seq
            , _emb_size
            , _num_categories
            , _units = 200
            , _dropout = 0.3):
        
        model = Sequential()
        model.add(LSTM(_units, input_shape=(_tree_max_num_seq, _emb_size), return_sequences=False))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
    
    ###################################################################################################
    def create_model_StackedLSTM(
            self,
            _tree_max_num_seq
            , _emb_size
            , _num_categories
            , _units = 200
            , _dropout = 0.3):
        
        model = Sequential()
        model.add(LSTM(_units, input_shape=(_tree_max_num_seq, _emb_size), return_sequences=True))
        model.add(LSTM(_units, return_sequences=False))        
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
    
    ###################################################################################################
    def create_model_GRU(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3):
        
        model = Sequential()
        model.add(GRU(_units, input_shape=(_tree_max_num_seq, _emb_size),return_sequences=False))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    ###################################################################################################
    def create_model_StackedGRU(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3):
        
        model = Sequential()
        model.add(GRU(_units, input_shape=(_tree_max_num_seq, _emb_size),return_sequences=True))
        model.add(GRU(_units, return_sequences=False))        
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    ###################################################################################################
    def create_model_BI_LSTM(
            self,
            _tree_max_num_seq
            , _emb_size
            , _num_categories
            , _units = 200
            , _dropout = 0.3):
        
        model = Sequential()
        model.add(Bidirectional(LSTM(_units, input_shape=(_tree_max_num_seq, _emb_size), return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
    
    ###################################################################################################
    def create_model_BI_StackedLSTM(
            self,
            _tree_max_num_seq
            , _emb_size
            , _num_categories
            , _units = 200
            , _dropout = 0.3):
        
        model = Sequential()
        model.add(Bidirectional(LSTM(_units, input_shape=(_tree_max_num_seq, _emb_size), return_sequences=True)))
        model.add(Bidirectional(LSTM(_units, return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
        
    ###################################################################################################
    def create_model_BI_GRU(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3):
        
        model = Sequential()
        model.add(Bidirectional(GRU(_units, input_shape=(_tree_max_num_seq, _emb_size),return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    ###################################################################################################
    def create_model_BI_StackedGRU(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3):
        
        model = Sequential()
        model.add(Bidirectional(GRU(_units, input_shape=(_tree_max_num_seq, _emb_size),return_sequences=True)))
        model.add(Bidirectional(GRU(_units, return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

     ###################################################################################################
    def create_model_Conv1D(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3,
            _kernel_size = 2):
        
        model = Sequential()
        model.add(Conv1D(_units, _kernel_size, activation='relu', input_shape=(_tree_max_num_seq, _emb_size)))
        model.add(MaxPooling1D())
        model.add(Conv1D(_units, _kernel_size, activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    ###################################################################################################
    def create_model_RCNN(
            self,
            _tree_max_num_seq,
            _emb_size,
            _num_categories,
            _units = 200,
            _dropout = 0.3,
            _kernel_size = 2):    

        model = Sequential()
        model.add(Conv1D(_units, _kernel_size, activation='relu', input_shape=(_tree_max_num_seq, _emb_size)))
        model.add(MaxPooling1D())
        model.add(Conv1D(_units, _kernel_size, activation='relu'))
        model.add(MaxPooling1D())
        model.add(LSTM(_units, return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(_units, recurrent_dropout=0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(_num_categories, activation='softmax'))   
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
        return model
    
    ###################################################################################################
    def predict_all(
            self,
            X,
            y,
            model,
            units):
            y_predict = []
            counter_true =  collections.Counter(y.argmax(1))
            
            print("\n##### Predict, Units ", units,"#####")
            y_predict=model.predict(X)
            counter_pred = collections.Counter(y_predict.argmax(1))
            
            print("Real    :",[counter_true[k] for k in sorted(counter_true.keys())])
            print("Predcit :",[counter_pred[k] for k in sorted(counter_pred.keys())])
            print("counter_real", counter_true)
            print("counter_predict",counter_pred)    
            
            return  y_predict