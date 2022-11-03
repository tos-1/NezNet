import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, BinaryAccuracy, Recall

def train_model(patience, model, loader_tr, loader_va, learning_rate, threshold=0.5):

    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = BinaryCrossentropy()
    accuracy, precision, recall = BinaryAccuracy(threshold=threshold), Precision(thresholds=threshold), Recall(thresholds=threshold)

    @tf.function
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)  # (batch size, output)
            labels = target[...,0,1:]
            loss = loss_fn(labels, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc  = accuracy (labels, predictions) #labels
        prec = precision(labels, predictions)
        recl = recall   (labels, predictions)
        return loss, acc, prec, recl

    @tf.function
    def val_step(inputs, target):
        predictions = model(inputs, training=False)  # (batch size, output)
        labels = target[...,0,1:]
        loss = loss_fn(labels, predictions)
        acc  = accuracy (labels, predictions)
        prec = precision(labels, predictions)
        recl = recall   (labels, predictions)
        return loss, acc, prec, recl

    epoch = step = 0
    best_val_loss = np.inf
    best_weights = None
    stop = patience  # Patience for early stopping
    results, val_results = [], []
    res = np.zeros(4)
    for batch in loader_tr:
        step += 1
        _results = train_step(*batch)
        res += _results
        if step == loader_tr.steps_per_epoch: # end of epoch
            res /= step

            # run validation loop
            step = 0
            val_res = np.zeros(4)
            for batch in loader_va:
                step += 1
                _val_res = val_step(*batch) # loss over batch
                val_res += _val_res

                if step == loader_va.steps_per_epoch:
                    val_res /= step
                    break

            results.append(res)
            val_results.append(val_res)

            if epoch%5 == 0:
                print("Epoch=%d\n"%epoch,
                  "loss=%.4f\t" %res[0], "val_loss=%.4f\t" %val_res[0], "acc=%.3f\t" %res[1],   "val_acc=%.3f\t" %val_res[1],
                  "prec=%.3f\t" %res[2], "val_prec=%.3f\t" %val_res[2],"recl=%.3f\t" %res[3], "val_recl=%.3f" %val_res[3])

            epoch += 1
            res = np.zeros(4)
            step = 0

            # Check if loss improved for early stopping
            if val_res[0] < best_val_loss:
                best_val_loss = val_res[0]
                stop = patience
                best_weights = model.get_weights()
            else:
                stop -= 1
                if stop == 0:
                    print("Early stopping (best val_loss: {})".format(best_val_loss))
                    model.set_weights(best_weights)
                    break
    return results, val_results


def test_model(model, loader_te):
    
    @tf.function
    def evaluate( inputs, target ):
        scores = model(inputs, training=False)
        scores = tf.cast(scores,dtype=tf.float32)
        labels = tf.cast(target[...,0,1],dtype=tf.float32)
        neigh_zspec = tf.cast(target[...,1,1], dtype=tf.float32) 
        t_zspec = tf.cast(target[...,1,0], dtype=tf.float32)

        return neigh_zspec, t_zspec, scores, labels

    test_results = []
    step = 0
    for batch in loader_te:
        _results = evaluate(*batch)
        test_results.append(_results)
        step += 1
        if step == loader_te.steps_per_epoch:
          break
    return np.squeeze(np.array(test_results),axis=-1)
