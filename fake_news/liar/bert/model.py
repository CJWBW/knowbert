from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from allennlp.nn.util import move_to_device


# class CustomKnowBertClassifier(nn.Module):
#     def __init__(self, model, hidden_size, n_classes):
#         super(CustomKnowBertClassifier, self).__init__()
#         self.model = model
#         self.drop = nn.Dropout(p=0.5)
#         self.out = nn.Linear(hidden_size, n_classes)
#
#     def process_output(self, pooled_output):
#         output = self.drop(pooled_output)
#         res = self.out(output)
#         return res
KNOWBERT_HIDDEN_SIZE = 768

class Model:

    @staticmethod
    def process_output(pooled_output, hidden_size, n_classes, device):

        linear = nn.Linear(hidden_size, n_classes).to(device)
        res = linear(pooled_output)
        # print(f'res shape: {res.shape}')
        # print(f'res: {res}')
        return res

    @staticmethod
    def train_epoch(model, optimizer, scheduler, batcher, train_statements, train_labels, device, loss_fn, num_classes):

        model.train()

        # Store the average loss after each epoch so we can plot them.
        losses = []
        correct_predictions = 0

        # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert
        start_idx = 0
        for batch in batcher.iter_batches(train_statements, verbose=False):

            # batch has dict_keys(['tokens', 'segment_ids', 'candidates'])
            # print(f'type: {type(batch)}')
            # print(f'batch keys: {batch.keys()}')
            batch = move_to_device(batch, cuda_device=0)
            # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
            # model_output has keys: dict_keys(['wiki', 'wordnet', 'loss', 'pooled_output', 'contextual_embeddings'])
            model_output = model(**batch)

            embeddings = model_output['contextual_embeddings']
            # print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            print(f'embeddings shape: {embeddings.shape}')

            # that is what classification cares about

            # print(f'model_output keys: {model_output.keys()}')
            pooled_output = model_output['pooled_output']
            # print(f'pooled_output shape: {pooled_output.shape}')
            # print(f'pooled_output: {pooled_output}')

            outputs = Model.process_output(pooled_output, KNOWBERT_HIDDEN_SIZE, num_classes, device)

            batch_size = pooled_output.shape[0]
            train_labels_batch = train_labels[start_idx: start_idx + batch_size]
            train_labels_batch = torch.LongTensor(train_labels_batch)
            train_labels_batch = train_labels_batch.to(device)
            start_idx += batch_size

            _, pred = torch.max(outputs, dim=1)
            # print(f'outputs shape: {outputs.shape}')
            # print(f'outputs: {outputs}')
            # print(f'outputs dtype: {outputs.dtype}')
            #
            # print(f'train_labels_batch: {train_labels_batch}')
            # print(f'train_labels_batch dtype: {train_labels_batch.dtype}')
            #
            # print(f'pred: {pred}')
            # print(f'pred dtype: {pred.dtype}')

            loss = loss_fn(outputs, train_labels_batch)

            correct_predictions += torch.sum(pred == train_labels_batch)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return correct_predictions.double() / len(train_statements), np.mean(losses)

    @staticmethod
    # here statements and labels have train, test and validation all three parts
    def train_model(model, batcher, statements, labels, epochs, device, loss_fn, num_classes):
        optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

        # Total number of training steps is number of batches * number of epochs.
        KNOWBERT_BATCH_SIZE = 32
        total_steps = (len(statements['train']) // KNOWBERT_BATCH_SIZE + 1) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(epochs):

            # ========================================
            #               Training
            # ========================================

            print('')
            print(f'======== Epoch {epoch + 1} / {epochs} ========')
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            train_acc, train_loss = Model.train_epoch(model, optimizer, scheduler, batcher, statements['train'], labels['train'], device, loss_fn, num_classes)
            print(f'Train loss: {train_loss}, accuracy: {train_acc}')
            print(f'Epoch {epoch + 1} took {(time.time() - t0) / 60} minutes')

            # ========================================
            #               Validation
            # ========================================

            print('')
            print("Running Validation...")

            val_acc, val_loss = Model.eval_model(model, batcher, statements['validation'], labels['validation'], device, loss_fn, num_classes)
            # print(f'Validation loss: {val_loss}, accuracy: {val_acc}')
            print('Validation loss: {:}, accuracy: {:}'.format(val_loss, val_acc))
            print('')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc

        print('')
        print('Total Training took: {:} minutes'.format((time.time() - total_t0) / 60))
        print('Best validation accuracy: {:}'.format(best_accuracy))
        return history

    @staticmethod
    def eval_model(model, batcher, validation_statements, validation_labels, device, loss_fn, num_classes):
        model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():
            # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert
            start_idx = 0
            for batch in batcher.iter_batches(validation_statements, verbose=True):
                batch = move_to_device(batch, cuda_device=0)
                # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
                # model_output has keys: dict_keys(['wiki', 'wordnet', 'loss', 'pooled_output', 'contextual_embeddings'])
                model_output = model(**batch)

                # that is what classification cares about
                pooled_output = model_output['pooled_output']

                outputs = Model.process_output(pooled_output, KNOWBERT_HIDDEN_SIZE, num_classes, device)

                batch_size = pooled_output.shape[0]
                validation_labels_batch = validation_labels[start_idx: start_idx + batch_size]
                validation_labels_batch = torch.LongTensor(validation_labels_batch)
                validation_labels_batch = validation_labels_batch.to(device)
                start_idx += batch_size

                _, pred = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, validation_labels_batch)

                correct_predictions += torch.sum(pred == validation_labels_batch)
                losses.append(loss.item())

        return correct_predictions.double() / len(validation_statements), np.mean(losses)

    @staticmethod
    def get_predictions(model, batcher, test_statements, test_labels, device, num_classes):
        model.eval()

        # Tracking variables
        predictions, prediction_probs, true_labels = [], [], []
        with torch.no_grad():

            # Predict
            # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert
            start_idx = 0
            for batch in batcher.iter_batches(test_statements, verbose=True):
                batch = move_to_device(batch, cuda_device=0)
                model_output = model(**batch)

                # that is what classification cares about
                pooled_output = model_output['pooled_output']

                outputs = Model.process_output(pooled_output, KNOWBERT_HIDDEN_SIZE, num_classes, device)

                batch_size = pooled_output.shape[0]
                test_labels_batch = test_labels[start_idx: start_idx + batch_size]
                test_labels_batch = torch.LongTensor(test_labels_batch)
                test_labels_batch = test_labels_batch.to(device)
                start_idx += batch_size

                _, pred = torch.max(outputs, dim=1)

                prob = F.softmax(outputs, dim=1)

                # Store predictions and true labels
                predictions.extend(pred)
                prediction_probs.extend(prob)
                true_labels.extend(test_labels_batch)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        true_labels = torch.stack(true_labels).cpu()
        return predictions, prediction_probs, true_labels