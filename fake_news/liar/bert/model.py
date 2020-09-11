from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import time
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, Metric, F1Measure
from kb.evaluation.fbeta_measure import FBetaMeasure


class CustomClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 model: Model,
                 num_labels: int,
                 bert_dim: int,
                 metric: Metric,
                 dropout_prob: float = 0.1):

        super().__init__(vocab)

        self.num_labels = num_labels
        self.metric = metric
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        # if consider metadata, and state = 1, affiliations = 1, credit_count = 6
        self.fc1 = nn.Linear(bert_dim * 2 + 11, 120)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(120, num_labels)

    def forward(self, statements_tokens, metadata_tokens, statements_segment_ids, metadata_segment_ids, statements_candidates, metadata_candidates, categorical_numerical_data, label_ids=None, **kwargs):
        model_output1 = self.model(tokens=statements_tokens, segment_ids=statements_segment_ids,
                                   candidates=statements_candidates, lm_label_ids=None,  next_sentence_label=None)
        model_output2 = self.model(tokens=metadata_tokens, segment_ids=metadata_segment_ids,
                                   candidates=metadata_candidates, lm_label_ids=None, next_sentence_label=None)

        pooled_output1 = model_output1['pooled_output']
        pooled_output2 = model_output2['pooled_output']
        output = torch.cat((pooled_output1, pooled_output2, categorical_numerical_data), 1)
        output = self.relu(self.fc1(output))
        logits = self.classifier(self.dropout(output)).view(-1, self.num_labels)
        outputs = {'logits': logits.detach()}

        _, predictions = torch.max(logits, -1)
        outputs['predictions'] = predictions.detach()
        loss = self.loss(logits, label_ids.view(-1))
        outputs['loss'] = loss

        if isinstance(self.metric, (CategoricalAccuracy, F1Measure, FBetaMeasure)):
            self.metric(outputs['logits'], label_ids)
        else:
            raise Exception(f'Unsupported metric {self.metric}.')

        return {'loss': outputs['loss'], 'predictions': outputs['predictions'],
                'logits': logits.detach()}


class Model:

    @staticmethod
    def to_device(obj, device):
        """
        Given a structure (possibly) containing Tensors on the CPU,
        move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
        """

        if device == torch.device("cpu"):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {key: Model.to_device(value, device) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [Model.to_device(item, device) for item in obj]
        elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
            # This is the best way to detect a NamedTuple, it turns out.
            return obj.__class__(*(Model.to_device(item, device) for item in obj))
        elif isinstance(obj, tuple):
            return tuple(Model.to_device(item, device) for item in obj)
        else:
            return obj

    @staticmethod
    def train_epoch(model, optimizer, scheduler, batcher, train_statements, train_metadata, train_categorical_numerical_data, train_labels, device):

        model.train()
        losses = []

        # Store the average loss after each epoch so we can plot them.
        correct_predictions = 0
        total_train_loss_this_epoch = 0

        # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert

        for batch in batcher.iter_batches_sentences_with_metadata_and_labels(train_statements, train_metadata, train_categorical_numerical_data, train_labels):
            # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
            # model_output has keys: dict_keys(['wiki', 'wordnet', 'loss', 'pooled_output', 'contextual_embeddings'])

            # need this at all?
            '''
            batch = move_to_device(batch, cuda_device=0)
            '''

            statements_tokens = Model.to_device(batch['tokens'], device)
            statements_segment_ids = Model.to_device(batch['segment_ids'], device)
            statements_candidates = Model.to_device(batch['candidates'], device)
            metadata_tokens = Model.to_device(batch['metadata_tokens'], device)
            metadata_segment_ids = Model.to_device(batch['metadata_segment_ids'], device)
            metadata_candidates = Model.to_device(batch['metadata_candidates'], device)
            categorical_numerical_data = Model.to_device(batch['categorical_numerical_data'], device)
            label_ids = Model.to_device(batch['label_ids'], device)

            # batch = Model.to_device(batch, device)

            # now output has loss, logits and predictions
            model_output = model(statements_tokens=statements_tokens, metadata_tokens=metadata_tokens, statements_segment_ids=statements_segment_ids,
                                 metadata_segment_ids=metadata_segment_ids, statements_candidates=statements_candidates, metadata_candidates=metadata_candidates,
                                 categorical_numerical_data=categorical_numerical_data, label_ids=label_ids)

            loss = model_output['loss']
            preds = model_output['predictions']

            correct_predictions += torch.sum(preds == label_ids)
            total_train_loss_this_epoch += loss.item()
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return correct_predictions.double() / len(train_statements), np.mean(losses)

    @staticmethod
    # here statements and labels have train, test and validation all three parts
    def train_model(model, batcher, statements, metadata, categorical_numerical_data, labels, epochs, device, batch_size):
        optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

        # Total number of training steps is number of batches * number of epochs.
        total_steps = (len(statements['train']) // batch_size + 1) * epochs

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
            train_acc, train_loss = Model.train_epoch(model, optimizer, scheduler, batcher, statements['train'], metadata['train'], categorical_numerical_data['train'],
                                                      labels['train'], device)
            print(f'Train loss: {train_loss}, accuracy: {train_acc}')
            print(f'Epoch {epoch + 1} took {(time.time() - t0) / 60} minutes')

            # ========================================
            #               Validation
            # ========================================

            print('')
            print("Running Validation...")

            val_acc, val_loss = Model.eval_model(model, batcher, statements['validation'], metadata['validation'], categorical_numerical_data['validation'],  labels['validation'],
                                                 device)
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
    def eval_model(model, batcher, validation_statements, validation_metadata, validation_categorical_numerical_data, validation_labels, device):
        model.eval()
        losses = []

        correct_predictions = 0
        total_eval_loss = 0

        with torch.no_grad():
            # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert
            for batch in batcher.iter_batches_sentences_with_metadata_and_labels(validation_statements, validation_metadata, validation_categorical_numerical_data, validation_labels):
                # model_output['contextual_embeddings'] is (batch_size, seq_len, embed_dim) tensor of top layer activations
                # model_output has keys: dict_keys(['wiki', 'wordnet', 'loss', 'pooled_output', 'contextual_embeddings'])

                # need this at all?
                '''
                batch = move_to_device(batch, cuda_device=0)
                '''

                statements_tokens = Model.to_device(batch['tokens'], device)
                statements_segment_ids = Model.to_device(batch['segment_ids'], device)
                statements_candidates = Model.to_device(batch['candidates'], device)
                metadata_tokens = Model.to_device(batch['metadata_tokens'], device)
                metadata_segment_ids = Model.to_device(batch['metadata_segment_ids'], device)
                metadata_candidates = Model.to_device(batch['metadata_candidates'], device)
                categorical_numerical_data = Model.to_device(batch['categorical_numerical_data'], device)
                label_ids = Model.to_device(batch['label_ids'], device)

                # batch = Model.to_device(batch, device)

                # now output has loss, logits and predictions
                model_output = model(statements_tokens=statements_tokens, metadata_tokens=metadata_tokens,
                                     statements_segment_ids=statements_segment_ids,
                                     metadata_segment_ids=metadata_segment_ids,
                                     statements_candidates=statements_candidates,
                                     metadata_candidates=metadata_candidates,
                                     categorical_numerical_data=categorical_numerical_data, label_ids=label_ids)

                loss = model_output['loss']
                preds = model_output['predictions']

                # labels = batch['label_ids']

                correct_predictions += torch.sum(preds == label_ids)
                total_eval_loss += loss.item()
                losses.append(loss.item())

        return correct_predictions.double() / len(validation_statements), np.mean(losses)

    @staticmethod
    def get_predictions(model, batcher, test_statements, test_metadata, test_categorical_numerical_data, test_labels, device):
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []
        with torch.no_grad():
            # Predict
            # batcher takes raw untokenized sentences and yields batches of tensors needed to run KnowBert

            for batch in batcher.iter_batches_sentences_with_metadata_and_labels(test_statements, test_metadata, test_categorical_numerical_data, test_labels):

                statements_tokens = Model.to_device(batch['tokens'], device)
                statements_segment_ids = Model.to_device(batch['segment_ids'], device)
                statements_candidates = Model.to_device(batch['candidates'], device)
                metadata_tokens = Model.to_device(batch['metadata_tokens'], device)
                metadata_segment_ids = Model.to_device(batch['metadata_segment_ids'], device)
                metadata_candidates = Model.to_device(batch['metadata_candidates'], device)
                categorical_numerical_data = Model.to_device(batch['categorical_numerical_data'], device)
                label_ids = Model.to_device(batch['label_ids'], device)

                # batch = Model.to_device(batch, device)

                # now output has loss, logits and predictions
                model_output = model(statements_tokens=statements_tokens, metadata_tokens=metadata_tokens,
                                     statements_segment_ids=statements_segment_ids,
                                     metadata_segment_ids=metadata_segment_ids,
                                     statements_candidates=statements_candidates,
                                     metadata_candidates=metadata_candidates,
                                     categorical_numerical_data=categorical_numerical_data, label_ids=label_ids)

                preds = model_output['predictions']

                # labels = batch['label_ids']

                # Store predictions and true labels
                predictions.extend(preds)
                true_labels.extend(label_ids)

        predictions = torch.stack(predictions).cpu()
        true_labels = torch.stack(true_labels).cpu()
        return predictions, true_labels
