from kb.include_all import ModelArchiveFromParams, SimpleClassifier, F1Metric
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
from fake_news.liar.bert.model import Model
from sklearn.metrics import classification_report
from fake_news.liar.data_processor.data_processor import DataProcessor
from allennlp.models.archival import load_archive
import pickle
from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.training.metrics import CategoricalAccuracy

import torch


NUM_LABELS = 2
# NUM_LABELS = 6
# BATCH_SIZE = 32
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 128  # seems not useful
KNOWBERT_HIDDEN_SIZE = 768


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def main():
    class_names = ['True', 'Fake']

    # load labels and statements
    print('main - getting statements and labels')
    labels, statements = DataProcessor.load_dataset()
    # convert text labels to 0-5
    labels = {'train': DataProcessor.convert_labels(NUM_LABELS, labels['train']),
              'test': DataProcessor.convert_labels(NUM_LABELS, labels['test']),
              'validation': DataProcessor.convert_labels(NUM_LABELS, labels['validation'])}



    # archive = load_archive(model_archive_file)
    # model = archive.model
    #
    # reader = DatasetReader.from_params(archive.config['dataset_reader'])
    #
    # iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 32}))
    # vocab = Vocabulary.from_params(archive.config['vocabulary'])
    # iterator.index_with(vocab)
    #
    # model.cuda()
    # model.eval()
    #
    # label_ids_to_label = {0: 'F', 1: 'T'}
    #
    # instances = reader.read(test_file)
    # predictions = []
    # for batch in iterator(instances, num_epochs=1, shuffle=False):
    #     batch = move_to_device(batch, cuda_device=0)
    #     output = model(**batch)
    #
    #     batch_labels = [
    #         label_ids_to_label[i]
    #         for i in output['predictions'].cpu().numpy().tolist()
    #     ]
    #
    #     predictions.extend(batch_labels)


    # a pretrained model, e.g. for Wordnet+Wikipedia
    archive_file = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'

    #
    #
    # vocab = Vocabulary.from_params(archive.config['vocabulary'])

    # load model and batcher

    params = Params({"archive_file": archive_file})

    model = ModelArchiveFromParams.from_params(params=params)
    model.cuda()


    # model = SimpleClassifier(
    #     vocab=vocab,
    #     model=knowbert_model,
    #     task='classification',
    #     num_labels=NUM_LABELS,
    #     bert_dim=12,
    #     metric_a=CategoricalAccuracy(),
    #     dropout_prob=0.1
    # )
    # archive_model = ModelArchiveFromParams.from_params(params=params)
    # model = CustomKnowBertClassifier(archive_model, KNOWBERT_HIDDEN_SIZE, NUM_LABELS)
    batcher = KnowBertBatchifier(model_archive=archive_file, batch_size=BATCH_SIZE)


    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    # model.to(device)
    print('main - training')
    train_history = Model.train_model(model, batcher, statements, labels, EPOCHS, device, loss_fn, NUM_LABELS)

    # evaluate model on test dataset

    test_acc, _ = Model.eval_model(model, batcher, statements['test'], labels['test'], device, loss_fn, NUM_LABELS)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, pred_probs, test_labels = Model.get_predictions(model, batcher, statements['test'], labels['test'], device,
                                                          NUM_LABELS)

    with open('record.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(pred_probs, f)
        pickle.dump(test_labels, f)

    print(classification_report(test_labels, pred, target_names=class_names))


if __name__ == '__main__':
    main()
