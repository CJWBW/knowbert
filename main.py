from kb.include_all import SimpleClassifier
from kb.knowbert_utils import KnowBertBatchifier
from fake_news.liar.bert.model import Model, CustomClassifier
from sklearn.metrics import classification_report
from fake_news.liar.data_processor.data_processor import DataProcessor
from allennlp.models.archival import load_archive
import pickle
from allennlp.data import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy
import torch


NUM_LABELS = 2
BATCH_SIZE = 16
EPOCHS = 4


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device

    else:
        # print('No GPU available, using the CPU instead.')
        # device = torch.device("cpu")
        print('no GPU')
        exit()


def main():
    device = get_device()

    if NUM_LABELS == 2:
        class_names = ['True', 'Fake']
    else:
        class_names = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    # load labels and statements
    print('main - getting statements and labels')
    data_processor = DataProcessor()
    labels, statements, metadata, categorical_numerical_data = data_processor.load_dataset()
    # convert text labels to 0-5
    labels = {'train': DataProcessor.convert_labels(NUM_LABELS, labels['train']),
              'test': DataProcessor.convert_labels(NUM_LABELS, labels['test']),
              'validation': DataProcessor.convert_labels(NUM_LABELS, labels['validation'])}

    archive_file_wiki_wordnet = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_wordnet_model.tar.gz'
    archive_file_wordnet = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wordnet_model.tar.gz'
    archive_file_wiki = 'https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wiki_model.tar.gz'

    archive_file = archive_file_wiki_wordnet

    archive = load_archive(archive_file)
    knowbert_model = archive.model
    vocab = Vocabulary.from_params(archive.config['vocabulary'])

    model = CustomClassifier(
        vocab=vocab,
        model=knowbert_model,
        num_labels=NUM_LABELS,
        bert_dim=768,
        metric=CategoricalAccuracy()
    )

    batcher = KnowBertBatchifier(model_archive=archive_file, batch_size=BATCH_SIZE)

    model.to(device)
    print('main - training')
    train_history = Model.train_model(model, batcher, statements, metadata, categorical_numerical_data, labels, EPOCHS, device, BATCH_SIZE)

    # evaluate model on test dataset

    test_acc, _ = Model.eval_model(model, batcher, statements['test'], metadata['test'], categorical_numerical_data['test'], labels['test'], device)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, test_labels = Model.get_predictions(model, batcher, statements['test'], metadata['test'], categorical_numerical_data['test'], labels['test'], device)

    with open('record_.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(test_labels, f)

    print(classification_report(test_labels, pred, target_names=class_names))


if __name__ == '__main__':
    main()
