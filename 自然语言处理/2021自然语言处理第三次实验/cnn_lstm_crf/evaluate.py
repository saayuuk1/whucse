import time
from collections import Counter
from models.crf import CRFModel
from models.model_crf import Model
from utils import save_model, flatten_lists
from evaluating import Metrics


def train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, word_emb, opt, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    model = Model(vocab_size, out_size, word_emb, opt)
    model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "{}{}{}".format(opt.model, '+CRF' if opt.crf else '', '+w2v' if opt.use_w2v else '')
    save_model(model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists
