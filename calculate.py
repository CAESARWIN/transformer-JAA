import numpy as np
import logging


def statistics(pred, y, lenth):
    class_nb = 12

    statistics_list = []
    for j in range(class_nb):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(lenth):
            if pred[i][j] == 1:
                if y[i][j] == 1:
                    TP += 1
                elif y[i][j] == 0:
                    FP += 1
                else:
                    assert False
            elif pred[i][j] == 0:
                if y[i][j] == 1:
                    FN += 1
                elif y[i][j] == 0:
                    TN += 1
                else:
                    assert False
            else:
                assert False
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list


def calc_f1_score(statistics_list):
    f1_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']

        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)

    return mean_f1_score, f1_score_list


def calc_acc_score(statistics_list):
    acc_score_list = []

    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        TN = statistics_list[i]['TN']

        accuracy = (TP + TN) / (TP + TN + FN + FP + 1e-20)

        acc_score_list.append(accuracy)
    mean_acc_score = sum(acc_score_list) / len(acc_score_list)

    return mean_acc_score, acc_score_list


def update_statistics_list(old_list, new_list):
    if not old_list:
        return new_list

    assert len(old_list) == len(new_list)

    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']

    return old_list


if __name__ == '__main__':
    '''
    logging.basicConfig(level=logging.INFO,
                        format='(%(asctime)s %(levelname)s) %(message)s',
                        datefmt='%d %b %H:%M:%S',
                        filename='logs/test.log',
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('(%(levelname)s) %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    '''
    epoch = 1
    y = np.loadtxt("/home/zhangchenggong/Competition/list/AU_list/AU_val_AUoccur.txt")
    AUoccur_pred_prob = np.loadtxt("/home/zhangchenggong/Competition/GCN-pretrained-JAA/val_predAUprob_all_.txt")
    
    # AUs
    pred = np.zeros(AUoccur_pred_prob.shape)
    pred[AUoccur_pred_prob < 0.5] = 0
    pred[AUoccur_pred_prob >= 0.5] = 1

    lenth = len(y)

    statistics_list = statistics(pred, y, lenth)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc_score, accuracy_list = calc_acc_score(statistics_list)

    f1_score_list = ['%.4f' % f1_score for f1_score in f1_score_list]

    accuracy_list = ['%.4f' % acc_score for acc_score in accuracy_list]

    print('epoch[%d] mean_f1_score:%.4f mean_accuracy:%.4f'
          '\n\tf1_score_list:%s\n\taccuracy_list:%s'
          % (epoch, mean_f1_score, mean_acc_score, f1_score_list, accuracy_list))

    logging.info('[TEST] epoch[%d] mean_f1_score:%.4f mean_accuracy:%.4f %s %s'
                 % (epoch, mean_f1_score, mean_acc_score, f1_score_list, accuracy_list))
