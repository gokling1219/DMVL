import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV
import joblib
from kappa import kappa
from scipy.io import loadmat
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
#import input_data_PU

num_labeled = 5

# 加载数据


# 独热编码转换为类别标签
#train_label=np.argmax(train_label,1)
#print(train_label)
#test_label=np.argmax(test_label,1)
#print(test_label)
#print(np.unique(test_label))


################################################### KNN训练及分类 ###################################################
OA_list = [0,0,0,0,0,0,0,0,0,0]
AA_list = [0,0,0,0,0,0,0,0,0,0]
kappa_list = [0,0,0,0,0,0,0,0,0,0]
CA_list = np.zeros((10,16))

number = 10249

seed_number = [1,2,3,4,5,6,7,8,9,10]

for INDEX in range(10):

    np.random.seed(seed_number[INDEX])

    f = h5py.File('data/IP_feature.h5', 'r')
    data = f['data'][:]
    print(data.shape)
    label = f['label'][:]
    print(label.shape)
    f.close()

    # m, n = data.max(), data.min()
    # data = (data - n) / (m - n)

    indices = np.arange(data.shape[0])
    shuffled_indices = np.random.permutation(indices)
    images = data[shuffled_indices]
    labels = label[shuffled_indices]
    y = labels  # np.array([numpy.arange(9)[l==1][0] for l in labels])
    n_classes = int(y.max() + 1)
    i_labeled = []
    for c in range(n_classes):
        i = indices[y == c][:num_labeled]  # 50
        i_labeled += list(i)
    l_images = images[i_labeled]
    l_labels = y[i_labeled]




    start = time.time()
    C = np.logspace(-2, 8, 11, base=2)  # 2为底数，2的-2次方到2的8次方，一共11个数
    gamma = np.logspace(-2, 8, 11, base=2)

    parameters = {'C': C,
                  'gamma': gamma}
    # 问题：参数设置规律？？？

    clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=3, refit=True)
    # Refit an estimator using the best found parameters on the whole dataset.

    #print(l_images, l_labels)
    clf.fit(l_images, l_labels)

    print("time = ", time.time() - start)

    # 输出最佳参数组合
    # print('随机搜索-度量记录：',clf.cv_results_)  # 包含每次训练的相关信息
    print('随机搜索-最佳度量值:', clf.best_score_)  # 获取最佳度量值
    print('随机搜索-最佳参数：', clf.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
    print('随机搜索-最佳模型：', clf.best_estimator_)  # 获取最佳度量时的分类器模型

    # 存储结果学习模型，方便之后的调用
    # joblib.dump(clf.best_estimator_, r'.\IP\IP_BEST_MODEL_' + str(INDEX + 1) + '.m')
    #
    # clf = joblib.load(r".\IP\IP_BEST_MODEL_" + str(INDEX + 1) + ".m")

    start = time.time()
    predict_label = clf.predict(data)  # (42776,)

    matrix = metrics.confusion_matrix(label, predict_label)
    print(matrix)

    print('OA = ', np.sum(np.trace(matrix)) / float(number) * 100)
    OA_list[INDEX] = np.sum(np.trace(matrix)) / float(number) * 100

    kappa_temp, aa_temp, ca_temp = kappa(matrix, 16)
    AA_list[INDEX] = aa_temp
    CA_list[INDEX] = ca_temp
    kappa_list[INDEX] = kappa_temp * 100
    print("kappa = ", kappa_temp * 100)

    gt = loadmat('D:\hyperspectral_data\Indian_pines_gt.mat')
    gt = gt['indian_pines_gt']

    # 将预测的结果匹配到图像中
    new_show = np.zeros((gt.shape[0], gt.shape[1]))
    k = 0
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i][j] != 0:
                new_show[i][j] = predict_label[k]
                new_show[i][j] += 1
                k += 1

    np.save("IP/IP_" + str(np.sum(np.trace(matrix)) / float(number) * 100) + "npy", new_show)
    # print new_show.shape

    # 展示地物

    colors = ['black', 'gray', 'lime', 'cyan', 'forestgreen', 'hotpink', 'saddlebrown',
              'purple', 'red', 'yellow', 'blue', 'steelblue', 'olive', 'sandybrown', 'mediumaquamarine',
              'darkorange',
              'whitesmoke']

    cmap = mpl.colors.ListedColormap(colors)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(new_show, cmap=cmap)
    plt.savefig("IP/IP_" + str(np.sum(np.trace(matrix)) / float(number)) + "_" + str(INDEX + 1) + ".png", dpi=1000)  # 保存图像
    #plt.show()
    print("time = ", time.time() - start)

print("\n")
print("OA_list", OA_list)
print(np.array(OA_list).mean())
print(np.array(OA_list).std())
print("AA_list", AA_list)
print(np.array(AA_list).mean())
print(np.array(AA_list).std())
print("kappa_list", kappa_list)
print(np.array(kappa_list).mean())
print(np.array(kappa_list).std())
print("CA_list", CA_list)
print(np.array(CA_list).mean(axis=0))
print(np.array(CA_list).std(axis=0))

f = open('IP/IP_results.txt', 'w')
for index in range(CA_list.mean(axis=0).shape[0]):
    f.write(str(np.array(CA_list).mean(axis=0)[index]) + '\n')
f.write(str(np.array(OA_list).mean()) + '\n')
f.write(str(np.array(AA_list).mean()) + '\n')
f.write(str(np.array(kappa_list).mean()) + '\n')
f.write("\n\n\n")
for index in range(CA_list.std(axis=0).shape[0]):
    f.write(str(np.array(CA_list).std(axis=0)[index]) + '\n')
f.write(str(np.array(OA_list).std()) + '\n')
f.write(str(np.array(AA_list).std()) + '\n')
f.write(str(np.array(kappa_list).std()) + '\n')