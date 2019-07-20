#-*- coding:utf-8 -*-
from sklearn import svm

def SVM(X, y):
	# C：错误项的惩罚系数。C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。相反，减小C的话，容许训练样本中有一些误分类错误样本，泛化能力强。对于训练样本带有噪声的情况，一般采用后者，把训练样本集中错误分类的样本作为噪声。
    # kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
    # kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    # kernel='poly'时，多项式函数,degree 表示多项式的程度-----支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
    # kernel='sigmoid'时，支持非线性分类。更高gamma值，将尝试精确匹配每一个训练数据集，可能会导致泛化误差和引起过度拟合问题。
    # gamma：float参数 默认为auto。核函数系数，只对‘rbf’,‘poly’,‘sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features.
    # decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
    # decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    # probability：bool参数 默认为False，是否启用概率估计。 这必须在调用fit()之前启用，并且会fit()方法速度变慢。
    # cache_size：float参数 默认为200，指定训练所需要的内存，以MB为单位，默认为200MB。
    # class_weight：字典类型或者‘balance’字符串。默认为None，给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C.如果给定参数‘balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。
    # max_iter ：int参数 默认为-1，最大迭代次数，如果为-1，表示不限制
	model = svm.SVC(kernel='linear', C=1,) 
	# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
	model.fit(X, y)
	model.score(X, y)
	#Predict Output
	#predicted= model.predict(x_test)

	#joblib.dump(model, './svm.model')
	return model

def evalue(model,X_train, y_train, X_test, y_test):
    """
    评估
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:accurate
    """
    #model = joblib.load('../Models/WaterEval_svm.model')
    #print("accurate of train：" + str(model.score(X_train, y_train)))
    #print("accurate of test:" + str(model.score(X_test, y_test)))
    return model.score(X_train, y_train),model.score(X_test, y_test)#accurate