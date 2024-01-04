import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray

__all__ = ['SegmentationMetric']



class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    supported_nets = ["unet", "unetpp", "resunetpp", "mobileunet", "mobilev2unetpp"]
    x=[1,2,3,4]

    net_a = [[None for _ in range(4)] for _ in range(5)]
    for j,net in enumerate(supported_nets):
        root=r"E:\MobileV2Unet++\base\predict\pre_"+net
        nums=len(os.listdir(root))//2

        pa_values=[]
        cpa_values=[]
        mpa_values=[]
        mIoU_values=[]
        pa_mean=0
        cpa_mean=0
        mpa_mean=0
        mIoU_mean=0
        for i in range(nums):
            imgLabel = asarray(Image.open(root+"\predict_%d_g.png" % i))
            imgPredict = asarray(Image.open(root+ "\predict_%d_o.png" % i))
            imgPredict = np.where(imgPredict >= 1, 1, imgPredict)
            imgLabel = np.where(imgLabel >= 1, 1, imgLabel)
            metric = SegmentationMetric(2)
            metric.addBatch(imgPredict, imgLabel)
            pa_values.append(metric.pixelAccuracy())
            cpa_values.append(metric.classPixelAccuracy()[1])
            mpa_values.append(metric.meanPixelAccuracy())
            mIoU_values.append(metric.meanIntersectionOverUnion())
        pa_mean=sum(pa_values) / len(pa_values)
        cpa_mean=sum(cpa_values) / len(cpa_values)
        mpa_mean=sum(mpa_values) / len(mpa_values)
        mIoU_mean=sum(mIoU_values) / len(mIoU_values)

        net_a[j][0]=pa_mean
        net_a[j][1]=cpa_mean
        net_a[j][2]=mpa_mean
        net_a[j][3]=mIoU_mean


        print(net+'_pa is : %f' % pa_mean)
        print(net+'_cpa is :%f' % cpa_mean)
        print(net+'_mpa is : %f' % mpa_mean)
        print(net+'_mIoU is : %f' % mIoU_mean)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    line1, = plt.plot(x, net_a[0], color='b', marker='o', linestyle='--', markersize=10, alpha=0.5, linewidth=3)
    line2, = plt.plot(x, net_a[1], color='g', marker='s', linestyle='--', markersize=10, alpha=0.5, linewidth=3)
    line3, = plt.plot(x, net_a[2], color='r', marker='p', linestyle='--', markersize=10, alpha=0.5, linewidth=3)
    line4, = plt.plot(x, net_a[3], color='c', marker='*', linestyle='--', markersize=10, alpha=0.5, linewidth=3)
    line5, = plt.plot(x, net_a[4], color='c', marker='h', linestyle='--', markersize=10, alpha=0.5, linewidth=3)
    plt.xlabel(u'100epoch内top-1', fontsize=11, color='r')
    plt.ylabel(u'', fontsize=14, color='b')
    plt.title(u"语义分割指标", fontsize=14, color='k')
    plt.legend([line1, line2,line3,line4,line5], ["unet", "unet++", "resunet++", "mobileunet", "mobilev2unet++(ours)"], loc='lower right')
    a = [1, 2, 3, 4]
    labels = ["pa", "cpa","mpa","mIou"]
    plt.xticks(a, labels, rotation=0, fontsize=12)
    plt.show()


