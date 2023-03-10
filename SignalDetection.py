from statistics import NormalDist
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math

class SignalDetection:
    def __init__(self, hits, misses, falseAlarm, correctRejections):
        self.hit = hits
        self.misses = misses
        self.falseAlarm = falseAlarm
        self.correctRejections = correctRejections
        self.hit_rate = hits / (hits + misses)
        self.false_alarm_rate = falseAlarm / (falseAlarm + correctRejections)
        self.hit_dist = NormalDist().inv_cdf(self.hit_rate)
        self.false_dist = NormalDist().inv_cdf(self.false_alarm_rate)
    def d_prime(self):
        d = self.hit_dist - (self.false_dist)
        return d
    def criterion(self):
        c = (-0.5) * ((self.hit_dist) + (self.false_dist))
        return c
    def __add__(self, other):
        return SignalDetection(self.hit + other.hit, self.misses + other.misses, self.falseAlarm + other.falseAlarm, self.correctRejections + other.correctRejections)
    def __mul__(self, scalar):
        return SignalDetection(self.hit * scalar, self.misses * scalar, self.falseAlarm * scalar, self.correctRejections * scalar)
    def plot_hit_false(self):
       x = [0, self.hit_rate, 1]
       y = [0, self.false_alarm_rate, 1]
       plt.plot(x, y, 'b')
       plt.plot(self.hit_rate, self.false_alarm_rate, 'bo')
       plt.xlabel("Hit rate")
       plt.ylabel("False alarm rate")
       plt.title("ROC curve")
       plt.show()
       
    def plot_std(self):
        x = np.arange(-4, 4, 0.01)
        #N
        plt.plot(x, scipy.stats.norm.pdf(x, 0, 1), 'b', label = "N")
        #S
        plt.plot(x, scipy.stats.norm.pdf(x, self.d_prime(), 1), 'r', label = "S")
        #C
        plt.axvline((self.d_prime()/2) + self.criterion(),color = 'black', linestyle = '--').set_label("C")
        #D
        plt.plot([self.d_prime(), 0], [0.4, 0.4], 'k', label = "D")

        plt.xlabel("Decision variable")
        plt.ylabel("Probability")
        plt.title("Signal Detection Theory")
        plt.legend()
        plt.show()

    @staticmethod
    def simulate(dPrime, criteriaList, signalCount, noiseCount):
        s_list = []
        for i in range(len(criteriaList)):
            k = criteriaList[i] + (dPrime / 2)
            hit_rate = 1 - scipy.stats.norm.cdf(k - dPrime)
            false_alarm_rate = 1 - scipy.stats.norm.cdf(k)
            hits = np.random.binomial(signalCount, hit_rate)
            misses = signalCount - hits
            falseAlarm = np.random.binomial(noiseCount, false_alarm_rate)
            correctRejections = noiseCount - falseAlarm
            s_list.append(SignalDetection(hits, misses, falseAlarm, correctRejections))
        return s_list

    @staticmethod
    def plot_roc(sdtList):
        for i in range(len(sdtList)):
            plt.plot(sdtList[i].false_alarm_rate, sdtList[i].hit_rate, 'b')
        plt.plot(0, 0, 'b')
        plt.plot(1, 1, 'b')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit rate")
        plt.title("ROC curve")               

    def nLogLikelihood(self, hitRate, falseAlarmRate):
        likelihood = (-(self.hit) * (math.log(hitRate))) - (self.misses * (math.log(1 - hitRate))) - (self.falseAlarm * (math.log(falseAlarmRate))) - (self.correctRejections * (math.log(1 - falseAlarmRate)))
        return likelihood