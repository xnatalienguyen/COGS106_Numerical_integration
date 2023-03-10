from statistics import NormalDist
import numpy as np
import math

class SignalDetection:
    def __init__(self, hits, misses, falseAlarm, correctRejections):
        self.hit = hits
        self.misses = misses
        self.falseAlarm = falseAlarm
        self.correctRejections = correctRejections
        self.hit_rate = hits / (hits + misses)
        self.false_alarm_rate = falseAlarm / (falseAlarm + correctRejections)
        self.hit_dist = NormalDist().inv_cdf(self_hit.rate)
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
       x = [0, self.__hit_rate, 1]
       y = [0, self.__false_alarm_rate, 1]
       plt.plot(x, y, 'b')
       plt.plot(self.__hit_rate, self.__false_alarm_rate, 'bo')
       plt.xlabel("Hit rate")
       plt.ylabel("False alarm rate")
       plt.title("ROC curve")
       plt.show()
       
    def plot_std(self):
        x = np.arange(-4, 4, 0.01)
        #N
        plt.plot(x, norm.pdf(x, 0, 1), 'b', label = "N")
        #S
        plt.plot(x, norm.pdf(x, self.d_prime(), 1), 'r', label = "S")
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
            k = criteriaList[i] + (dprime / 2)
            hit_rate = 1 - norm.cdf(k - dprime)
            false_alarm_rate = 1 - norm.cdf(k)
            hits = np.random.binomial(signalCount, hit_rate)
            misses = signalCount - hits
            falseAlarm = np.random.binomail(noiseCount, false_alarm_rate)
            correctRejections = noiseCount - falseAlarm
            s_list.append(SignalDetection(hits, misses, falseAlarm, correctRejections))
        return s_list
    
    def nLogLikelihood(self, hitRate, falseAlarmRate):
        likelihoood = (-(self.hit) * (math.log(hitRate))) - (self.misses * (math.log(1 - hitRate))) - (self.falseAlarm * (math.log(falseAlarmRate))) - (self.correctRejections * (math.log(1 - falseAlarmRate)))
        return likelihood
        
import unittest 

class TestSignalDetection(unittest.TestCase):
    def test_d_prime_zero(self):
        sd   = SignalDetection(15, 5, 15, 5)
        expected = 0
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_d_prime_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        expected = -0.421142647060282
        obtained = sd.d_prime()
        # Compare calculated and expected d-prime
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_zero(self):
        sd   = SignalDetection(5, 5, 5, 5)
        # Calculate expected criterion        
        expected = 0
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_criterion_nonzero(self):
        sd   = SignalDetection(15, 10, 15, 5)
        # Calculate expected criterion        
        expected = -0.463918426665941
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertAlmostEqual(obtained, expected, places=6)
    def test_addition(self):
        sd = SignalDetection(1, 1, 2, 1) + SignalDetection(2, 1, 1, 3)
        expected = SignalDetection(3, 2, 3, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
    def test_multiplication(self):
        sd = SignalDetection(1, 2, 3, 1) * 4
        expected = SignalDetection(4, 8, 12, 4).criterion()
        obtained = sd.criterion()
        # Compare calculated and expected criterion
        self.assertEqual(obtained, expected)
    def test_corruption(self):
        sd = SignalDetection(1, 2, 3, 1)
        sd.hit = 5
        sd.misses = 5
        sd.correctRejections = 5
        sd.falseAlarm = 5
        sd.hit_rate = 5
        sd.false_alarm_rate = 5
        sd.hit_dist = 5
        sd.false_dist = 5
        expected_c = SignalDetection(1, 2, 3, 1).criterion()
        obtained_c = sd.criterion()
        expected_d = SignalDetection(1, 2, 3, 1).d_prime()
        obtained_d = sd.d_prime()
        self.assertEqual(obtained_c, expected_c)
        self.assertEqual(obtained_d, expected_d)

    def test_simulate_single_criterion(self):
        """
        Test SignalDetection.simulate method with a single criterion value.
        """
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000
        
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]
        
        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.falseAlarms      , sdtList[0].falseAlarms)
        self.assertEqual(sdt.correctRejections, sdtList[0].correctRejections)

    def test_simulate_multiple_criteria(self):
        """
        Test SignalDetection.simulate method with multiple criterion values.
        """
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual (sdt.hits              ,  signalCount)
            self.assertLessEqual (sdt.misses            ,  signalCount)
            self.assertLessEqual (sdt.falseAlarms       ,  noiseCount)
            self.assertLessEqual (sdt.correctRejections ,  noiseCount)
   
    def test_nLogLikelihood(self):
        """
        Test case to verify nLogLikelihood calculation for a SignalDetection object.
        """
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)
        
    def test_rocLoss(self):
        """
        Test case to verify rocLoss calculation for a list of SignalDetection objects.
        """
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)
        
    def test_integration(self):
        """
        Test case to verify integration of SignalDetection simulation and ROC fitting.
        """
        dPrime  = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat    = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()

if __name__ == '__main__':
    unittest.main()
