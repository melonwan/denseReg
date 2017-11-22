import matplotlib.pyplot as plt
import numpy.linalg as alg

class Evaluation(object):
    def __init__(self):
        pass

    @classmethod
    def maxJntError(cls_obj, skel1, skel2):
        diff = skel1.reshape(-1,3) - skel2.reshape(-1,3)
        diff = alg.norm(diff, axis=1)
        return diff.max() 

    @classmethod
    def meanJntError(cls_obj, skel1, skel2):
        diff = skel1.reshape(-1,3) - skel2.reshape(-1,3)
        diff = alg.norm(diff, axis=1)
        return diff.mean() 

    
    @classmethod
    def averageMaxJntError(cls_obj, score_list):
        score_list = sorted(score_list)

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=10.5):
                th_idx += 1
        print '10mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=20.5):
                th_idx += 1
        print '20mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=30.5):
                th_idx += 1
        print '30mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=40.5):
                th_idx += 1
        print '40mm percentage: %f'%(float(th_idx)/len(score_list))

        thresh_list = [thresh*5.0+0.5 for thresh in range(0, 17)]
        precent_list = [1]*len(thresh_list)

        cur_score_idx = 0
        for i in range(0, len(thresh_list)):
            th_idx = 0
            for j in range(0, len(score_list)):
                if(score_list[j]<thresh_list[i]):
                    th_idx += 1
            precent_list[i] = float(th_idx) / len(score_list)

        return (thresh_list, precent_list)

    @classmethod
    def plotError(cls_obj, score_list, fig_path):
        score_list = sorted(score_list)

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=10.5):
                th_idx += 1
        print '10mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=20.5):
                th_idx += 1
        print '20mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=30.5):
                th_idx += 1
        print '30mm percentage: %f'%(float(th_idx)/len(score_list))

        th_idx = 0
        for i in range(0, len(score_list)):
            if(score_list[i]<=40.5):
                th_idx += 1
        print '40mm percentage: %f'%(float(th_idx)/len(score_list))

        thresh_list = [thresh*5.0+0.5 for thresh in range(0, 17)]
        precent_list = [1]*len(thresh_list)

        cur_score_idx = 0
        for i in range(0, len(thresh_list)):
            th_idx = 0
            for j in range(0, len(score_list)):
                if(score_list[j]<thresh_list[i]):
                    th_idx += 1
            precent_list[i] = float(th_idx) / len(score_list)

        f = open(fig_path, 'w')
        for thresh, p in zip(thresh_list, precent_list):
            f.write('%f %f\n'%(thresh, p*100.))

        # plt.clf();
        # plt.plot(thresh_list, precent_list, '-', color='b')
        # plt.grid()
        # plt.savefig(fig_path)

