import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

class Wine:
    def __init__(self):
        self.df = pd.read_csv("winequality-red.csv")
        self.init_columns = self.df.columns
        self.labels=self.df["quality"]
        self.thresholds=[]
        self.cluster_centers=[]
        self.normalised = pd.DataFrame()
        self.decision_entropy = []
        self.TDE = 0 # total decision entropy
   
    def normalise(self): # to normalise the data frame to such that each column has values between 0 and 1
        x = self.df.values 
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)# uses formula (xi - min)/(max - min) 
        self.df = pd.DataFrame(x_scaled)
        self.df[11]=self.labels
    
    def radius(self,column_number):
        a=np.array(self.df[column_number])
        b=np.mean(a)
        radius = np.max(a)-b# mean is either distance to maximum or minimum, from the centroid
        return radius
   
    def assign_clusters(self):
        columns = self.df.columns
        for i in columns:
            threshold = self.radius(i)*0.45
            col=np.resize(self.df[i].values,(len(self.df),1)) # the column in a numpy format
            k=2
            while k<=10 :
                kmeans = KMeans(n_clusters=k).fit(col)
                cc=kmeans.cluster_centers_
                count = 0 # number of points that are outliers
                for row in xrange(len(self.df)):
                    x=col[row][0]
                    num_clusters = 0
                    for j in cc: # to check if the entry in the data frame lies in one cluster at least
                        distance = abs(j-x)
                        if distance<threshold:
                            num_clusters = 1
                    if num_clusters == 0:
                        count = count + 1
                if count == 0: # if only 0 outliers are there stick to the number of clusters generated
                    
                    break
                else :
                    k = k + 1
                    continue
            k = min(10,k)
            kmeans = KMeans(n_clusters=k).fit(col)
            self.df[i] = kmeans.predict(col)+1
            self.thresholds.append(threshold)
            self.cluster_centers.append(kmeans.cluster_centers_)
    
    def reassign_clusters(self,col_number,threshold):#i is column number
            i=col_number
            col=np.resize(self.normalised[i].values,(len(self.df),1))
            k=2
            while k<=10 :
                kmeans = KMeans(n_clusters=k).fit(col)
                cc=kmeans.cluster_centers_
                count = 0 # number of points that are outliers
                for row in xrange(len(self.df)):
                    x=col[row][0]
                    num_clusters = 0
                    for j in cc: # to check if the entry in the data frame lies in one cluster at least
                        distance = abs(j-x)
                        if distance<threshold:
                            num_clusters = 1
                    if num_clusters == 0:
                        count = count + 1
                if count == 0: # if only 3 outliers or less are there stick to the number of clusters generated
                    
                    break
                else :
                    k = k + 1
                    continue
            k = min(10,k)
            kmeans = KMeans(n_clusters=k).fit(col)
            self.df[i] = kmeans.predict(col)+1
            self.thresholds[col_number]=threshold
            self.cluster_centers[col_number]=kmeans.cluster_centers_
        
        
    
    
    def plot_graphs(self):
        columns = self.df.columns
        for i in columns:
            col=np.resize(self.df[i].values,(len(self.df),1))
            threshold = np.arange(0.01,self.radius(i),0.01)
            resulting_k = []
            for j in threshold:
                num_excluded = 1
                k=2
                while num_excluded > 0:
                    kmeans = KMeans(n_clusters=k).fit(col)
                    cc=kmeans.cluster_centers_
                    num_excluded = 0 
                    for row in xrange(len(self.df)):
                        x=col[row][0]
                        num_clusters = 0
                        for z in cc: # to check if the entry in the data frame lies in one cluster at least
                            distance = abs(z-x)
                            if distance<j:
                                num_clusters = 1
                        if num_clusters == 0:
                                num_excluded = num_excluded + 1
                    k = k+1
                resulting_k.append(k)
            filename="plot"+str(i)+".png"
            plot=plt.figure()
            plt.figure(figsize=(5,5))
            plt.ylim(0,100)
            plt.xlim(0,self.radius(i))
            plt.title("column "+str(i))
            plt.xlabel("lambda")
            plt.ylabel("k")
            plt.plot(threshold,np.array(resulting_k),color="black")            
            plt.savefig(filename)
     
    def read_file(self, filename):
        names= range(0,12)
        self.normalised  = pd.read_csv(filename,names=names)
        self.normalised.drop(self.normalised.index[0],inplace = True)
        self.normalised[11]=self.labels
       
    def total_decision_entropy(self,attr=-1): # number of equivalence classes 
        if attr==-1:
            df = self.df
        else :
            df = self.df.drop(attr,1)
        dic = {}
        for i in range(len(df)):
            a=tuple( df.iloc[i][:-1])
            b=list( df.iloc[i][-1:])
            if not a in dic :
                dic[a]=[]
                dic[a].append(i)
            else:
                dic[a].append(i)
      
        z=[]
        for i in dic:
            x=dic[i]
            if len(x)==1:
                continue
            a=x[0]
            val = self.df.iloc[a][11]
            flag = 1
            for j in x:
                temp = self.df.iloc[j][11]
                if temp != val :
                    flag = 0 
                    break
            if flag==0 :
                z.append(i)

        for i in z:
            dic.pop(i)
        if attr==-1:
            self.TDE=len(dic)
            return 
        else:
            return len(dic)
    
    def calc_significance_measures(self):
        l = len(self.df.columns)-1
        self.decision_entropy=[]
        self.significances=[]
        for i in xrange(l):
            sig = self.total_decision_entropy(i)
            self.decision_entropy.append(sig)
            self.significances.append((self.TDE-sig)/1600.0)
        return 
         
    def plot_graphs2(self):
        temp=self.normalised.copy()
        self.df=temp
        columns = self.df.columns
        qmean=np.mean(self.df[11])
        for i in columns:
            col=np.resize(self.df[i].values,(len(self.df),1))
            threshold = np.arange(0.05,0.5,0.05)
            resulting_k = []
            for j in threshold:
                num_excluded = 1
                k=2
                while num_excluded > 0 and k<200:
                    kmeans = KMeans(n_clusters=k).fit(col)
                    cc=kmeans.cluster_centers_
                    num_excluded = 0
                    print k
                    for row in xrange(len(self.df)):
                        x=col[row][0]
                        num_clusters = 0
                        for z in cc: # to check if the entry in the data frame lies in one cluster at least
                            distance = np.sqrt(((qmean-self.df.iloc[row][11])**2)+(x-z)**2)
                            if distance<j:
                                num_clusters = 1
                        if num_clusters == 0:
                                num_excluded = num_excluded + 1
                    k = k+1
                resulting_k.append(k)
            filename="s_plot"+str(i)+".png"
            plot=plt.figure()
            plt.figure(figsize=(5,5))
            plt.ylim(0,300)
            plt.xlim(0,0.8)
            plt.title("column "+str(i))
            plt.xlabel("lambda")
            plt.ylabel("k")
            plt.plot(threshold,np.array(resulting_k),color="black")            
            plt.savefig(filename)   
    def assign_clusters2(self):
        columns = self.df.columns
        for i in columns:
            threshold = 0.01
            col=np.resize(self.df[i].values,(len(self.df),1)) # the column in a numpy format
            k=2
            while k<=50 :
                kmeans = KMeans(n_clusters=k).fit(col)
                cc=kmeans.cluster_centers_
                count = 0 # number of points that are outliers
                for row in xrange(len(self.df)):
                    x=col[row][0]
                    num_clusters = 0
                    for j in cc: # to check if the entry in the data frame lies in one cluster at least
                        distance = abs(j-x)
                        if distance<threshold:
                            num_clusters = 1
                    if num_clusters == 0:
                        count = count + 1
                if count == 0: # if only 0 outliers are there stick to the number of clusters generated
                    
                    break
                else :
                    k = k + 1
                    continue
            k = min(50,k)
            kmeans = KMeans(n_clusters=k).fit(col)
            self.df[i] = kmeans.predict(col)+1
            self.thresholds.append(threshold)
            self.cluster_centers.append(kmeans.cluster_centers_)
            

def main():
        wine = Wine()
        wine.normalise()
        #wine.plot_graphs()
        df = pd.DataFrame(wine.df)
        df.to_csv("NormalisedBeforeCompatibility.csv")
        wine.read_file("NormalisedBeforeCompatibility.csv")
        wine.assign_clusters()
        wine.df[11]=wine.labels
        df = pd.DataFrame(wine.df) 
        df.to_csv("ResultBeforeReassigning.csv")
        dd=pd.DataFrame()
        dd['thresholds'] = wine.thresholds
        dd['cluster_centers'] = wine.cluster_centers
        dd.to_csv("threshold_clusterCentersBefore.csv")


        reassign_array = [0,2,5,6,7,8,9,10]
        thresholds = [0.26,0.42,0.45,0.3,0.36,0.28,0.28,0.24]
        for x,y in zip(reassign_array,thresholds):
            wine.reassign_clusters(x,y)

        wine.df[11]=wine.labels
        df=pd.DataFrame(wine.df)
        df.to_csv("ResultAfterReassigning.csv")
        dd=pd.DataFrame()
        dd['thresholds'] = wine.thresholds
        dd['cluster_centers'] = wine.cluster_centers
        dd.to_csv("threshold_clusterCentersAfter.csv")


        wine.total_decision_entropy()

        wine.calc_significance_measures()

        print "The total decision entropy without compatibility considerations is ", wine.TDE

        print "The decision entropies of each column without compatibility considerations are "
        print (wine.decision_entropy)

        print "The significances without compatibility considerations are "
        print (wine.significances)
        sig_before = wine.significances

        a=wine.normalised.copy()
        k=a[11]*a[11]
        for i in a.columns[:-1]:
            a[i]=a[i]+k
        wine.df=a

        wine.normalise()
        df = pd.DataFrame(wine.df)
        df.to_csv("NormalisedAfterCompatibility.csv")
        wine.assign_clusters2()
        wine.df[11]=wine.labels
        df=pd.DataFrame(wine.df)
        df.to_csv("ResultAfterCompatibility.csv")
        dd=pd.DataFrame()
        dd['thresholds'] = wine.thresholds
        dd['cluster_centers'] = wine.cluster_centers
        dd.to_csv("threshold_clusterCentersAfterCompatibility.csv")
        wine.total_decision_entropy()
        wine.calc_significance_measures()

        print "The total decision entropy after compatibility considerations is ", wine.TDE

        print "The decision entropies of each column after compatibility considerations are "
        print (wine.decision_entropy)

        print "The significances after compatibility considerations are "
        print (wine.significances)
        sig_after = wine.significances

        print "BEFORE vs AFTER"
        
        print (sig_before)
        print (sig_after)
        
if __name__ == '__main__':
    main()