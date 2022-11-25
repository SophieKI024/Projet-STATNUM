import pandas as pd # Pour lire les fichiers
import numpy as np # Pour effectuer des calculs mathématiques
import matplotlib.pyplot as plt # Pour réaliser des graphiques
from scipy import stats # Pour effectuer des calculs statistiques
from sklearn.preprocessing import StandardScaler # Pour normaliser les données
from sklearn import decomposition # Pour effectuer une ACP
import seaborn as sns


from scipy.cluster.hierarchy import dendrogram, linkage # Pour calculer un dendrogramme
from sklearn.cluster import AgglomerativeClustering # Pour effectuer une CAH
from sklearn.cluster import KMeans # Pour effectuer un K-means

pd.options.display.width = 0
df_init = pd.read_csv('white wine.txt', sep=';')


df = df_init
df_no_quality = df.drop("quality")

#print(df[df['quality'] == 0].mean(),df[df['Potability'] == 1].mean())
rows = df.columns
#####################################################################################################################
# FILL THE GAPS IN THE DATAFRAME

# from sklearn.impute import KNNImputer
#
# imputer = KNNImputer(n_neighbors=10, weights="uniform")
# l=imputer.fit_transform(df_init)
# df=pd.DataFrame(l,columns=df_init.columns)


#####################################################################################################################
# REMOVE NAN ROWS

df = df_init.dropna(inplace = False)
df.reset_index(drop = True, inplace = True)

df2 = df_init.dropna(inplace = False)
df2.reset_index(drop = True, inplace = True)
#####################################################################################################################
# WINE QUALITY CLASS
# create a list of our conditions
tiers = [
    (df['quality'] <= 4),
    (df['quality'] > 4) & (df['quality'] <= 6),
    (df['quality'] > 6) & (df['quality'] <= 10)
    ]


###########################################################################################
#sns.pairplot(df, palette = 'rocket') #WARNING COMPUTING TIME
# plt.title('Pairplot ')
plt.show()

j=1
for r in rows[:-1]:
    plt.subplot(2,6,j)
    sns.barplot(data=df, x='quality', y=r)
    plt.title(r + ' with quality barplot', size = 8)
    #plt.legend()
    j += 1
plt.show()

# from IPython.display import display
print(df.describe())
#import dataframe_image as dfi


# sns.heatmap(df.describe(),annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')
# plt.show()
# display(df)

#####################################################################################################################
# STATISTICAL TESTS
PH = np.array(df['pH'])#/(np.max(df['pH']))
# PH = (PH - np.mean(PH))/(np.var(PH))**(1/2)
print(PH)
print(stats.anderson(df['pH'], dist = 'norm'))
# print(stats.ks_1samp(df['pH'], stats.t.cdf(x = 7, df = 10)))

for r in rows:
    print('p-valeur ' + r + ' : ', stats.shapiro(df[r])[1])


#####################################################################################################################
# PIE PLOT BY QUALITY
total_count = len(df)
tiers_count = np.array([len(df[tiers[k]]) for k in range(3)])
pie_data = tiers_count/total_count
pie_labels = [str(tiers_count[0]) +' bad qualtiy wine ', str(tiers_count[1])+' average quality wine',str(tiers_count[2])+' good quality wine']
pie_colors = ['salmon', 'mediumturquoise', 'palegreen']
plt.pie(pie_data, labels= pie_labels, colors = pie_colors, autopct='%.0f%%')
plt.title('Wine quality pie plot (' + str(total_count) + ')')
plt.show()


#####################################################################################################################
# DISPLAY HISTOGRAMS
j=1
for r in rows:
    plt.subplot(3,4,j)
    sns.histplot(df[r], bins=50, kde=True, color='blue', stat='density', alpha=0.2)
    plt.title(r + ' Histogramme', size = 8)
    j += 1
plt.show()

sns.histplot(df['pH'], bins=50, kde=True, color='blue', stat='density', alpha=0.2)
sns.kdeplot(stats.norm.rvs(loc = 3.188267, scale = 0.151001, size = 10000))
plt.title('test normal distribution on pH')
plt.show()

#####################################################################################################################
# DISPLAY HISTOGRAMS SEPARATED BY TIER QUALITY
j=1
for r in rows[:-1]:
    plt.subplot(4,3,j)
    sns.histplot(df[tiers[0]][r],kde = True, color = 'salmon', alpha = 0.2, stat = 'density', element= 'step', bins = 20)
    sns.histplot(df[tiers[1]][r], kde=True, color='mediumturquoise', alpha=0.2, stat = 'density', element= 'step', bins = 20)
    sns.histplot(df[tiers[2]][r],kde = True,  color='palegreen', alpha=0.2, stat = 'density', element= 'step', bins = 20)
    plt.title(r + ' Histogramme', size = 8)
    #plt.legend()
    j += 1
plt.show()


#####################################################################################################################
# DISPLAY BOXPLOTS BY QUALITY
j=1

for r in rows[:-1]:
    plt.subplot(3,4,j)
    plt.title(r + ' boxplot', size = 8)
    # plt.xticks(np.arange(7),labels = [ str(k) for k in range(3,10)])
    box = plt.boxplot([df[ df['quality']==t][r]for t in range(3,10)],0, sym ='',patch_artist= True, )
    colors = ['darkred', 'darkgreen', 'darkblue']
    colors = sns.color_palette('RdBu')
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    j += 1
plt.show()





#####################################################################################################################
# DISPLAY CORRELATION MATRIX
corr_mat = df.corr()
sns.heatmap(corr_mat,annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')
plt.show()

sns.heatmap(df.corr()[['quality']].sort_values(by='quality', ascending=False),annot=True, cmap='YlGnBu')
plt.title('Descending Correlation with quality',pad=20, fontsize=16)
plt.show()
sns.heatmap(df.corr()[['alcohol']].sort_values(by='alcohol', ascending=False),annot=True, cmap='YlGnBu')
plt.title('Descending Correlation with alcohol',pad=20, fontsize=16)
plt.show()
#####################################################################################################################
# DISPLAY SCATTER MATRIX
pd.plotting.scatter_matrix(df)
plt.show()

#####################################################################################################################
# ACP
n = df.shape[0]
p = df.shape[1]
n_cp = p

norm = StandardScaler()
df_acp_norm = norm.fit_transform(df)
acp = decomposition.PCA(svd_solver='full', n_components=p)
coord = acp.fit_transform(df_acp_norm)

#####################################################################################################################
# HEATMAP CORRELATION WITH PRINCIPAL COMPONENTS
corr_mat_acp = pd.DataFrame(np.array([[np.corrcoef(df[c],coord[:,n])[1,0]
               for n in range(acp.n_components_)] for c in df]), columns = ['CP'+ str(k) for k in range(12)], index = df.columns)
sns.heatmap(corr_mat_acp,annot=True,linewidths=0.5,fmt='.2f',cmap='YlGnBu')
plt.title('Correlation with principal components',pad=20, fontsize=16)
plt.show()
print(corr_mat_acp)

#####################################################################################################################
val_prop = (n-1)/n * acp.explained_variance_
part_inertie_expl = acp.explained_variance_ratio_

# EXPLAINED VARIANCE BY EIGEN VALUE
plt.subplots(figsize=(8, 6))
plt.bar(np.arange(1, p+1), val_prop)
#plt.grid()
plt.title('Eboulis des valeurs propres')
plt.xlabel('Composante principale')
plt.ylabel('Valeur propre')
plt.show()


# CUMULATIVE EXPLAINED VARIANCE
plt.subplots(figsize=(8, 6))
print(np.cumsum(part_inertie_expl))
plt.bar(np.arange(1, p+1), np.cumsum(part_inertie_expl))
#plt.grid()
plt.title("Part d'inertie expliquée cumulée")
plt.xlabel('Composante principale')
plt.ylabel('Pourcentage')
plt.show()

#####################################################################################################################
# ACP - CORRELATION CIRCLE
a,b = 2,6 #plan factoriel considéré


sqrt_val_prop = np.sqrt(val_prop)
cor_var = np.zeros((p,n_cp))
for i in range(n_cp):
    cor_var[:, i] = acp.components_[i, :] * sqrt_val_prop[i]


fig, ax = plt.subplots(figsize=(10, 10))


for i in range(0, p):
    ax.arrow(0,
             0,
             cor_var[i, a],
             cor_var[i, b],
             head_width=0.03,
             head_length=0.03,
             length_includes_head=True)

    plt.text(cor_var[i, a]+0.01,
             cor_var[i, b],
             df.columns.values[i],
            c='red')

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Cercle de corrélations')
plt.xlabel('CP' + str(a+1))
plt.ylabel('CP' + str(b+1))
plt.axhline(y=0)
plt.axvline(x=0)

plt.show()

#=======================
# PROJECTION OF INDIVIDUAL


for i in range(0, n_cp):
    df['CP' + str(i + 1)] = coord[:, i]


Colors = ['red' for k in range(n)]
for k in range(n):
    if df.at[k, 'quality'] > 6:
        Colors[k] = 'blue'
    elif df.at[k, 'quality'] >= 5:
        Colors[k] = 'orange'

for i in range(1,n_cp+1):
    for j in range(1,n_cp+1):

        plt.subplot(n_cp,n_cp,(i-1)*n_cp+j)
        plt.scatter(df['CP'+str(j)], df['CP'+str(i)], c=Colors, s = 0.5)
        if j == 1:
            plt.ylabel('CP'+str(i), size = 15)
        if i == n_cp:
            plt.xlabel('CP' + str(j), size=15)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)

plt.show()

n_cp_kept = 3
for i in range(1,n_cp_kept+1):
    for j in range(1,n_cp_kept+1):

        plt.subplot(n_cp_kept,n_cp_kept,(i-1)*n_cp_kept+j)
        plt.scatter(df['CP'+str(j)], df['CP'+str(i)], c=Colors, s = 2)
        if j == 1:
            plt.ylabel('CP'+str(i), size = 15)
        if i == n_cp:
            plt.xlabel('CP' + str(j), size=15)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)

plt.show()

#####################################################################################################################
# CLUSTERING
# kmeans = KMeans(init='k-means++',
#                 max_iter=1000,
#                 n_clusters=3,
#                 n_init=20).fit(df2)
#
# df2['Classes_Kmeans'] = kmeans.labels_
#
# df2.sort_values('Classes_Kmeans')['Classes_Kmeans']
#
## print(df2.loc[df2['Classes_Kmeans']==1,'Classes_Kmeans'])
# result_clustering = np.array(df2)
# for k in range(len(df2)):
#     print('eau n°',k,' : ', result_clustering[k,-1])
#
#
# K = [k for k in range(1,len(df2)+1)]
# Colors = [('blue' if df.at[k,'Potability'] == 1.0 else 'red') for k in range(n)]
# plt.bar(K,result_clustering[:,-1]+1, color = Colors )
# # plt.xticks([ 2*k for k in range(1,43)])
# plt.yticks([])
# plt.show()


#=========================================================================================================
# CLUSTERING CAH

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

n = df.shape[0]

cah_ward = AgglomerativeClustering(distance_threshold=0,
                                   affinity='euclidean',
                                   linkage='ward',
                                   n_clusters=None).fit(df)

# plot the top three levels of the dendrogram
plot_dendrogram(cah_ward, truncate_mode="level", p=10)
plt.show()

plt.vlines(np.arange(2, n+1), 0, np.flip(np.sort(cah_ward.distances_)), linewidth=10)
plt.grid()
plt.title("Gain d'inertie inter-classes lors du passage de (k-1) à k classes")
plt.xlabel("k")
plt.ylabel("Indice d'aggrégation")
plt.show()




