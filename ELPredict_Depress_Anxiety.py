# -*- coding: utf-8 -*-

###########################################################################################################
#imports

import pandas as pd
import numpy as np
import os
import scikitplot
import networkx as nx
import csv
import gensim
import eli5
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from ast import literal_eval
from node2vec import Node2Vec
from liwcUtils import readLiwcDic, makeXliwc
from sklearn.ensemble import StackingClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.contingency_tables import mcnemar



###########################################################################################################
#Reading corpus and network data

def read_corpus_network(rootpath, diag_type):
    #train
    df_diag_train = pd.read_csv(rootpath + '/versao_full/train/' + 'train_' + diag_type + '_SetembroBR_v6.csv', sep=';', encoding='utf-8-sig', dtype={'User_ID': str,'Diagnosed_YN':str,'Text':str})
    df_ctrl_train = pd.read_csv(rootpath + '/versao_full/train/' + 'train_' + diag_type + '_c_SetembroBR_v6.csv', sep=';', encoding='utf-8-sig', dtype={'User_ID': str,'Diagnosed_YN':str,'Text':str})
    df_training   = df_diag_train.append(df_ctrl_train, ignore_index=True)
    df_training   = class_binary_01(df_training) #creating binary class collumn
    
    #test
    df_diag_test = pd.read_csv(rootpath + '/versao_full/test/' + 'test_' + diag_type + '_SetembroBR_v6.csv', sep=';', encoding='utf-8-sig', dtype={'User_ID': str,'Diagnosed_YN':str,'Text':str})
    df_ctrl_test = pd.read_csv(rootpath + '/versao_full/test/' + 'test_' + diag_type + '_c_SetembroBR_v6.csv', sep=';', encoding='utf-8-sig', dtype={'User_ID': str,'Diagnosed_YN':str,'Text':str})
    df_testing   = df_diag_test.append(df_ctrl_test, ignore_index=True)
    df_testing   = class_binary_01(df_testing) #creating binary class collumn
    
    #network
    df_diag_network = pd.read_csv(rootpath + '/versao_full/network_data/' + diag_type + '_diag-users-network_final_v6.csv', header=0, sep=';', converters={'Timeline': literal_eval, 'TopMentions':literal_eval, 'Followers_Anon': literal_eval, 'Friends_Anon':literal_eval, 'Contacts_Anon':literal_eval}) 
    df_ctrl_network = pd.read_csv(rootpath + '/versao_full/network_data/' + diag_type + '_ctrl-users-network_final_v6.csv', header=0, sep=';', converters={'Contacts_Anon':literal_eval, 'TopMentions':literal_eval})
    df_ctrl_network['Timeline'] = df_ctrl_network['Timeline'].fillna('[0]').apply(literal_eval)
    df_ctrl_network['Followers_Anon'] = df_ctrl_network['Followers_Anon'].fillna('[0]').apply(literal_eval)
    df_ctrl_network['Friends_Anon'] = df_ctrl_network['Friends_Anon'].fillna('[0]').apply(literal_eval)
    df_network = df_diag_network.append(df_ctrl_network, ignore_index=True) #append diag and ctrl for network dataset
    df_network = pd.get_dummies(df_network, columns=["Gender"], prefix=["Gender"]) #changing gender collumn to numeric format
    df_network = df_network[['User_ID', 'Gender_f', 'Gender_m', 'Statuses', 'N_Friends', 'N_Followers', 'N_Contacts', 'Followers_Anon', 'Friends_Anon', 'Contacts_Anon', 'TopMentions', 'hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23']] #selecting the features that will be used

    return df_training, df_testing, df_network

###########################################################################################################
#creating binary class collumn

def class_binary_01(df):
    df['Class'] = 0 #setting 0 initially for all instances
    for i, text in enumerate(df["Diagnosed_YN"]):
        if(text == 'yes'):
            df.loc[i, "Class"]=1
    return df


###########################################################################################################
#creating users' graph

def get_graph(df_full, graph_type, diag_type):

    G_users = nx.Graph()

    if graph_type == 'following':
        field = "Friends_Anon"
    elif graph_type == 'follower':
        field = "Followers_Anon"
    elif graph_type == 'mention':
        field = "Contacts_Anon"
    elif graph_type == 'topmention':
        field = "TopMentions"
        
    
    for i, user in enumerate(df_full["User_ID"]):
        for j, relationship in enumerate(df_full[field][i]):
            if relationship==0:
                G_users.add_node(user)
            elif relationship=='*others*' and j==0:
                G_users.add_node(user)
                break
            elif relationship=='*others*':
                break
            elif graph_type == 'topmention':
                G_users.add_edge(user, relationship, weight=20-j)
            else:
                G_users.add_edge(user, relationship)
    
    users_list = []
    for node in G_users.nodes():
        d_relationships = [idx for idx in list(G_users.adj[node]) if str(idx).startswith(diag_type)]
        c_relationships = [idx for idx in list(G_users.adj[node]) if str(idx).startswith('C')]
        if len(d_relationships)==0 and len(c_relationships)==0: #diag and control users --> there are not edges to users of the same type
            continue 
        else:        
            users_list.append([node, len(d_relationships), len(c_relationships), len(d_relationships) - len(c_relationships)])
    
    print("Number of nodes: " + str(G_users.number_of_nodes()))
    print("Number of edges: " + str(G_users.number_of_edges()))
    
    return G_users, users_list


###########################################################################################################
#Univariate Selection:
    
def get_univariate_selection(G_users, users_list, n_first_users):

    G_users_aux = G_users.copy()   

    #ordering list by the strongest relationship with diagnosed users and select n first users
    users_list.sort(key=lambda x:x[1], reverse=True)
    d_top_users = np.array(users_list)[:n_first_users, 0]
    
    #ordering list by the strongest relationship with control users and select n first users
    users_list.sort(key=lambda x:x[2], reverse=True)
    c_top_users = np.array(users_list)[:n_first_users, 0] 
    
    #joing two lists 
    top_users = set(np.append(c_top_users, d_top_users)) #using "set" to remove duplicities
    
    #selecting users to be removed
    users_list_np = np.array(users_list)[:, 0] #transforming list in a numpy array to select first collumn
    remove_users = set(users_list_np.tolist()) - top_users
    
    #Removing nodes from the graph  
    G_users_aux.remove_nodes_from(remove_users)
    
    #creating an adjacency matrix
    df_users_matrix = nx.to_pandas_adjacency(G_users_aux, dtype=int)
    df_users_matrix = df_users_matrix.drop(df_users_matrix.filter(regex='[ADC]', axis=1).columns, axis=1) #A - Anxiety | D - Depression | C - Control --> removing collumns that starts with A, D ou C

    print("Number of nodes: " + str(G_users.number_of_nodes()))
    print("Number of edges: " + str(G_users.number_of_edges()))
    print("Number of nodes US: " + str(G_users_aux.number_of_nodes()))
    print("Number of edges US: " + str(G_users_aux.number_of_edges()))
    print("Excluded nodes: " + str(len(remove_users)))
    print("Network relationship: " + str(len(df_users_matrix.columns)))
    
    return df_users_matrix


###########################################################################################################
#Node2vec:

def get_Node2vec(G_users, users_list, cut_value, location, dimensions, walk_length, num_walks, workers, window, min_count, batch_words):    

    G_users_aux = G_users
    if cut_value > 1:
        remove_users = np.array(list(filter(lambda c: (c[1] < cut_value and c[2] < cut_value), users_list)))[:, 0].tolist()
        G_users_aux.remove_nodes_from(remove_users) #Remove nodes from the graph   

    if os.path.exists(location):
        model = gensim.models.Word2Vec.load(location)
    else:    
        node2vec = Node2Vec(G_users_aux, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers, temp_folder=rootpath)  # Use temp_folder for big graphs
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        model.wv.save_word2vec_format(location.replace('_MODEL', '')) #Save embeddings for later use
        model.save(location) #Save model for later use

    node2vec_list=[]
    for node in G_users_aux.nodes():
        if str(node).startswith(diag_type) or str(node).startswith('C'):
            node_list = model.wv[node].tolist()
            node_list.insert(0, node)
            node2vec_list.append(node_list)
        
    df_node2vec = pd.DataFrame(node2vec_list)
    df_node2vec = df_node2vec.rename(columns={0: "User_ID"})
    df_node2vec = df_node2vec.set_index("User_ID")
    
    return df_node2vec

###########################################################################################################
#Joining Corpus and Adjacency Matrix:

def join_adj_matrix(df_training_full, df_testing_full, df_matrix):
    df_training_AM = df_training_full.set_index('User_ID').join(df_matrix)
    df_testing_AM = df_testing_full.set_index('User_ID').join(df_matrix)

    #return index User_ID as a collumn
    df_training_AM = df_training_AM.reset_index() 
    df_testing_AM = df_testing_AM.reset_index() 
    
    return df_training_AM, df_testing_AM


###########################################################################################################
#LIWC

def get_liwc(df_training_full, df_testing_full, rootpath, diag_type):

    if os.path.exists(rootpath + '/LIWC/' + 'liwc_train_' + diag_type + '.csv') and os.path.exists(rootpath + '/LIWC/' + 'liwc_test_' + diag_type + '.csv'):
        df_liwc_train = pd.read_csv(rootpath + '/LIWC/' + 'liwc_train_' + diag_type + '.csv', header=0, sep=',', converters={'LIWC': literal_eval})

        for i, user in enumerate(df_liwc_train["User_ID"]):
            for j, feature in enumerate(df_liwc_train["LIWC"][i]):
                df_liwc_train.loc[i, j] = df_liwc_train["LIWC"][i][j]        

        df_liwc_train = df_liwc_train.drop(columns=['LIWC', 'User_ID'])      
        
       
        df_liwc_test = pd.read_csv(rootpath + '/LIWC/' + 'liwc_test_' + diag_type + '.csv', header=0, sep=',', converters={'LIWC': literal_eval})

        for i, user in enumerate(df_liwc_test["User_ID"]):
            for j, feature in enumerate(df_liwc_test["LIWC"][i]):
                df_liwc_test.loc[i, j] = df_liwc_test["LIWC"][i][j]        

        df_liwc_test = df_liwc_test.drop(columns=['LIWC', 'User_ID'])

    else:
        liwcDic = {}
        liwcDic = readLiwcDic()
        
        train_text_column = df_training_full.loc[:,'Text'].str.lower()
        X_train_liwc = makeXliwc(liwcDic, train_text_column.values)
        df_liwc_train = pd.DataFrame(X_train_liwc)
        
        testing_text_column = df_testing_full.loc[:,'Text'].str.lower()
        X_test_liwc = makeXliwc(liwcDic, testing_text_column.values)
        df_liwc_test = pd.DataFrame(X_test_liwc)
        
        
        # Transforma córpus em arquivo LIWC para exportação (train)
        df_liwc_train_exp = df_training_full[['User_ID']] #seleciona apenas coluna de ID
        dict_liwc_train = { 'LIWC': pd.Series(X_train_liwc.tolist())} #transforma categorias LIWC em Series e dá nome à coluna
        df_liwc_train_exp = df_liwc_train_exp.join(pd.DataFrame(dict_liwc_train)) #equivale a df_liwc_train['LIWC'] = X_train_liwcI.tolist()
        df_liwc_train_exp.to_csv(rootpath + '/LIWC/' + 'liwc_train_' + diag_type + '.csv', index=False) #salva em csv (train)
    
           
        # Transforma córpus em arquivo LIWC para exportação (test)
        df_liwc_test_exp = df_testing_full[['User_ID']] #seleciona apenas coluna de ID
        dict_liwc_test = { 'LIWC': pd.Series(X_test_liwc.tolist())} #transforma categorias LIWC em Series e dá nome à coluna
        df_liwc_test_exp = df_liwc_test_exp.join(pd.DataFrame(dict_liwc_test)) #equivale a df_liwc_train['LIWC'] = X_train_liwcI.tolist()
        df_liwc_test_exp.to_csv(rootpath + '/LIWC/' + 'liwc_test_' + diag_type + '.csv', index=False) #salva em csv (test)
    
    return df_liwc_train, df_liwc_test
    

#############################################################################################################
#evaluating classifier by common metrics

def evaluate_classifier(y, y_preds):
    
    precision = precision_score(y.ravel(), y_preds, average=None)
    recall = recall_score(y.ravel(), y_preds, average=None)
    f1 = f1_score(y.ravel(), y_preds, average=None)    
    support = precision_recall_fscore_support(y.ravel(), y_preds, average=None)[3]
    
    accuracy = accuracy_score(y.ravel(), y_preds)
    
    precision_macro = precision_score(y.ravel(), y_preds, average='macro')
    recall_macro = recall_score(y.ravel(), y_preds, average='macro')
    f1_macro = f1_score(y.ravel(), y_preds, average='macro')
    
    precision_wtd = precision_score(y.ravel(), y_preds, average='weighted')
    recall_wtd = recall_score(y.ravel(), y_preds, average='weighted')
    f1_wtd = f1_score(y.ravel(), y_preds, average='weighted')
    
    #show metrics
    print(classification_report(y.ravel(), y_preds))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1-Score: {}'.format(f1))
    #print('Accuracy: {}'.format(accuracy))
    #print('F1-Score (Micro): {}'.format(f1_micro))
    #print('F1-Score (Macro): {}'.format(f1_macro))
    #print('F1-Score (Weighted): {}'.format(f1_wtd))
    print()
    
    metrics = []
    metrics.append([0, precision[0], recall[0], f1[0], support[0]])
    metrics.append([1, precision[1], recall[1], f1[1], support[1]])
    metrics.append(['accuracy', '', '', accuracy, support[0] + support[1]])
    metrics.append(['macro', precision_macro, recall_macro, f1_macro, support[0] + support[1]])
    metrics.append(['wtd', precision_wtd, recall_wtd, f1_wtd, support[0] + support[1]])
    with open(r"/home/rafael/Documentos/USP/corpus_v6/metrics.csv", 'w') as f:
        write = csv.writer(f) 
        write.writerow(["class", "precision", "recall", "f1", "support"])
        write.writerows(metrics)
    
    return precision, recall, f1


############################################################################################################
#SVM

def evaluate_SVC(X_train, y_train, X_test, C, kernel):  
    
    clf = SVC(gamma='auto', kernel=kernel, probability=True, C=C, class_weight='balanced')       
    clf.fit(X_train, y_train.ravel())
    y_pred=clf.predict(X_test)
    y_prob=clf.predict_proba(X_test)[:,1] #probability vector 
    print('SVC')
    
    return y_pred, y_prob


############################################################################################################
#Logistic Regression

def evaluate_RL(X_train, y_train, X_test):
    clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400).fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    print('Logistic Regression')    
    
    return y_pred, y_prob 

############################################################################################################
#MLP

def evaluate_MLP(X_train, y_train, X_test, solver, activation):
    clf = MLPClassifier(random_state=0, max_iter=400, solver=solver, activation=activation).fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    print('MLP')    
    
    return y_pred, y_prob 


############################################################################################################
#Dummy Classifier

def evaluate_Dummy(X_train, y_train, X_test, y_test):
    clf = DummyClassifier(random_state=0, strategy='most_frequent').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob=clf.predict_proba(X_test)
    print('Dummy Clissifier')    
    precision, recall, f1 = evaluate_classifier(y_test, y_pred)
    scikitplot.metrics.plot_confusion_matrix(y_test, y_pred) 
    
    return y_pred, y_prob 


############################################################################################################
#K-best selection - defining K

def K_definition_kbs(kvalue, d, X_train, y_train, X_test, y_test, file_csv):
    kbs_results = []
        
    while kvalue > 0:
        print("TR antes da selecao: " + str(X_train.shape))
        best_sel = SelectKBest(k=kvalue)
        best_fit = best_sel.fit(X_train, y_train.ravel())
        x_train_best = best_fit.transform(X_train)
        x_test_best = best_fit.transform(X_test)
        print("TR depois da selecao: " + str(x_train_best.shape))
        print("kvalue: " + str(kvalue)) 
        
        #LR Hold-out
        y_pred_RL, y_prob_RL = evaluate_RL(x_train_best, y_train, x_test_best)
        precision_kbs, recall_kbs, f1_kbs  = evaluate_classifier(y_test, y_pred_RL)
        
        kbs_results.append([kvalue, precision_kbs[0], recall_kbs[0], f1_kbs[0], precision_kbs[1], recall_kbs[1], f1_kbs[1]])
        
        kvalue = kvalue - d
        
    with open(file_csv, 'w') as f:          
        write = csv.writer(f) 
        write.writerow(["kvalue", "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1"])
        write.writerows(kbs_results)

############################################################################################################
#K-best selection 

def k_best_selection(kvalue, X_train, y_train, X_test, df_training_full):

    print("TR antes da selecao: " + str(X_train.shape))
    best_sel = SelectKBest(k=kvalue)
    best_fit = best_sel.fit(X_train, y_train.ravel())
    x_train_best = best_fit.transform(X_train)
    x_test_best = best_fit.transform(X_test)
    print("TR depois da selecao: " + str(x_train_best.shape))
    
    X_indices = best_fit.get_support(indices=True)
    nw_users_train = df_training_full.drop(columns=['User_ID', 'Text', 'Class', 'Gender_f', 'Gender_m', 'Statuses', 'N_Friends', 'N_Followers', 'N_Contacts', 'Followers_Anon', 'Friends_Anon', 'Contacts_Anon', 'TopMentions']).columns
    nw_users_train_kbs = nw_users_train[X_indices]
    nw_users_train_kbs_list = list(map(str, nw_users_train_kbs.tolist())) #creating list by converting int values in string values

    return x_train_best, x_test_best, nw_users_train_kbs_list


############################################################################################################
#Eli5 - explainning model

def explain_eli5(X_train, y_train, url, nw_users_train):
    clf = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
    clf.fit(X_train, y_train)
    weights = eli5.show_weights(clf, feature_names = nw_users_train, target_names=['ctrl', 'diag'], top=(20, 20)) #global explanation
    with open(url,'wb') as f:  
        f.write(weights.data.encode("UTF-8"))


###########################################################################################################
#ensembles
def ensemble(method, X_train, y_train, X_test, n_sets):

    # Base estimator 1 uses the first 64 features:
    #clf_1 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
    clf_1 = SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced')
    clf_1_transformer = FunctionTransformer(lambda X_train: X_train[:, :64])  
    tclf_1 = Pipeline([('transformer_1', clf_1_transformer), ('clf_1', clf_1)])
    
    if n_sets > 1:
        # Base estimator 2 uses the next 64 features:
        #clf_2 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)  
        clf_2 = SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced')        
        clf_2_transformer = FunctionTransformer(lambda X_train: X_train[:, 64:128])
        tclf_2 = Pipeline([('transformer_2', clf_2_transformer), ('clf_2', clf_2)])
    
    if n_sets > 2:
        # Base estimator 3 uses the next 64 features:
        #clf_3 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
        clf_3 = SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced')       
        clf_3_transformer = FunctionTransformer(lambda X_train: X_train[:, 128:192])
        tclf_3 = Pipeline([('transformer_3', clf_3_transformer), ('clf_3', clf_3)])
 
    
    if n_sets > 3:
        # Base estimator 4 uses the next 64 features:
        #clf_4 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
        clf_4 = SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced')       
        clf_4_transformer = FunctionTransformer(lambda X_train: X_train[:, 192:256])
        tclf_4 = Pipeline([('transformer_4', clf_4_transformer), ('clf_4', clf_4)])
        
        
    if n_sets > 4:
        # Base estimator 5 uses the next 24 features (Hour):
        #clf_5 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
        clf_5 = SVC(gamma='auto', kernel='linear', probability=True, C=1, class_weight='balanced')       
        clf_5_transformer = FunctionTransformer(lambda X_train: X_train[:, 256:280])
        tclf_5 = Pipeline([('transformer_5', clf_5_transformer), ('clf_5', clf_5)])
        
    if n_sets > 5:
        # Base estimator 6 uses the next 64 features (LIWC):
        clf_6 = LogisticRegression(random_state=0, class_weight='balanced', max_iter=400)
        #clf_5 = SVC(gamma='auto', kernel='linear', probability=True, C=1, class_weight='balanced')       
        clf_6_transformer = FunctionTransformer(lambda X_train: X_train[:, 280:344])
        tclf_6 = Pipeline([('transformer_6', clf_6_transformer), ('clf_6', clf_6)])
    
        
    # The meta-learner uses the transformed-classifiers as base estimators:
    if method == 'Stacking':
       if n_sets > 5:
            sclf = StackingClassifier([('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3), ('tclf_4', tclf_4), ('tclf_5', tclf_5), ('tclf_6', tclf_6)], final_estimator=SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced'), passthrough=False, stack_method='predict_proba')
       if n_sets > 4:
            sclf = StackingClassifier([('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3), ('tclf_4', tclf_4), ('tclf_5', tclf_5)], final_estimator=SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced'), passthrough=False, stack_method='predict_proba')
       elif n_sets > 3:
            sclf = StackingClassifier([('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3), ('tclf_4', tclf_4)], final_estimator=SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced'), passthrough=False, stack_method='predict_proba')
       elif n_sets > 2:
            sclf = StackingClassifier([('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3)], final_estimator=SVC(gamma='auto', kernel='rbf', probability=True, C=1, class_weight='balanced'), passthrough=False, stack_method='predict_proba') 
    elif method == 'Voting':
        if n_sets > 5:
            sclf = VotingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3),  ('tclf_4', tclf_4), ('tclf_5', tclf_5), ('tclf_6', tclf_6)], voting='hard')
        if n_sets > 4:
            sclf = VotingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3),  ('tclf_4', tclf_4), ('tclf_5', tclf_5)], voting='hard')
        elif n_sets > 3:
            sclf = VotingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3),  ('tclf_4', tclf_4)], voting='hard')
        elif n_sets > 2:
            sclf = VotingClassifier(estimators=[('tclf_1', tclf_1), ('tclf_2', tclf_2), ('tclf_3', tclf_3)], voting='hard')
    elif method == 'GradientBoosting':
        sclf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    elif method == 'AdaBoost':
        sclf = AdaBoostClassifier(n_estimators=100, random_state=0, algorithm='SAMME')
    
    
    sclf.fit(X_train, y_train)
    y_pred = sclf.predict(X_test)
    
    return y_pred
    


############################################################################################################
#Statistical analysis

def main_mcnemar(pred_csv1, pred_csv2, diag_type, classes, rootpath):

    # IMPORTANTE: O ARQUIVO NAO EH A PREDICAO DE CLASSE EM SI, MAS APENAS O MATCH (1=CERTOU, 0=ERROU)
    if classes == 'all':
        pre = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv1, sep=';', header=None)
        pos = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv2, sep=';', header=None)
    if classes == 'diag' and diag_type == 'D':
        pre = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv1, sep=';', nrows=337, header=None)
        pos = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv2, sep=';', nrows=337, header=None)
    if classes == 'diag' and diag_type == 'A':
        pre = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv1, sep=';', nrows=444, header=None)
        pos = pd.read_csv(rootpath + '/Exports/teste_McNemar/' + pred_csv2, sep=';', nrows=444, header=None)

    # cria tabela de contingencia
    df = pd.concat([pre,pos],axis=1) # lado
    df.columns = ['pre','pos']
    df['right_right'] = df.apply(lambda x: int(x.pre==1 and x.pos==1),axis=1)
    df['right_wrong'] = df.apply(lambda x: int(x.pre==1 and x.pos==0),axis=1)
    df['wrong_right'] = df.apply(lambda x: int(x.pre==0 and x.pos==1),axis=1)
    df['wrong_wrong'] = df.apply(lambda x: int(x.pre==0 and x.pos==0),axis=1)
    cell_11 = df.right_right.sum()
    cell_12 = df.right_wrong.sum()
    cell_21 = df.wrong_right.sum()
    cell_22 = df.wrong_wrong.sum()
    table = [
             [cell_11, cell_12],
             [cell_21, cell_22] 
            ]
                
    if cell_11<25 or cell_12<25 or cell_21<25 or cell_22<25:
        result = mcnemar(table, exact=True)
    else:
        result = mcnemar(table, exact=False, correction=True)   

    print('stat=%.3f, p=%.7f' % (result.statistic, result.pvalue),end='')

    significant = False
    for alpha in [0.001, 0.01, 0.05]:
        if result.pvalue < alpha:
            print(' p<' + str(alpha))
            significant = True
            break
    if not significant:
        print(' not significant')
        
###########################################################################################################
#Export to csv

def export_csv(array, file_csv, rootpath):
     df = pd.DataFrame(array.tolist())
     df.to_csv(rootpath + '/Exports/' + file_csv, index=False, header=False, sep=';')


###########################################################################################################
#main

#parameters
rootpath  = '/home/rafael/Documentos/USP/corpus_v6'
os.chdir(rootpath)
diag_type = 'D' # A - Anxiety | D - Depression
network_type = 'mention' # following | follower | mention | topmention | all | none
features_type = 'US' # N2V - Node2Vec | US - Univariate Selection | none
flag_KBS = 'Y' # Y - Yes | N - No
flag_LIWC = 'N' # Y - Yes | N - No
flag_hour = 'N' # Y - Yes | N - No
classifier = 'LR' # LR - Logistic Regression | SVM - Support Vector Machines | MLP - Multi-layer Perceptron | Ensemble (Only make sense when network_type = all)


#reading corpus and network data
df_training, df_testing, df_network = read_corpus_network(rootpath, diag_type)

#joining corpus and network data
df_training_full = pd.merge(df_training[['User_ID', 'Text', 'Class']], df_network, how="left", on='User_ID')
df_testing_full = pd.merge(df_testing[['User_ID', 'Text', 'Class']], df_network, how="left", on='User_ID')
df_full = df_training_full.append(df_testing_full, ignore_index=True) #dataset full 

#creating graphs and lists
if network_type == 'following' or network_type == 'all':
    G_following, list_following = get_graph(df_full, 'following', diag_type)
if network_type == 'follower' or network_type == 'all':
    G_follower, list_follower = get_graph(df_full, 'follower', diag_type)
if network_type == 'mention' or network_type == 'all':
    G_mention, list_mention = get_graph(df_full, 'mention', diag_type)
if network_type == 'topmention' or network_type == 'all':
    G_topmention, list_topmention = get_graph(df_full, 'topmention', diag_type)
    
    
#network descriptive data extraction (list_following, list_follower, list_mention, list_topmention)
#list_follower.sort(key=lambda x:x[1], reverse=True)
#d_top_users = np.array(list_follower)[[0,1,2,3,4,5,6,7,8,9,99,499,999,3499,4999,9999,14999,19999,24999], :3]
#list_follower.sort(key=lambda x:x[2], reverse=True)
#c_top_users = np.array(list_follower)[[0,1,2,3,4,5,6,7,8,9,99,499,999,3499,4999,9999,14999,19999,24999], :3]



#Network Features
if features_type == 'N2V' and diag_type == 'D':
    if network_type == 'following' or network_type == 'all':
        df_matrix_following = get_Node2vec(G_following, list_following, 10, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut10_d64_wl30_nw10_D_fllwing', 64, 30, 10, 1, 10, 1, 4)
    if network_type == 'follower' or network_type == 'all':
        df_matrix_follower = get_Node2vec(G_follower, list_follower, 10, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut10_d64_wl80_nw10_D_fllwer', 64, 80, 10, 1, 10, 1, 4)
    if network_type == 'mention' or network_type == 'all':
        df_matrix_mention = get_Node2vec(G_mention, list_mention, 2, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut2_d64_wl80_nw10_D_mention', 64, 80, 10, 1, 10, 1, 4)
    if network_type == 'topmention' or network_type == 'all':
        df_matrix_topmention = get_Node2vec(G_topmention, list_topmention, 0, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut0_d64_wl80_nw10_D_topmention', 64, 80, 10, 1, 10, 1, 4)
elif features_type == 'N2V'and diag_type == 'A':
    if network_type == 'following' or network_type == 'all':
        df_matrix_following = get_Node2vec(G_following, list_following, 30, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut30_d64_wl30_nw10_A_fllwing', 64, 30, 10, 1, 10, 1, 4)
    if network_type == 'follower' or network_type == 'all':
        df_matrix_follower = get_Node2vec(G_follower, list_follower, 15, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut15_d64_wl80_nw10_A_fllwer', 64, 80, 10, 1, 10, 1, 4)
    if network_type == 'mention' or network_type == 'all':
        df_matrix_mention = get_Node2vec(G_mention, list_mention, 2, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut2_d64_wl80_nw10_A_mention', 64, 80, 10, 1, 10, 1, 4)
    if network_type == 'topmention' or network_type == 'all':
        df_matrix_topmention = get_Node2vec(G_topmention, list_topmention, 0, rootpath + '/Node2Vec/' + 'EMBEDDING_MODEL_cut0_d64_wl80_nw10_A_topmention', 64, 80, 10, 1, 10, 1, 4)
elif features_type == 'US':
    if network_type == 'following' or network_type == 'all':
        df_matrix_following = get_univariate_selection(G_following, list_following, 15000)
    if network_type == 'follower' or network_type == 'all':
        df_matrix_follower = get_univariate_selection(G_follower, list_follower, 15000)
    if network_type == 'mention' or network_type == 'all':
        df_matrix_mention = get_univariate_selection(G_mention, list_mention, 15000)
    if network_type == 'topmention' or network_type == 'all':
        df_matrix_topmention = get_univariate_selection(G_topmention, list_topmention, 16651)

  


#defining the dataframe that will have the features set aggregated
df_training_full_ft = df_training_full
df_testing_full_ft = df_testing_full


#joining training and testing datasets with feature's matrix
if network_type == 'following':
    df_training_full_ft, df_testing_full_ft = join_adj_matrix(df_training_full_ft, df_testing_full_ft, df_matrix_following)
elif network_type == 'follower':
    df_training_full_ft, df_testing_full_ft = join_adj_matrix(df_training_full_ft, df_testing_full_ft, df_matrix_follower)
elif network_type == 'mention':
    df_training_full_ft, df_testing_full_ft = join_adj_matrix(df_training_full_ft, df_testing_full_ft, df_matrix_mention)
elif network_type == 'topmention':
    df_training_full_ft, df_testing_full_ft = join_adj_matrix(df_training_full_ft, df_testing_full_ft, df_matrix_topmention)
elif network_type == 'all':
    df_matrix_all = df_matrix_following.add_suffix('flwing').join(df_matrix_follower.add_suffix('flwer')).join(df_matrix_mention.add_suffix('mtion')).join(df_matrix_topmention.add_suffix('topmtion')) #following + follower + mention + topmention
    #df_matrix_all = df_matrix_following.add_suffix('flwing').join(df_matrix_follower.add_suffix('flwer')).join(df_matrix_mention.add_suffix('mtion')) #following + follower + mention
    df_training_full_ft, df_testing_full_ft = join_adj_matrix(df_training_full, df_testing_full, df_matrix_all)

    
#joining training and testing datasets with liwc features
if flag_LIWC == 'Y':
    df_liwc_train, df_liwc_test = get_liwc(df_training_full_ft, df_testing_full_ft, rootpath, diag_type)
    df_training_full_ft = df_training_full_ft.join(df_liwc_train)
    df_testing_full_ft = df_testing_full_ft.join(df_liwc_test)


if flag_hour == 'N':
    df_training_full_ft = df_training_full_ft.drop(columns=['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'])
    df_testing_full_ft = df_testing_full_ft.drop(columns=['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'])

    

#Defining X and Y
X_train_aux = df_training_full_ft.drop(columns=['User_ID', 'Text', 'Class', 'Gender_f', 'Gender_m', 'Statuses', 'N_Friends', 'N_Followers', 'N_Contacts', 'Followers_Anon', 'Friends_Anon', 'Contacts_Anon', 'TopMentions']).to_numpy()
X_test_aux  = df_testing_full_ft.drop(columns=['User_ID',  'Text', 'Class', 'Gender_f', 'Gender_m', 'Statuses', 'N_Friends', 'N_Followers', 'N_Contacts', 'Followers_Anon', 'Friends_Anon', 'Contacts_Anon',  'TopMentions']).to_numpy()
y_train = df_training_full_ft[['Class']].to_numpy() 
y_test  = df_testing_full_ft[['Class']].to_numpy() 


#k definition for K best selection
#K_definition_kbs(21000, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_following_HO80T_RL_DEPRESS.csv')
#K_definition_kbs(22500, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_follower_HO80T_RL_DEPRESS.csv')
#K_definition_kbs(23500, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_mention_HO80T_RL_DEPRESS.csv')
#K_definition_kbs(16500, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_topmention_HO80T_RL_DEPRESS.csv')
#K_definition_kbs(20500, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_following_HO80T_RL_ANX.csv')
#K_definition_kbs(21500, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_follower_HO80T_RL_ANX.csv')
#K_definition_kbs(23000, 500, X_train_aux, y_train, X_test_aux, y_test, rootpath + '/KBS/' + 'kbs_500_mention_HO80T_RL_ANX.csv')


#Definning X_train and X_test
if flag_KBS == 'Y' and features_type == 'US':    
    if network_type == 'following' and diag_type == 'D':
        #X_train, X_test, nw_users_train = k_best_selection(3500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
        X_train, X_test, nw_users_train = k_best_selection(14500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    elif network_type == 'following' and diag_type == 'A':
        #X_train, X_test, nw_users_train = k_best_selection(6500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
        X_train, X_test, nw_users_train = k_best_selection(17000, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    elif network_type == 'follower' and diag_type == 'D':
        #X_train, X_test, nw_users_train = k_best_selection(7500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
        X_train, X_test, nw_users_train = k_best_selection(13000, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    elif network_type == 'follower' and diag_type == 'A':
        #X_train, X_test, nw_users_train = k_best_selection(5000, X_train_aux, y_train, X_test_aux, df_training_full_ft)
        X_train, X_test, nw_users_train = k_best_selection(21000, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    elif network_type == 'mention'and diag_type == 'D':
        #X_train, X_test, nw_users_train = k_best_selection(6500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
        X_train, X_test, nw_users_train = k_best_selection(19500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    elif network_type == 'mention' and diag_type == 'A':
        X_train, X_test, nw_users_train = k_best_selection(10500, X_train_aux, y_train, X_test_aux, df_training_full_ft)
elif flag_KBS == 'Y' and features_type == 'N2V': 
    if network_type == 'all' and diag_type == 'D':
        X_train, X_test, nw_users_train = k_best_selection(114, X_train_aux, y_train, X_test_aux, df_training_full_ft)
    if network_type == 'all' and diag_type == 'A':
        X_train, X_test, nw_users_train = k_best_selection(158, X_train_aux, y_train, X_test_aux, df_training_full_ft)
else:
    X_train = X_train_aux
    X_test  = X_test_aux
    df_nw_users_train = df_training_full.drop(columns=['User_ID', 'Text', 'Class', 'Gender_f', 'Gender_m', 'Statuses', 'N_Friends', 'N_Followers', 'N_Contacts', 'Followers_Anon', 'Friends_Anon', 'Contacts_Anon', 'TopMentions']).columns
    nw_users_train = list(map(str, df_nw_users_train.tolist())) 


#classifier
if classifier == 'SVM':
    y_pred, y_prob = evaluate_SVC(X_train, y_train, X_test, 1, 'rbf')
elif classifier == 'LR':
    y_pred, y_prob = evaluate_RL(X_train, y_train, X_test)
elif classifier == 'MLP':
    y_pred, y_prob = evaluate_MLP(X_train, y_train, X_test, 'adam', 'tanh')
elif classifier == 'Ensemble':
    y_pred = ensemble('Stacking', X_train, y_train, X_test, 6) #method: Stacking, Voting, GradientBoosting, AdaBoost


#Evaluation of classifier
precision, recall, f1 = evaluate_classifier(y_test, y_pred)
scikitplot.metrics.plot_confusion_matrix(y_test.ravel(), y_pred) 


#export predictions to csv and compare with true labels for using in statistical analysis
export_csv(y_pred, 'y_pred_D_USLR19500_mention_teste.csv', rootpath)
export_csv(np.where(y_test.ravel() == y_pred, 1, 0), 'y_pred_D_USLR19500_mention_compare.csv', rootpath)


##Statistical analysis
main_mcnemar('y_pred_A_all_compare.csv', 'y_pred_A_ASMTL_compare.csv', diag_type, 'diag', rootpath) # 'all' | 'diag'

    
#explain features
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "friend_depress.htm", nw_users_train)
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "friend_anxiety.htm", nw_users_train)
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "follower_depress.htm", nw_users_train)
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "follower_anxiety.htm", nw_users_train)
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "mention_depress.htm", nw_users_train)
explain_eli5(X_train, y_train, rootpath + '/ELI5/' + "mention_anxiety.htm", nw_users_train)





